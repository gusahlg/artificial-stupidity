use std::sync::Arc;

use anyhow::Result;
use ml_project::{Executor, MatmulCall, MatmulPipeline, Tensor, VulkanContext};

/// Compute backend. Vulkan when available; CPU as a portable fallback so the
/// project (and the auto-trainer) works on machines without a working Vulkan
/// loader. The CPU path is also faster for the tiny matvecs we run, because
/// GPU dispatch overhead dwarfs the actual math at these sizes.
pub enum Backend {
    Vulkan(VulkanBackend),
    Cpu,
}

pub struct VulkanBackend {
    pub ctx: Arc<VulkanContext>,
    pub exec: Executor,
}

pub struct Gpu {
    pub backend: Backend,
}

impl Gpu {
    /// Try Vulkan; fall back to CPU on any init failure (missing loader,
    /// missing device, etc.).
    pub fn new() -> Result<Self> {
        match Self::new_vulkan() {
            Ok(g) => Ok(g),
            Err(e) => {
                eprintln!("Vulkan unavailable ({e}); falling back to CPU backend.");
                Ok(Self {
                    backend: Backend::Cpu,
                })
            }
        }
    }

    pub fn new_vulkan() -> Result<Self> {
        let ctx = VulkanContext::new(false)?;
        let pipeline = Arc::new(MatmulPipeline::new(&ctx)?);
        let exec = Executor::new(ctx.clone(), pipeline, 2, 8)?;
        Ok(Self {
            backend: Backend::Vulkan(VulkanBackend { ctx, exec }),
        })
    }

    pub fn new_cpu() -> Self {
        Self {
            backend: Backend::Cpu,
        }
    }

    pub fn device_name(&self) -> String {
        match &self.backend {
            Backend::Vulkan(v) => v.ctx.device_name().to_string(),
            Backend::Cpu => "cpu".to_string(),
        }
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self.backend, Backend::Cpu)
    }
}

/// Pre-allocated GPU-side scratch for a single layer's matvec. Only present
/// for the Vulkan backend.
pub struct LayerGpu {
    pub gpu_weights: Tensor,
    pub gpu_input: Tensor,
    pub gpu_output: Tensor,
}

impl LayerGpu {
    pub fn new(v: &VulkanBackend, rows: usize, cols: usize) -> Result<Self> {
        let gpu_weights = Tensor::zeros_device(&v.ctx, &[rows as u32, cols as u32])?;
        let gpu_input = Tensor::zeros_device(&v.ctx, &[cols as u32, 1])?;
        let gpu_output = Tensor::zeros_device(&v.ctx, &[rows as u32, 1])?;
        Ok(Self {
            gpu_weights,
            gpu_input,
            gpu_output,
        })
    }
}

/// y = W @ x where W is [rows, cols] row-major. CPU path: straightforward
/// triple-loop, but treats x as effectively sparse so the (mostly-zero)
/// bag-of-words input layer doesn't waste cycles.
pub fn cpu_matvec(weights: &[f32], rows: usize, cols: usize, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(weights.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(y.len(), rows);
    for j in 0..rows {
        let row = &weights[j * cols..(j + 1) * cols];
        let mut acc = 0.0f32;
        for k in 0..cols {
            // Branch is predictable for the long zero runs in our BoW input.
            let v = x[k];
            if v != 0.0 {
                acc += row[k] * v;
            }
        }
        y[j] = acc;
    }
}

impl Gpu {
    /// y = W @ x. The Vulkan path uses pre-allocated device tensors and only
    /// re-uploads `weights` when the caller flips `weights_dirty` to true
    /// (i.e. after a training step). The CPU path ignores `layer_gpu`.
    pub fn matvec(
        &self,
        weights: &[f32],
        rows: usize,
        cols: usize,
        x_host: &[f32],
        y_host: &mut [f32],
        layer_gpu: Option<&LayerGpu>,
        weights_dirty: bool,
    ) -> Result<()> {
        match &self.backend {
            Backend::Vulkan(v) => {
                let lg = layer_gpu.expect("Vulkan backend requires LayerGpu");
                if weights_dirty {
                    v.exec.upload(weights, &lg.gpu_weights)?;
                }
                v.exec.upload(x_host, &lg.gpu_input)?;
                v.exec.run_matmuls(&[MatmulCall {
                    a: &lg.gpu_weights,
                    b: &lg.gpu_input,
                    c: &lg.gpu_output,
                    alpha: 1.0,
                    accumulate: false,
                }])?;
                v.exec.download(&lg.gpu_output, y_host)?;
                Ok(())
            }
            Backend::Cpu => {
                cpu_matvec(weights, rows, cols, x_host, y_host);
                Ok(())
            }
        }
    }
}
