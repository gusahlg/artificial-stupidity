use std::sync::Arc;

use anyhow::Result;
use ml_project::{Executor, MatmulCall, MatmulPipeline, Tensor, VulkanContext};

pub struct Gpu {
    pub ctx: Arc<VulkanContext>,
    pub exec: Executor,
}

impl Gpu {
    pub fn new() -> Result<Self> {
        let ctx = VulkanContext::new(false)?;
        let pipeline = Arc::new(MatmulPipeline::new(&ctx)?);
        let exec = Executor::new(ctx.clone(), pipeline, 2, 8)?;
        Ok(Self { ctx, exec })
    }

    /// y = W @ x where W is [m, k] (row-major) and x is [k].
    /// `w`, `x_dev`, `y_dev` are pre-allocated device tensors.
    pub fn matvec(
        &self,
        w: &Tensor,
        x_dev: &Tensor,
        y_dev: &Tensor,
        x_host: &[f32],
        y_host: &mut [f32],
    ) -> Result<()> {
        self.exec.upload(x_host, x_dev)?;
        self.exec.run_matmuls(&[MatmulCall {
            a: w,
            b: x_dev,
            c: y_dev,
            alpha: 1.0,
            accumulate: false,
        }])?;
        self.exec.download(y_dev, y_host)?;
        Ok(())
    }
}
