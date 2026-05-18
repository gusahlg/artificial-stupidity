use crate::embeddings::Embedding;
use crate::gpu::Gpu;
use crate::neural_network::{Activation, Layer, Network, input_size_for};
use anyhow::{Result, anyhow, bail};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

const MAGIC: u32 = 0x4D4F_444C; // "MODL"
const VERSION: u32 = 2;

fn write_u32<W: Write>(w: &mut W, v: u32) -> Result<()> {
    w.write_all(&v.to_le_bytes())?;
    Ok(())
}
fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn write_u8<W: Write>(w: &mut W, v: u8) -> Result<()> {
    w.write_all(&[v])?;
    Ok(())
}
fn read_u8<R: Read>(r: &mut R) -> Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}
fn write_f32_slice<W: Write>(w: &mut W, data: &[f32]) -> Result<()> {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    w.write_all(bytes)?;
    Ok(())
}
fn read_f32_vec<R: Read>(r: &mut R, n: usize) -> Result<Vec<f32>> {
    let mut buf = vec![0u8; n * 4];
    r.read_exact(&mut buf)?;
    let mut out = Vec::with_capacity(n);
    for chunk in buf.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn activation_code(a: Activation) -> u8 {
    match a {
        Activation::Tanh => 0,
        Activation::Linear => 1,
    }
}
fn activation_from_code(c: u8) -> Result<Activation> {
    match c {
        0 => Ok(Activation::Tanh),
        1 => Ok(Activation::Linear),
        _ => Err(anyhow!("unknown activation code: {c}")),
    }
}

pub fn save<P: AsRef<Path>>(net: &Network, path: P) -> Result<()> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);
    write_u32(&mut w, MAGIC)?;
    write_u32(&mut w, VERSION)?;
    write_u32(&mut w, net.embed_dim as u32)?;
    write_u32(&mut w, net.context_window as u32)?;
    write_u32(&mut w, net.vocab_size as u32)?;
    write_u32(&mut w, net.hidden_size as u32)?;
    write_u32(&mut w, net.hidden_layers as u32)?;

    // Embedding
    write_f32_slice(&mut w, &net.embedding.weights)?;

    // Dense layers
    write_u32(&mut w, net.layers.len() as u32)?;
    for layer in &net.layers {
        write_u32(&mut w, layer.rows as u32)?;
        write_u32(&mut w, layer.cols as u32)?;
        write_u8(&mut w, activation_code(layer.activation))?;
        write_f32_slice(&mut w, &layer.weights)?;
        write_f32_slice(&mut w, &layer.biases)?;
    }
    w.flush()?;
    Ok(())
}

pub struct LoadedShape {
    pub embed_dim: usize,
    pub context_window: usize,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub hidden_layers: usize,
}

/// Load a v2 model. Vocab growth (saved vocab < expected vocab) is handled by
/// extending the embedding table and the output layer with fresh random rows;
/// the rest of the structure must match exactly.
pub fn load<P: AsRef<Path>>(
    path: P,
    gpu: &Gpu,
    expected: LoadedShape,
) -> Result<Option<Network>> {
    let path = path.as_ref();
    if !path.exists() {
        return Ok(None);
    }
    let f = File::open(path)?;
    let mut r = BufReader::new(f);

    let magic = read_u32(&mut r)?;
    if magic != MAGIC {
        bail!("model file has wrong magic: 0x{:08x}", magic);
    }
    let version = read_u32(&mut r)?;
    if version != VERSION {
        bail!(
            "model file version {} not supported (expected {})",
            version,
            VERSION
        );
    }

    let embed_dim = read_u32(&mut r)? as usize;
    let context_window = read_u32(&mut r)? as usize;
    let saved_vocab_size = read_u32(&mut r)? as usize;
    let hidden_size = read_u32(&mut r)? as usize;
    let hidden_layers = read_u32(&mut r)? as usize;

    if embed_dim != expected.embed_dim
        || context_window != expected.context_window
        || hidden_size != expected.hidden_size
        || hidden_layers != expected.hidden_layers
    {
        bail!(
            "model shape mismatch: saved (embed={}, ctx={}, hidden={}x{}) vs expected (embed={}, ctx={}, hidden={}x{})",
            embed_dim,
            context_window,
            hidden_size,
            hidden_layers,
            expected.embed_dim,
            expected.context_window,
            expected.hidden_size,
            expected.hidden_layers,
        );
    }
    if saved_vocab_size > expected.vocab_size {
        bail!(
            "saved vocab ({}) is larger than current vocab ({}); refusing to truncate",
            saved_vocab_size,
            expected.vocab_size
        );
    }

    let embed_weights = read_f32_vec(&mut r, saved_vocab_size * embed_dim)?;
    let mut embedding = Embedding::from_parts(saved_vocab_size, embed_dim, embed_weights);

    let layer_count = read_u32(&mut r)? as usize;
    let mut layers = Vec::with_capacity(layer_count);
    for i in 0..layer_count {
        let rows = read_u32(&mut r)? as usize;
        let cols = read_u32(&mut r)? as usize;
        let act = activation_from_code(read_u8(&mut r)?)?;
        let weights = read_f32_vec(&mut r, rows * cols)?;
        let biases = read_f32_vec(&mut r, rows)?;
        // Sanity: the output (last) layer's rows should equal saved_vocab_size;
        // input (first) layer's cols should equal input_size_for(embed_dim, ctx).
        if i == 0 && cols != input_size_for(embed_dim, context_window) {
            bail!(
                "first layer cols ({}) does not match expected input size ({})",
                cols,
                input_size_for(embed_dim, context_window)
            );
        }
        if i == layer_count - 1 && rows != saved_vocab_size {
            bail!(
                "output layer rows ({}) does not match saved vocab ({})",
                rows,
                saved_vocab_size
            );
        }
        layers.push(Layer::from_parts(rows, cols, act, weights, biases, gpu)?);
    }

    // Grow embedding + output layer if the corpus has added new vocab entries.
    let mut rng = rand::thread_rng();
    if expected.vocab_size > saved_vocab_size {
        embedding.extend_to(expected.vocab_size, &mut rng);
        let last = layers.len() - 1;
        layers[last].extend_rows(expected.vocab_size, gpu, &mut rng)?;
        eprintln!(
            "Extended saved model from vocab {} to {} (new rows initialized randomly).",
            saved_vocab_size, expected.vocab_size
        );
    }

    Ok(Some(Network {
        embedding,
        layers,
        vocab_size: expected.vocab_size,
        hidden_size,
        hidden_layers,
        embed_dim,
        context_window,
        adam_step: 0,
        backprop_scratch: crate::teacher::BackpropScratch::default(),
    }))
}
