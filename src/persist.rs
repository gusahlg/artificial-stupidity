use crate::gpu::Gpu;
use crate::neural_network::{Activation, Layer, Network};
use anyhow::{Result, anyhow, bail};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

const MAGIC: u32 = 0x4D4F_444C; // "MODL"
const VERSION: u32 = 1;

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
    // Reinterpret as bytes — host is little-endian on every platform we run on.
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
    write_u32(&mut w, net.input_size as u32)?;
    write_u32(&mut w, net.vocab_size as u32)?;
    write_u32(&mut w, net.hidden_size as u32)?;
    write_u32(&mut w, net.hidden_layers as u32)?;
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
    pub input_size: usize,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub hidden_layers: usize,
}

/// Attempt to load a model. Returns Ok(Some(net)) on success, Ok(None) if the
/// file is absent. Returns the shape mismatch info as an error so the caller
/// can decide to retrain from scratch.
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
        bail!("model file version {} not supported", version);
    }
    let input_size = read_u32(&mut r)? as usize;
    let vocab_size = read_u32(&mut r)? as usize;
    let hidden_size = read_u32(&mut r)? as usize;
    let hidden_layers = read_u32(&mut r)? as usize;
    let layer_count = read_u32(&mut r)? as usize;

    if input_size != expected.input_size
        || vocab_size != expected.vocab_size
        || hidden_size != expected.hidden_size
        || hidden_layers != expected.hidden_layers
    {
        bail!(
            "model shape mismatch: saved (in={}, vocab={}, hidden={}x{}) vs expected (in={}, vocab={}, hidden={}x{})",
            input_size,
            vocab_size,
            hidden_size,
            hidden_layers,
            expected.input_size,
            expected.vocab_size,
            expected.hidden_size,
            expected.hidden_layers,
        );
    }

    let mut layers = Vec::with_capacity(layer_count);
    for _ in 0..layer_count {
        let rows = read_u32(&mut r)? as usize;
        let cols = read_u32(&mut r)? as usize;
        let act = activation_from_code(read_u8(&mut r)?)?;
        let weights = read_f32_vec(&mut r, rows * cols)?;
        let biases = read_f32_vec(&mut r, rows)?;
        layers.push(Layer::from_parts(rows, cols, act, weights, biases, gpu)?);
    }

    Ok(Some(Network {
        layers,
        input_size,
        vocab_size,
        hidden_size,
        hidden_layers,
    }))
}
