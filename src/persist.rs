use crate::embeddings::Embedding;
use crate::gpu::Gpu;
use crate::neural_network::{Activation, Layer, Network, input_size_for};
use anyhow::{Result, anyhow, bail};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

const MAGIC: u32 = 0x4D4F_444C; // "MODL"
/// v2: weights + biases only. Adam state was reset on every load.
/// v3: also persists Adam moments (w_m/w_v/b_m/b_v per layer, embedding m/v)
///     and the global adam_step counter, so resumed training continues with
///     warm Adam state instead of paying the first-step bias-correction tax.
const VERSION_V2: u32 = 2;
const VERSION_V3: u32 = 3;
const VERSION_CURRENT: u32 = VERSION_V3;

fn write_u32<W: Write>(w: &mut W, v: u32) -> Result<()> {
    w.write_all(&v.to_le_bytes())?;
    Ok(())
}
fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn write_u64<W: Write>(w: &mut W, v: u64) -> Result<()> {
    w.write_all(&v.to_le_bytes())?;
    Ok(())
}
fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
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

/// Save the network at `path`. Always writes the current version (v3),
/// including Adam moments and the global Adam step counter. The file
/// roughly triples in size vs v2 (weights + m + v ≈ 3× the v2 payload).
pub fn save<P: AsRef<Path>>(net: &Network, path: P) -> Result<()> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);
    write_u32(&mut w, MAGIC)?;
    write_u32(&mut w, VERSION_CURRENT)?;
    write_u32(&mut w, net.embed_dim as u32)?;
    write_u32(&mut w, net.context_window as u32)?;
    write_u32(&mut w, net.vocab_size as u32)?;
    write_u32(&mut w, net.hidden_size as u32)?;
    write_u32(&mut w, net.hidden_layers as u32)?;
    write_u64(&mut w, net.adam_step)?;

    // Embedding: weights + Adam moments.
    write_f32_slice(&mut w, &net.embedding.weights)?;
    write_f32_slice(&mut w, &net.embedding.m)?;
    write_f32_slice(&mut w, &net.embedding.v)?;

    // Dense layers: weights, biases + Adam moments per layer.
    write_u32(&mut w, net.layers.len() as u32)?;
    for layer in &net.layers {
        write_u32(&mut w, layer.rows as u32)?;
        write_u32(&mut w, layer.cols as u32)?;
        write_u8(&mut w, activation_code(layer.activation))?;
        write_f32_slice(&mut w, &layer.weights)?;
        write_f32_slice(&mut w, &layer.biases)?;
        write_f32_slice(&mut w, &layer.w_m)?;
        write_f32_slice(&mut w, &layer.w_v)?;
        write_f32_slice(&mut w, &layer.b_m)?;
        write_f32_slice(&mut w, &layer.b_v)?;
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

/// Load a v2 or v3 model. Vocab growth (saved vocab < expected vocab) is
/// handled by extending the embedding table and the output layer with
/// fresh random rows (and zeroed Adam moments for those new rows); the
/// rest of the structure must match exactly.
///
/// v2 files are loaded with zeroed Adam moments and `adam_step = 0`. This
/// preserves the historical behavior so existing checkpoints stay
/// readable, at the cost of one warmup epoch on resume. New saves are
/// always v3.
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
    match version {
        VERSION_V2 => load_v2(&mut r, gpu, expected).map(Some),
        VERSION_V3 => load_v3(&mut r, gpu, expected).map(Some),
        v => bail!(
            "model file version {} not supported (expected {} or {})",
            v,
            VERSION_V2,
            VERSION_V3,
        ),
    }
}

/// Common geometry header. Both v2 and v3 share the same five dimension
/// fields after MAGIC + VERSION.
struct Header {
    embed_dim: usize,
    context_window: usize,
    saved_vocab_size: usize,
    hidden_size: usize,
    hidden_layers: usize,
}

fn read_header<R: Read>(r: &mut R, expected: &LoadedShape) -> Result<Header> {
    let embed_dim = read_u32(r)? as usize;
    let context_window = read_u32(r)? as usize;
    let saved_vocab_size = read_u32(r)? as usize;
    let hidden_size = read_u32(r)? as usize;
    let hidden_layers = read_u32(r)? as usize;

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
    Ok(Header {
        embed_dim,
        context_window,
        saved_vocab_size,
        hidden_size,
        hidden_layers,
    })
}

fn validate_layer_dims(
    i: usize,
    layer_count: usize,
    rows: usize,
    cols: usize,
    embed_dim: usize,
    context_window: usize,
    saved_vocab_size: usize,
) -> Result<()> {
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
    Ok(())
}

fn maybe_extend_vocab(
    embedding: &mut Embedding,
    layers: &mut [Layer],
    expected_vocab_size: usize,
    saved_vocab_size: usize,
    gpu: &Gpu,
) -> Result<()> {
    if expected_vocab_size > saved_vocab_size {
        let mut rng = rand::thread_rng();
        embedding.extend_to(expected_vocab_size, &mut rng);
        let last = layers.len() - 1;
        layers[last].extend_rows(expected_vocab_size, gpu, &mut rng)?;
        eprintln!(
            "Extended saved model from vocab {} to {} (new rows initialized randomly, Adam moments zeroed for new rows).",
            saved_vocab_size, expected_vocab_size
        );
    }
    Ok(())
}

fn load_v2<R: Read>(r: &mut R, gpu: &Gpu, expected: LoadedShape) -> Result<Network> {
    let h = read_header(r, &expected)?;

    let embed_weights = read_f32_vec(r, h.saved_vocab_size * h.embed_dim)?;
    let mut embedding = Embedding::from_parts(h.saved_vocab_size, h.embed_dim, embed_weights);

    let layer_count = read_u32(r)? as usize;
    let mut layers = Vec::with_capacity(layer_count);
    for i in 0..layer_count {
        let rows = read_u32(r)? as usize;
        let cols = read_u32(r)? as usize;
        let act = activation_from_code(read_u8(r)?)?;
        let weights = read_f32_vec(r, rows * cols)?;
        let biases = read_f32_vec(r, rows)?;
        validate_layer_dims(
            i,
            layer_count,
            rows,
            cols,
            h.embed_dim,
            h.context_window,
            h.saved_vocab_size,
        )?;
        layers.push(Layer::from_parts(rows, cols, act, weights, biases, gpu)?);
    }

    maybe_extend_vocab(
        &mut embedding,
        &mut layers,
        expected.vocab_size,
        h.saved_vocab_size,
        gpu,
    )?;

    Ok(Network {
        embedding,
        layers,
        vocab_size: expected.vocab_size,
        hidden_size: h.hidden_size,
        hidden_layers: h.hidden_layers,
        embed_dim: h.embed_dim,
        context_window: h.context_window,
        // v2 had no persisted Adam state; first training step will pay
        // the warmup bias-correction tax. v3 avoids this.
        adam_step: 0,
        backprop_scratch: crate::teacher::BackpropScratch::default(),
        profile: crate::neural_network::StepProfile::default(),
    })
}

fn load_v3<R: Read>(r: &mut R, gpu: &Gpu, expected: LoadedShape) -> Result<Network> {
    let h = read_header(r, &expected)?;
    let adam_step = read_u64(r)?;

    let embed_len = h.saved_vocab_size * h.embed_dim;
    let embed_weights = read_f32_vec(r, embed_len)?;
    let embed_m = read_f32_vec(r, embed_len)?;
    let embed_v = read_f32_vec(r, embed_len)?;
    let mut embedding = Embedding::from_parts_with_adam(
        h.saved_vocab_size,
        h.embed_dim,
        embed_weights,
        embed_m,
        embed_v,
    );

    let layer_count = read_u32(r)? as usize;
    let mut layers = Vec::with_capacity(layer_count);
    for i in 0..layer_count {
        let rows = read_u32(r)? as usize;
        let cols = read_u32(r)? as usize;
        let act = activation_from_code(read_u8(r)?)?;
        let weights = read_f32_vec(r, rows * cols)?;
        let biases = read_f32_vec(r, rows)?;
        let w_m = read_f32_vec(r, rows * cols)?;
        let w_v = read_f32_vec(r, rows * cols)?;
        let b_m = read_f32_vec(r, rows)?;
        let b_v = read_f32_vec(r, rows)?;
        validate_layer_dims(
            i,
            layer_count,
            rows,
            cols,
            h.embed_dim,
            h.context_window,
            h.saved_vocab_size,
        )?;
        layers.push(Layer::from_parts_with_adam(
            rows, cols, act, weights, biases, w_m, w_v, b_m, b_v, gpu,
        )?);
    }

    maybe_extend_vocab(
        &mut embedding,
        &mut layers,
        expected.vocab_size,
        h.saved_vocab_size,
        gpu,
    )?;

    Ok(Network {
        embedding,
        layers,
        vocab_size: expected.vocab_size,
        hidden_size: h.hidden_size,
        hidden_layers: h.hidden_layers,
        embed_dim: h.embed_dim,
        context_window: h.context_window,
        adam_step,
        backprop_scratch: crate::teacher::BackpropScratch::default(),
        profile: crate::neural_network::StepProfile::default(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_network::{
        CONTEXT_WINDOW, EMBED_DIM, HIDDEN_SIZE, NUMBER_OF_HIDDEN_LAYERS, network_init,
    };

    fn tmp_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        let unique = format!(
            "{}_{}_{}",
            name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        p.push(unique);
        p
    }

    /// Roundtrip: save a network, load it back, assert every persisted
    /// field matches exactly. Covers v3 (weights + biases + Adam state).
    #[test]
    fn v3_roundtrip_preserves_weights_biases_and_adam_state() {
        let gpu = Gpu::new_cpu();
        // Use full project dims so the tested code path matches production.
        let vocab = 17usize;
        let mut net = network_init(
            &gpu,
            EMBED_DIM,
            CONTEXT_WINDOW,
            HIDDEN_SIZE,
            NUMBER_OF_HIDDEN_LAYERS,
            vocab,
        )
        .expect("network_init");

        net.adam_step = 12345;
        // Sprinkle non-default values into Adam state and embeddings so a
        // load that returns zeroed state would obviously fail.
        net.embedding.weights[0] = 0.125;
        net.embedding.m[1] = 0.5;
        net.embedding.v[2] = 0.25;
        for (li, layer) in net.layers.iter_mut().enumerate() {
            let scale = 0.01 * (li as f32 + 1.0);
            let w_v_idx = 1 % layer.w_v.len();
            let b_v_last = layer.b_v.len() - 1;
            layer.w_m[0] = scale;
            layer.w_v[w_v_idx] = scale * 2.0;
            layer.b_m[0] = scale * 3.0;
            layer.b_v[b_v_last] = scale * 4.0;
            layer.biases[0] = 0.7 + scale;
        }

        let path = tmp_path("persist_v3_roundtrip");
        save(&net, &path).expect("save v3");

        let shape = LoadedShape {
            embed_dim: EMBED_DIM,
            context_window: CONTEXT_WINDOW,
            vocab_size: vocab,
            hidden_size: HIDDEN_SIZE,
            hidden_layers: NUMBER_OF_HIDDEN_LAYERS,
        };
        let loaded = load(&path, &gpu, shape)
            .expect("load")
            .expect("file present");

        assert_eq!(loaded.adam_step, 12345);
        assert_eq!(loaded.embedding.weights, net.embedding.weights);
        assert_eq!(loaded.embedding.m, net.embedding.m);
        assert_eq!(loaded.embedding.v, net.embedding.v);
        assert_eq!(loaded.layers.len(), net.layers.len());
        for (orig, got) in net.layers.iter().zip(loaded.layers.iter()) {
            assert_eq!(orig.rows, got.rows);
            assert_eq!(orig.cols, got.cols);
            assert_eq!(orig.weights, got.weights);
            assert_eq!(orig.biases, got.biases);
            assert_eq!(orig.w_m, got.w_m);
            assert_eq!(orig.w_v, got.w_v);
            assert_eq!(orig.b_m, got.b_m);
            assert_eq!(orig.b_v, got.b_v);
        }

        let _ = std::fs::remove_file(&path);
    }

    /// Hand-craft a minimal v2 file and ensure the loader still accepts it,
    /// dropping Adam state (zeroed) but preserving weights/biases. Keeps
    /// existing on-disk checkpoints readable.
    #[test]
    fn v2_files_still_load_with_zeroed_adam() {
        let gpu = Gpu::new_cpu();
        // Build a tiny v2 file in memory using the same byte format as the
        // pre-v3 save() did.
        let embed_dim = 3usize;
        let context = 2usize;
        let vocab = 5usize;
        let hidden = 4usize;
        let layers_count = 2usize; // one hidden + output

        let input_size = embed_dim * context + 1; // POSITION_FEATURES
        let layer_specs: Vec<(usize, usize, Activation)> = vec![
            (hidden, input_size, Activation::Tanh),
            (vocab, hidden, Activation::Linear),
        ];

        let mut bytes: Vec<u8> = Vec::new();
        bytes.extend_from_slice(&MAGIC.to_le_bytes());
        bytes.extend_from_slice(&VERSION_V2.to_le_bytes());
        bytes.extend_from_slice(&(embed_dim as u32).to_le_bytes());
        bytes.extend_from_slice(&(context as u32).to_le_bytes());
        bytes.extend_from_slice(&(vocab as u32).to_le_bytes());
        bytes.extend_from_slice(&(hidden as u32).to_le_bytes());
        bytes.extend_from_slice(&((layers_count - 1) as u32).to_le_bytes()); // hidden_layers count

        // Embedding weights: vocab * embed_dim floats, all 0.5 so we can
        // assert later they were read.
        let embed_floats = vec![0.5f32; vocab * embed_dim];
        for f in &embed_floats {
            bytes.extend_from_slice(&f.to_le_bytes());
        }

        bytes.extend_from_slice(&(layers_count as u32).to_le_bytes());
        for (rows, cols, act) in &layer_specs {
            bytes.extend_from_slice(&(*rows as u32).to_le_bytes());
            bytes.extend_from_slice(&(*cols as u32).to_le_bytes());
            bytes.push(activation_code(*act));
            for _ in 0..(rows * cols) {
                bytes.extend_from_slice(&0.25f32.to_le_bytes());
            }
            for _ in 0..*rows {
                bytes.extend_from_slice(&0.75f32.to_le_bytes());
            }
        }

        let path = tmp_path("persist_v2_compat");
        std::fs::write(&path, &bytes).expect("write tmp v2 file");

        let shape = LoadedShape {
            embed_dim,
            context_window: context,
            vocab_size: vocab,
            hidden_size: hidden,
            hidden_layers: layers_count - 1,
        };
        let net = load(&path, &gpu, shape)
            .expect("v2 load")
            .expect("file present");

        assert_eq!(net.adam_step, 0, "v2 must come back with fresh Adam step");
        assert!(net.embedding.weights.iter().all(|&w| w == 0.5));
        assert!(net.embedding.m.iter().all(|&x| x == 0.0));
        assert!(net.embedding.v.iter().all(|&x| x == 0.0));
        for layer in &net.layers {
            assert!(layer.w_m.iter().all(|&x| x == 0.0));
            assert!(layer.w_v.iter().all(|&x| x == 0.0));
            assert!(layer.b_m.iter().all(|&x| x == 0.0));
            assert!(layer.b_v.iter().all(|&x| x == 0.0));
        }

        let _ = std::fs::remove_file(&path);
    }
}
