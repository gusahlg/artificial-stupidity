use crate::embeddings::Embedding;
use crate::gpu::Gpu;
use crate::neural_network::{Activation, Layer, Network, input_size_for};
use anyhow::{Result, anyhow, bail};
use std::collections::hash_map::DefaultHasher;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

const MAGIC: u32 = 0x4D4F_444C; // "MODL"
/// v2: weights + biases only. Adam state was reset on every load.
/// v3: also persists Adam moments (w_m/w_v/b_m/b_v per layer, embedding m/v)
///     and the global adam_step counter, so resumed training continues with
///     warm Adam state instead of paying the first-step bias-correction tax.
/// v4: also persists a `vocab_hash` over the ordered token list. The
///     output layer is indexed by vocab position, so silently loading
///     a model whose saved vocab order disagrees with the current vocab
///     produces garbage outputs (every row predicts the wrong word).
///     v4 catches this at load time and refuses rather than letting it
///     happen quietly. v2 and v3 files still load (no hash → skip
///     check) for back-compat.
const VERSION_V2: u32 = 2;
const VERSION_V3: u32 = 3;
const VERSION_V4: u32 = 4;
const VERSION_CURRENT: u32 = VERSION_V4;

/// Stable u64 hash over the ordered, prefix-bounded vocab list. Used by
/// the v4 model format to detect silent vocab reorderings after corpus
/// cleanup. `prefix_len` lets a saved hash match against the head of
/// a longer current vocab — that's the supported growth path (vocab
/// extension, see `Embedding::extend_to` / `Layer::extend_rows`).
pub fn compute_vocab_hash(vocab: &[String]) -> u64 {
    compute_vocab_hash_prefix(vocab, vocab.len())
}

fn compute_vocab_hash_prefix(vocab: &[String], prefix_len: usize) -> u64 {
    let mut h = DefaultHasher::new();
    // Mix the prefix length into the hash so a 5-vocab subset of a
    // 10-vocab list isn't a hash collision against a 5-vocab list.
    (prefix_len as u64).hash(&mut h);
    let n = prefix_len.min(vocab.len());
    for w in &vocab[..n] {
        w.hash(&mut h);
    }
    h.finish()
}

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

/// Save the network at `path`. Always writes the current version (v4),
/// including Adam moments, the global Adam step counter, and a hash
/// over the vocab list so a later load can detect silent vocab
/// reorderings. The file roughly triples in size vs v2 (weights +
/// m + v ≈ 3× the v2 payload).
///
/// Atomic via temp-file + rename: the bytes are written to
/// `<path>.tmp` and then renamed onto `<path>` so an external watcher
/// (e.g. `sighurt-llm.path` reloading the inference server on
/// `model.bin` change) never observes a half-written file. If the
/// process dies mid-write, the original `<path>` is left intact and
/// a stale `<path>.tmp` is left behind.
pub fn save<P: AsRef<Path>>(net: &Network, path: P, vocab_hash: u64) -> Result<()> {
    let path = path.as_ref();
    let tmp = with_tmp_suffix(path);
    {
        let f = File::create(&tmp)
            .map_err(|e| anyhow!("create {:?}: {}", tmp, e))?;
        let mut w = BufWriter::new(f);
        write_u32(&mut w, MAGIC)?;
        write_u32(&mut w, VERSION_CURRENT)?;
        write_u32(&mut w, net.embed_dim as u32)?;
        write_u32(&mut w, net.context_window as u32)?;
        write_u32(&mut w, net.vocab_size as u32)?;
        write_u32(&mut w, net.hidden_size as u32)?;
        write_u32(&mut w, net.hidden_layers as u32)?;
        write_u64(&mut w, net.adam_step)?;
        write_u64(&mut w, vocab_hash)?;

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
        // Drop the BufWriter (and inner File) here so the rename below
        // sees a fully-closed file. The IN_CLOSE_WRITE inotify event
        // that systemd's `PathChanged=` uses fires on close, so renames
        // must come after the file is closed for the watcher semantics
        // to line up cleanly.
    }
    std::fs::rename(&tmp, path)
        .map_err(|e| anyhow!("rename {:?} -> {:?}: {}", tmp, path, e))?;
    Ok(())
}

fn with_tmp_suffix(path: &Path) -> std::path::PathBuf {
    let mut s: std::ffi::OsString = path.as_os_str().to_owned();
    s.push(".tmp");
    std::path::PathBuf::from(s)
}

pub struct LoadedShape {
    pub embed_dim: usize,
    pub context_window: usize,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub hidden_layers: usize,
    /// Hash over the ordered vocab list of the *current* run.
    /// `compute_vocab_hash(&vocab)` produces this. The v4 loader
    /// matches against the saved hash; v2 and v3 ignore it.
    pub vocab_hash: u64,
}

/// Load a v2, v3, or v4 model. Vocab growth (saved vocab < expected
/// vocab) is handled by extending the embedding table and the output
/// layer with fresh random rows (and zeroed Adam moments for those
/// new rows); the rest of the structure must match exactly.
///
/// v2 files load with zeroed Adam moments and `adam_step = 0`. No
/// vocab-hash check is performed against v2/v3 since they predate it.
/// v4 files include a vocab hash; the loader checks it against the
/// caller's `expected.vocab_hash`. If the hash disagrees with the
/// hash of the first `saved_vocab_size` tokens of the current vocab,
/// the load is refused — better than silently loading rows-against-
/// reordered-words and producing garbage.
pub fn load<P: AsRef<Path>>(
    path: P,
    gpu: &Gpu,
    expected: LoadedShape,
) -> Result<Option<Network>> {
    load_with_vocab(path, gpu, expected, None)
}

/// Variant of `load` that takes the current vocab for v4 hash
/// verification. When `vocab` is `Some`, the v4 loader checks that
/// `compute_vocab_hash_prefix(vocab, saved_vocab_size)` matches the
/// hash stamped in the file. `None` skips the prefix verification
/// and relies only on the `expected.vocab_hash` equality check, which
/// is the right choice when the caller can't easily pass the live
/// vocab (e.g. tests).
pub fn load_with_vocab<P: AsRef<Path>>(
    path: P,
    gpu: &Gpu,
    expected: LoadedShape,
    vocab: Option<&[String]>,
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
        VERSION_V4 => load_v4(&mut r, gpu, expected, vocab).map(Some),
        v => bail!(
            "model file version {} not supported (expected {}, {}, or {})",
            v,
            VERSION_V2,
            VERSION_V3,
            VERSION_V4,
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
        dropout_p: 0.0,
        label_smoothing: 0.0,
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
        dropout_p: 0.0,
        label_smoothing: 0.0,
        backprop_scratch: crate::teacher::BackpropScratch::default(),
        profile: crate::neural_network::StepProfile::default(),
    })
}

fn load_v4<R: Read>(
    r: &mut R,
    gpu: &Gpu,
    expected: LoadedShape,
    current_vocab: Option<&[String]>,
) -> Result<Network> {
    let h = read_header(r, &expected)?;
    let adam_step = read_u64(r)?;
    let saved_vocab_hash = read_u64(r)?;

    // Two layers of vocab-hash defense:
    //
    // 1. If the caller passed the live vocab, hash its first
    //    `saved_vocab_size` tokens and compare. This is the precise
    //    check: it allows safe growth (the saved tokens still occupy
    //    the same positions; current vocab has additional tail
    //    tokens which get random-init'd by `maybe_extend_vocab`).
    //
    // 2. Otherwise compare against `expected.vocab_hash`, the hash of
    //    the FULL current vocab. This is strict: even safe growth
    //    fails the comparison. Useful for callers that can't easily
    //    pass the live vocab (tests, one-shot tools).
    match current_vocab {
        Some(vocab) => {
            let prefix_hash = compute_vocab_hash_prefix(vocab, h.saved_vocab_size);
            if prefix_hash != saved_vocab_hash {
                bail!(
                    "vocab mismatch: saved model's first {} vocab tokens (hash {:#x}) \
                     do not match the current vocab's prefix (hash {:#x}). The output \
                     layer is row-indexed by vocab position, so loading would silently \
                     produce garbage. Move {} aside or rebuild the corpus before \
                     resuming.",
                    h.saved_vocab_size,
                    saved_vocab_hash,
                    prefix_hash,
                    "model.bin",
                );
            }
        }
        None => {
            if saved_vocab_hash != expected.vocab_hash {
                bail!(
                    "vocab mismatch: saved hash {:#x} != current hash {:#x}",
                    saved_vocab_hash,
                    expected.vocab_hash,
                );
            }
        }
    }

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
        dropout_p: 0.0,
        label_smoothing: 0.0,
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

    fn dummy_vocab(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("tok_{}", i)).collect()
    }

    /// Roundtrip: save a network, load it back, assert every persisted
    /// field matches exactly. Covers v4 (v3 contents + vocab hash check).
    #[test]
    fn v4_roundtrip_preserves_weights_biases_and_adam_state() {
        let gpu = Gpu::new_cpu();
        // Use full project dims so the tested code path matches production.
        let vocab_n = 17usize;
        let vocab = dummy_vocab(vocab_n);
        let vocab_hash = compute_vocab_hash(&vocab);
        let mut net = network_init(
            &gpu,
            EMBED_DIM,
            CONTEXT_WINDOW,
            HIDDEN_SIZE,
            NUMBER_OF_HIDDEN_LAYERS,
            vocab_n,
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

        let path = tmp_path("persist_v4_roundtrip");
        save(&net, &path, vocab_hash).expect("save v4");

        let shape = LoadedShape {
            embed_dim: EMBED_DIM,
            context_window: CONTEXT_WINDOW,
            vocab_size: vocab_n,
            hidden_size: HIDDEN_SIZE,
            hidden_layers: NUMBER_OF_HIDDEN_LAYERS,
            vocab_hash,
        };
        let loaded = load_with_vocab(&path, &gpu, shape, Some(&vocab))
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

    /// Save with one vocab, load with a reordered vocab — must bail
    /// rather than silently load with wrong row-to-word mapping.
    #[test]
    fn v4_load_bails_on_reordered_vocab() {
        let gpu = Gpu::new_cpu();
        let vocab_n = 17usize;
        let mut vocab = dummy_vocab(vocab_n);
        let saved_hash = compute_vocab_hash(&vocab);
        let net = network_init(
            &gpu,
            EMBED_DIM,
            CONTEXT_WINDOW,
            HIDDEN_SIZE,
            NUMBER_OF_HIDDEN_LAYERS,
            vocab_n,
        )
        .expect("network_init");

        let path = tmp_path("persist_v4_reorder_bail");
        save(&net, &path, saved_hash).expect("save v4");

        // Reorder: swap two non-reserved positions deep in the vocab.
        vocab.swap(10, 12);
        let bad_hash = compute_vocab_hash(&vocab);
        assert_ne!(saved_hash, bad_hash);

        let shape = LoadedShape {
            embed_dim: EMBED_DIM,
            context_window: CONTEXT_WINDOW,
            vocab_size: vocab_n,
            hidden_size: HIDDEN_SIZE,
            hidden_layers: NUMBER_OF_HIDDEN_LAYERS,
            vocab_hash: bad_hash,
        };
        let result = load_with_vocab(&path, &gpu, shape, Some(&vocab));
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("expected bail on vocab mismatch but load succeeded"),
        };
        let msg = format!("{}", err);
        assert!(
            msg.contains("vocab mismatch"),
            "error should mention vocab mismatch, got: {}",
            msg
        );

        let _ = std::fs::remove_file(&path);
    }

    /// Vocab growth (append-only) is the supported path: the saved
    /// hash matches the prefix of the larger current vocab, so load
    /// succeeds and `maybe_extend_vocab` adds random rows.
    #[test]
    fn v4_load_accepts_vocab_growth() {
        let gpu = Gpu::new_cpu();
        let saved_n = 17usize;
        let saved_vocab = dummy_vocab(saved_n);
        let saved_hash = compute_vocab_hash(&saved_vocab);
        let net = network_init(
            &gpu,
            EMBED_DIM,
            CONTEXT_WINDOW,
            HIDDEN_SIZE,
            NUMBER_OF_HIDDEN_LAYERS,
            saved_n,
        )
        .expect("network_init");

        let path = tmp_path("persist_v4_growth");
        save(&net, &path, saved_hash).expect("save v4");

        // Extend: same prefix, two new tokens at the tail.
        let mut grown = saved_vocab.clone();
        grown.push("tok_17".to_string());
        grown.push("tok_18".to_string());
        let grown_hash = compute_vocab_hash(&grown);
        assert_ne!(saved_hash, grown_hash); // full-vocab hash differs

        let shape = LoadedShape {
            embed_dim: EMBED_DIM,
            context_window: CONTEXT_WINDOW,
            vocab_size: grown.len(),
            hidden_size: HIDDEN_SIZE,
            hidden_layers: NUMBER_OF_HIDDEN_LAYERS,
            vocab_hash: grown_hash,
        };
        let loaded = load_with_vocab(&path, &gpu, shape, Some(&grown))
            .expect("load with grown vocab")
            .expect("file present");
        assert_eq!(loaded.vocab_size, grown.len());

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
            // v2 ignores vocab_hash entirely (no hash on disk to check).
            vocab_hash: 0,
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
