//! In-memory RAG store. At server boot, every turn in the corpus is
//! embedded (centroid over token-id embeddings) and indexed. At inference
//! time, the user's input is embedded and the top-K most similar turns are
//! prepended to the generation prompt as additional context.
//!
//! Brute-force cosine over the row set. Up to ~100k turns this is fine on
//! commodity hardware (a few ms per query). Above that, add an ANN index.

use crate::dialogs::Data;
use crate::embeddings::{Embedding, cosine};
use crate::neural_network::VocabIndex;
use crate::persons::{close_tag, open_tag};
use crate::tokenizer::tokenize;
use std::cmp::Ordering;

#[derive(Clone, Debug)]
pub struct RagRow {
    pub person_id: u32,
    pub tokens: Vec<String>,
    pub embedding: Vec<f32>,
}

pub struct RagStore {
    rows: Vec<RagRow>,
}

impl RagStore {
    pub fn new() -> Self {
        Self { rows: Vec::new() }
    }

    /// Build a fresh store by walking every turn in the corpus and embedding
    /// its tokens via the network's current embedding table.
    pub fn populate_from_corpus(data: &Data, embedding: &Embedding, words: &[String]) -> Self {
        let vocab = VocabIndex::new(words);
        let mut rows = Vec::new();
        for section in &data.sections {
            for turn in section {
                if turn.tokens.is_empty() {
                    continue;
                }
                let ids = vocab.ids_or_unk(&turn.tokens);
                let emb = embedding.centroid(&ids);
                rows.push(RagRow {
                    person_id: turn.person_id,
                    tokens: turn.tokens.clone(),
                    embedding: emb,
                });
            }
        }
        Self { rows }
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Index a single live turn into the store.
    pub fn insert_live(
        &mut self,
        person_id: u32,
        text: &str,
        embedding: &Embedding,
        words: &[String],
    ) {
        let tokens = tokenize(text);
        if tokens.is_empty() {
            return;
        }
        let vocab = VocabIndex::new(words);
        let ids = vocab.ids_or_unk(&tokens);
        let emb = embedding.centroid(&ids);
        self.rows.push(RagRow {
            person_id,
            tokens,
            embedding: emb,
        });
    }

    /// Return up to `k` rows with the highest cosine similarity to `query`.
    /// Results are ordered most-similar first.
    pub fn top_k(&self, query: &[f32], k: usize) -> Vec<&RagRow> {
        if self.rows.is_empty() || k == 0 {
            return Vec::new();
        }
        let mut scored: Vec<(f32, usize)> = self
            .rows
            .iter()
            .enumerate()
            .map(|(i, r)| (cosine(query, &r.embedding), i))
            .filter(|(s, _)| s.is_finite())
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        scored
            .into_iter()
            .take(k)
            .map(|(_, i)| &self.rows[i])
            .collect()
    }

    /// Format a hit as a wrapped turn for splicing into the generation prompt.
    pub fn render(row: &RagRow) -> String {
        let mut s = String::new();
        s.push_str(&open_tag(row.person_id));
        s.push(' ');
        s.push_str(&row.tokens.join(" "));
        s.push(' ');
        s.push_str(&close_tag(row.person_id));
        s
    }
}

impl Default for RagStore {
    fn default() -> Self {
        Self::new()
    }
}
