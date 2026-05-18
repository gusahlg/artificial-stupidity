//! Persistent Discord-user-id → PERSON_N mapping.
//!
//! Append-only: every row defines `person_id = line_number - 1`. Row 0 is
//! reserved for the bot itself — the speaker the model learns to be at
//! inference. Once a person_id is assigned, it is stable forever, so the
//! model's learned PERSON_3 embedding remains attached to the same Discord
//! user across all subsequent training runs.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::Path;

pub const PERSONS_FILE: &str = "data/persons.tsv";
pub const BOT_PERSON_ID: u32 = 0;

#[derive(Clone, Debug)]
pub struct PersonRow {
    pub person_id: u32,
    pub discord_user_id: u64,
    pub display_name: String,
}

pub struct PersonTable {
    rows: Vec<PersonRow>,
    by_user_id: HashMap<u64, u32>,
}

impl PersonTable {
    pub fn new() -> Self {
        Self {
            rows: Vec::new(),
            by_user_id: HashMap::new(),
        }
    }

    pub fn load() -> Result<Self> {
        Self::load_from(PERSONS_FILE)
    }

    pub fn load_from<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(Self::new());
        }
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("read {:?}", path))?;
        let mut t = Self::new();
        for (i, line) in content.lines().enumerate() {
            if line.is_empty() {
                continue;
            }
            let mut parts = line.splitn(3, '\t');
            let id: u32 = parts
                .next()
                .context("missing person_id column")?
                .parse()
                .with_context(|| format!("line {}: bad person_id", i + 1))?;
            let user_id: u64 = parts
                .next()
                .context("missing discord_user_id column")?
                .parse()
                .with_context(|| format!("line {}: bad discord_user_id", i + 1))?;
            let name = parts.next().unwrap_or("").to_string();
            if id != t.rows.len() as u32 {
                anyhow::bail!(
                    "persons.tsv out of order at line {}: expected id {} got {}",
                    i + 1,
                    t.rows.len(),
                    id
                );
            }
            t.by_user_id.insert(user_id, id);
            t.rows.push(PersonRow {
                person_id: id,
                discord_user_id: user_id,
                display_name: name,
            });
        }
        Ok(t)
    }

    pub fn save(&self) -> Result<()> {
        self.save_to(PERSONS_FILE)
    }

    pub fn save_to<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let mut body = String::new();
        for r in &self.rows {
            body.push_str(&format!(
                "{}\t{}\t{}\n",
                r.person_id, r.discord_user_id, r.display_name
            ));
        }
        std::fs::write(path, body).with_context(|| format!("write {:?}", path))?;
        Ok(())
    }

    /// Return the existing PERSON id for the given Discord user, or assign
    /// the next free id and append a new row. The row is held in memory
    /// until `save` is called.
    pub fn resolve_or_assign(&mut self, discord_user_id: u64, display_name: &str) -> u32 {
        if let Some(&id) = self.by_user_id.get(&discord_user_id) {
            return id;
        }
        let id = self.rows.len() as u32;
        self.by_user_id.insert(discord_user_id, id);
        self.rows.push(PersonRow {
            person_id: id,
            discord_user_id,
            display_name: display_name.to_string(),
        });
        id
    }

    pub fn lookup(&self, discord_user_id: u64) -> Option<u32> {
        self.by_user_id.get(&discord_user_id).copied()
    }

    pub fn name_of(&self, person_id: u32) -> Option<&str> {
        self.rows.get(person_id as usize).map(|r| r.display_name.as_str())
    }

    pub fn discord_id_of(&self, person_id: u32) -> Option<u64> {
        self.rows.get(person_id as usize).map(|r| r.discord_user_id)
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn rows(&self) -> &[PersonRow] {
        &self.rows
    }

    /// Ensure row 0 corresponds to the bot. On an empty table this seeds
    /// the row; on a populated one, if row 0 belongs to a different Discord
    /// id, this is a fatal misconfiguration (we'd retrain into the wrong
    /// identity) and the call fails.
    pub fn ensure_bot(&mut self, bot_discord_id: u64, bot_name: &str) -> Result<()> {
        match self.rows.first() {
            None => {
                self.by_user_id.insert(bot_discord_id, 0);
                self.rows.push(PersonRow {
                    person_id: 0,
                    discord_user_id: bot_discord_id,
                    display_name: bot_name.to_string(),
                });
                Ok(())
            }
            Some(r) if r.discord_user_id == bot_discord_id => Ok(()),
            Some(r) => anyhow::bail!(
                "persons.tsv row 0 is discord id {}; expected bot {}. Refusing to overwrite.",
                r.discord_user_id,
                bot_discord_id
            ),
        }
    }

    /// All PERSON open/close tags currently known. Corpus parser and
    /// tokenizer consult this to recognize them as whole-token specials.
    pub fn all_person_tags(&self) -> Vec<String> {
        let mut out = Vec::with_capacity(self.rows.len() * 2);
        for r in &self.rows {
            out.push(open_tag(r.person_id));
            out.push(close_tag(r.person_id));
        }
        out
    }
}

impl Default for PersonTable {
    fn default() -> Self {
        Self::new()
    }
}

pub fn open_tag(id: u32) -> String {
    format!("<PERSON_{}>", id)
}

pub fn close_tag(id: u32) -> String {
    format!("</PERSON_{}>", id)
}

pub fn parse_open_tag(tok: &str) -> Option<u32> {
    let inner = tok.strip_prefix("<PERSON_")?.strip_suffix('>')?;
    inner.parse().ok()
}

pub fn parse_close_tag(tok: &str) -> Option<u32> {
    let inner = tok.strip_prefix("</PERSON_")?.strip_suffix('>')?;
    inner.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_close_tag_roundtrip() {
        assert_eq!(open_tag(7), "<PERSON_7>");
        assert_eq!(close_tag(7), "</PERSON_7>");
        assert_eq!(parse_open_tag("<PERSON_7>"), Some(7));
        assert_eq!(parse_close_tag("</PERSON_7>"), Some(7));
        assert_eq!(parse_open_tag("<PERSON_>"), None);
        assert_eq!(parse_open_tag("PERSON_3"), None);
        assert_eq!(parse_close_tag("<PERSON_3>"), None);
    }

    #[test]
    fn resolve_assign_and_lookup() {
        let mut t = PersonTable::new();
        let a = t.resolve_or_assign(1001, "alice");
        let b = t.resolve_or_assign(1002, "bob");
        let a2 = t.resolve_or_assign(1001, "alice-renamed");
        assert_eq!(a, 0);
        assert_eq!(b, 1);
        assert_eq!(a2, 0);
        assert_eq!(t.name_of(0), Some("alice"));
        assert_eq!(t.lookup(1002), Some(1));
        assert_eq!(t.len(), 2);
    }

    #[test]
    fn ensure_bot_seeds_when_empty() {
        let mut t = PersonTable::new();
        t.ensure_bot(999, "supersighurt").unwrap();
        assert_eq!(t.discord_id_of(0), Some(999));
    }

    #[test]
    fn ensure_bot_rejects_mismatch() {
        let mut t = PersonTable::new();
        t.resolve_or_assign(42, "not-the-bot");
        let r = t.ensure_bot(999, "supersighurt");
        assert!(r.is_err());
    }
}
