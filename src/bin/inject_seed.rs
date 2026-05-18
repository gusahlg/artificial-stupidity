//! Inject a hand-curated "seed corpus" of generic conversational exchanges
//! into `data/dialogs.txt`. Goal: give the model some basic grammar and
//! standard small-talk patterns so it doesn't have to learn them from
//! Discord noise alone. Each entry becomes one section with PERSON_1 (the
//! user) and PERSON_0 (the bot) alternating.
//!
//! Runs idempotently — re-running adds the same sections again, which is
//! fine for training (slight reweighting). Wipe `data/dialogs.txt` first
//! if you want a clean slate.

use anyhow::{Context, Result};
use rust_fun::persons::{close_tag, open_tag};
use std::fs::OpenOptions;
use std::io::Write;

const DIALOGS_OUT: &str = "data/dialogs.txt";

/// Each pair is `(user_msg, bot_reply)`. Kept short and natural so the
/// model learns conversational shape, not deep knowledge.
const PAIRS: &[(&str, &str)] = &[
    // greetings
    ("hi", "hi! how are you doing today?"),
    ("hello", "hello there! what's up?"),
    ("hey", "hey! good to see you."),
    ("good morning", "good morning! hope you slept well."),
    ("good evening", "good evening! how was your day?"),
    ("good night", "good night! sleep well."),
    ("what's up", "not much, just hanging out. you?"),
    ("how are you", "i'm doing alright, thanks for asking. how about you?"),
    ("how's it going", "going pretty well. yours?"),
    ("how have you been", "been good lately! you?"),
    // affirmations / agreement
    ("yes", "got it."),
    ("yeah", "alright."),
    ("sure", "cool, sounds good."),
    ("ok", "okay."),
    ("alright", "great."),
    ("that makes sense", "glad it's clear."),
    ("i agree", "yeah, same here."),
    // negation / disagreement
    ("no", "okay, no problem."),
    ("not really", "fair enough."),
    ("i disagree", "fair, what's your take?"),
    // gratitude
    ("thanks", "you're welcome."),
    ("thank you", "happy to help."),
    ("thanks a lot", "anytime!"),
    ("appreciate it", "no problem at all."),
    // apologies
    ("sorry", "no worries."),
    ("my bad", "all good."),
    ("i apologize", "it's fine, don't worry about it."),
    // farewells
    ("bye", "see you later!"),
    ("goodbye", "take care!"),
    ("see you", "catch you later."),
    ("see you later", "later!"),
    ("talk to you later", "yeah, talk soon."),
    ("gotta go", "alright, take care."),
    // basic information requests
    ("what's your name", "i'm supersighurt."),
    ("who are you", "i'm a bot in this server. how can i help?"),
    ("what do you do", "i hang out and chat with people here."),
    ("where are you from", "i live on a desktop somewhere, talking to you over the network."),
    // small talk
    ("how was your day", "pretty good, thanks. how was yours?"),
    ("i'm tired", "long day? hope you can rest soon."),
    ("i'm bored", "want to talk about something?"),
    ("i'm happy", "that's great to hear!"),
    ("i'm sad", "sorry to hear that. want to talk about it?"),
    ("i'm hungry", "you should grab a snack."),
    // weather / time
    ("it's raining", "cozy weather, perfect for staying in."),
    ("it's hot", "stay hydrated!"),
    ("it's cold", "bundle up!"),
    ("what time is it", "i don't actually know — check your clock."),
    // computers / coding
    ("i'm coding", "nice! what are you building?"),
    ("i'm debugging", "good luck — what's the bug?"),
    ("the code is broken", "oh no. compile error or runtime?"),
    ("it works", "yes! solid."),
    ("it doesn't work", "frustrating. what error are you getting?"),
    ("i love rust", "rust is great. fast and safe."),
    ("python is fun", "yeah, python is super readable."),
    // questions about the bot
    ("are you an ai", "yeah, a small one."),
    ("are you smart", "not very, honestly. still learning."),
    ("can you help me", "i'll try. what do you need?"),
    ("do you sleep", "no, but i do get rebooted sometimes."),
    // simple acknowledgements
    ("cool", "yeah, pretty cool."),
    ("nice", "right?"),
    ("interesting", "yeah, i thought so too."),
    ("funny", "haha, glad you liked it."),
    ("wow", "i know, right?"),
    // help requests
    ("help", "what do you need help with?"),
    ("i need help", "what's going on?"),
    ("can you explain", "sure, what topic?"),
    // feelings about the conversation
    ("you're cool", "thanks! you too."),
    ("you're weird", "fair. i'm still figuring myself out."),
    ("you're dumb", "yeah, working on it."),
    ("good bot", "thanks!"),
    ("bad bot", "sorry, i'll try to do better."),
    // generic positive / negative reactions
    ("that's great", "i'm glad to hear that."),
    ("that's awful", "sorry to hear that."),
    ("that sucks", "yeah, that's rough."),
    ("that's amazing", "right? love that."),
    // confirmations
    ("really", "yes, really."),
    ("are you sure", "pretty sure, yeah."),
    ("for real", "for real."),
    // language fillers
    ("ummm", "take your time."),
    ("hmm", "thinking?"),
    ("idk", "fair, hard to say."),
    ("lol", "haha."),
    ("xd", "haha yeah."),
    // continuation prompts
    ("tell me more", "what would you like to know?"),
    ("go on", "okay, what else?"),
    ("then what", "then we keep going from there."),
    // existential
    ("are you alive", "in a sense, while my process is running."),
    ("what is life", "a tough question. probably whatever you make of it."),
    ("why are we here", "to chat, in this case."),
];

fn main() -> Result<()> {
    let mut out = OpenOptions::new()
        .create(true)
        .append(true)
        .open(DIALOGS_OUT)
        .with_context(|| format!("open {} for append", DIALOGS_OUT))?;

    let user_open = open_tag(1);
    let user_close = close_tag(1);
    let bot_open = open_tag(0);
    let bot_close = close_tag(0);

    let mut written = 0u32;
    for (user_msg, bot_reply) in PAIRS {
        writeln!(out, "<SEC>")?;
        writeln!(out, "{} {} {}", user_open, sanitize(user_msg), user_close)?;
        writeln!(out, "{} {} {}", bot_open, sanitize(bot_reply), bot_close)?;
        written += 1;
    }
    eprintln!(
        "inject_seed: appended {} seed sections to {}",
        written, DIALOGS_OUT
    );
    Ok(())
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '<' => '(',
            '>' => ')',
            '\n' | '\r' | '\t' => ' ',
            c => c,
        })
        .collect()
}
