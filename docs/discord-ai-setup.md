# Discord AI server setup

This guide runs the AI HTTP server on another Linux computer and connects an
existing Discord bot to it. Use a private network such as Tailscale when the
bot and AI server are on different machines.

## 1. Build the server

Clone `hello_rust` and its path dependency side by side:

```sh
mkdir -p ~/repos
cd ~/repos
git clone https://github.com/gusahlg/tensor-ash.git
git clone https://github.com/gusahlg/artificial-stupidity.git hello_rust
cd hello_rust
nix develop -c cargo build --release --bin serve
```

Without Nix, install current stable Rust plus `glslc`/shaderc, then run
`cargo build --release --bin serve`.

Copy the matching `model.bin` and `data/dialogs.txt` from the machine that
trained the model. Keep those two files together: the vocabulary derived from
the corpus determines how model rows are interpreted. Do not copy
`data/dialogs.bin`; it is a generated cache.

## 2. Configure the private endpoint

If the bot runs on this same computer, use `127.0.0.1:8088`. Otherwise,
connect both computers to Tailscale, run `tailscale ip -4` on the AI server,
and bind to that address. Do not bind this service to `0.0.0.0` on a
publicly reachable host.

Create a fresh key and a private environment file:

```sh
mkdir -p ~/.config/systemd/user
install -m 600 /dev/null ~/.config/sighurt-llm.env
openssl rand -hex 32
```

Put the generated key and the correct absolute model path in
`~/.config/sighurt-llm.env`:

```ini
SIGHURT_BIND=127.0.0.1:8088
SIGHURT_API_KEY=PASTE_THE_NEW_KEY_HERE
SIGHURT_MODEL=/home/YOUR_USER/repos/hello_rust/model.bin
```

For a remote bot, replace `127.0.0.1` with the AI server's Tailscale IPv4.

## 3. Install the user service

Create `~/.config/systemd/user/sighurt-llm.service`:

```ini
[Unit]
Description=SuperSighurt LLM HTTP server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=%h/repos/hello_rust
ExecStart=%h/repos/hello_rust/target/release/serve
EnvironmentFile=%h/.config/sighurt-llm.env
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

Enable it and optionally allow it to start before you log in:

```sh
systemctl --user daemon-reload
systemctl --user enable --now sighurt-llm.service
sudo loginctl enable-linger "$USER"
curl -fsS http://127.0.0.1:8088/healthz
```

Use the Tailscale address instead of localhost for the health check when the
service binds there. View logs with
`journalctl --user -u sighurt-llm.service -f`.

The probe scripts default to localhost. For a Tailscale-bound server, run
them with `HOST=http://AI_SERVER_ADDRESS:8088`.

## 4. Point the Discord bot at it

Use a version of [sighurt-bot](https://github.com/gusahlg/sighurt-bot) that
contains its `[chat]` HTTP client. Complete that repository's normal bot
setup first, including Discord's Message Content intent and permission to
view channels, read messages, and send messages.

On the bot computer, edit `~/discord-bot/config.toml`:

```toml
[chat]
enabled = true
endpoint_url = "http://AI_SERVER_ADDRESS:8088"
request_timeout_secs = 30
# Replace this example number with your Discord user ID.
admin_user_ids = [123456789012345678]
```

Add the same new key to the bot's private `~/discord-bot/.env` file:

```ini
LLM_API_KEY=PASTE_THE_NEW_KEY_HERE
```

Then verify the route and restart the bot:

```sh
chmod 600 ~/discord-bot/.env
curl -fsS http://AI_SERVER_ADDRESS:8088/healthz
systemctl --user restart discord-bot.service
journalctl --user -u discord-bot.service -n 20 --no-pager
```

The bot log should report `Chat runtime ready (initial = ON)`. Test with a DM
or by mentioning the bot in a server channel; ordinary unmentioned channel
messages are not forwarded. The listed admins can use `!ai on`, `!ai off`,
and `!ai status`.

After replacing `model.bin`, restart the server with
`systemctl --user restart sighurt-llm.service`.
