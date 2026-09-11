# Connect the hosted demo to local Ollama (V29)

Streamlit Cloud -> HTTPS Quick Tunnel -> authenticated gateway -> local Ollama.
This is a scheduled demo configuration, not an always-on production service.
The PC, Docker Desktop, Ollama and cloudflared must all remain running.
Only LLM requests go through this gateway. It does not share the local database
or MCP server; the hosted web app still uses its own uploaded or demo data.

## 1. Update files and create a token on your own PC

Extract ZIP contents directly into the existing project root. Preserve .env and
your data. The gateway uses a separate Compose project and does not touch team DB.

In PowerShell at the project root:

```powershell
.\.venv\Scripts\python.exe scripts/init_gateway.py
docker compose -f compose.gateway.yml up --build -d
.\.venv\Scripts\python.exe scripts/check_gateway.py
```

Expect two PASS lines, including successful Ollama JSON generation. On failure:

```powershell
docker compose -f compose.gateway.yml logs --tail 30 gateway
```

The generated .env.gateway contains your secret. Do not commit, upload, or
screenshot it. Running init again preserves the existing secret.
The gateway accepts only qwen2.5:7b by default, only POST /api/chat and authenticated
GET /health. It rejects unknown models and management routes, limits request size,
forces non-streaming output and generation limits, permits one active generation,
and caps authenticated requests at twenty per minute globally per gateway process.
These bounds do not make a public web app abuse-proof: its visitors can still
request work through the app. Share the demo deliberately and stop it when finished.

## 2. Open a temporary HTTPS tunnel

Download Windows 64-bit cloudflared from the official downloads page:
https://developers.cloudflare.com/tunnel/downloads/
Save it as cloudflared.exe in the project root. In a second PowerShell window:

```powershell
.\cloudflared.exe tunnel --url http://127.0.0.1:8787
```

Keep this terminal open. Copy the generated https://...trycloudflare.com URL.
Use port 8787 (the gateway), not Ollama's port 11434.
Quick Tunnels are for testing and generate a temporary hostname. Restarting a
tunnel may change its URL; update the hosted app Secrets accordingly. Slow cold
model starts can time out. Prewarm the model with check_gateway.py before a demo.
If a cloudflared configuration file prevents Quick Tunnel startup, consult the
official Quick Tunnel instructions; do not delete existing tunnel configurations.
https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/

## 3. Push application changes

```powershell
git add -- .gitignore .dockerignore src/agent.py src/analyst.py src/ollama_transport.py ollama_gateway.py compose.gateway.yml scripts/init_gateway.py scripts/check_gateway.py tests/test_gateway.py docs/LOCAL_LLM_WEB.md README.md
git diff --cached --stat
git commit -m "Add authenticated local Ollama demo gateway"
git push
```

Inspect staged files; never include .env.gateway, secrets.toml or cloudflared.exe.
These are explicitly ignored. Wait for the hosted app to receive the updated code.
No GitHub changes are performed automatically by downloading this ZIP.

## 4. Configure Streamlit Community Cloud Secrets

Open Manage app -> Settings -> Secrets. Merge these root-level TOML keys with
existing settings (do not erase unrelated secrets):

```toml
OLLAMA_URL = "https://YOUR-TUNNEL.trycloudflare.com/api/chat"
OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_GATEWAY_TOKEN = "YOUR-GENERATED-TOKEN"
```

Open .env.gateway locally in an editor and copy only the value after
GATEWAY_TOKEN= into OLLAMA_GATEWAY_TOKEN. Never send the token in chat.
The token authenticates app-to-gateway calls; HTTPS protects it in transit.
The client refuses redirects to prevent forwarding credentials to another host.
https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management

Save settings and restart the hosted app if needed. Choose Ollama and qwen2.5:7b.
Ask a segment-comparison question. Intent should report Ollama structured intent
when successful. A numeric-narrative rejection is still possible: Python then
supplies the interpretation. V29 traces show the actual fallback source.
401 means a missing/wrong token; 403 means model mismatch; 429 means busy/rate
limit; 502 means local model/upstream trouble; a connection error may mean the
tunnel or PC is offline. Do not change firewall rules just to bypass these errors.

## 5. Stop after the demonstration

Ctrl+C in the cloudflared terminal closes external access. Stop the gateway:

```powershell
docker compose -f compose.gateway.yml down
```

Team database and app remain unaffected. To rotate a leaked token, replace the
token locally with a new random value, recreate the gateway and update Cloud Secrets.
Do not reuse your database password as the gateway token.
