"""Create a private gateway token locally; never overwrite an existing token."""
from pathlib import Path
import secrets

path = Path(__file__).resolve().parents[1] / '.env.gateway'
try:
    with path.open('x', encoding='utf-8') as handle:
        handle.write('GATEWAY_TOKEN=' + secrets.token_urlsafe(32) + '\n')
        handle.write('GATEWAY_MODEL=qwen2.5:7b\n')
    path.chmod(0o600)
    print('Created .env.gateway. Keep it private; never commit or screenshot it.')
except FileExistsError:
    print('Existing .env.gateway retained.')
