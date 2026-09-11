"""Verify local gateway auth and generation without printing secrets."""
import json
from pathlib import Path
from urllib import request, error

root = Path(__file__).resolve().parents[1]
settings = dict(line.split('=', 1) for line in (root / '.env.gateway').read_text().splitlines()
                if '=' in line and not line.startswith('#'))
base = 'http://127.0.0.1:8787'
try:
    request.urlopen(base + '/health', timeout=5)
    raise SystemExit('FAILED: gateway allowed unauthenticated health request.')
except error.HTTPError as exc:
    if exc.code != 401:
        raise SystemExit(f'Unexpected HTTP status: {exc.code}')
print('PASS: request without token rejected.')
headers = {'Authorization': 'Bearer ' + settings['GATEWAY_TOKEN'],
           'Content-Type': 'application/json'}
payload = {'model': settings['GATEWAY_MODEL'], 'stream': False,
           'messages': [{'role': 'user', 'content': 'Return JSON with ok equal to true.'}],
           'format': {'type': 'object', 'properties': {'ok': {'type': 'boolean'}},
                      'required': ['ok'], 'additionalProperties': False}}
try:
    req = request.Request(base + '/api/chat', data=json.dumps(payload).encode(), headers=headers)
    with request.urlopen(req, timeout=120) as response:
        content = json.loads(response.read())['message']['content']
    if json.loads(content) != {'ok': True}:
        raise ValueError('Unexpected generation')
    print('PASS: authenticated gateway reached Ollama and returned valid JSON.')
except Exception:
    raise SystemExit('Generation failed. Check gateway logs, Ollama, and model configuration.')
