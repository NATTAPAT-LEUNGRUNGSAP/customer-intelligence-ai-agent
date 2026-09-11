"""Bounded, authenticated Ollama gateway for a scheduled portfolio demo.

Expose through HTTPS Tunnel only. This is not a general-purpose Ollama proxy.
"""
import hmac
import json
import os
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import request

MAX_BODY = 131072


def validate_payload(value, model):
    if not isinstance(value, dict) or set(value) - {'model', 'messages', 'format', 'options', 'stream'}:
        raise ValueError('Unsupported request fields')
    if value.get('model') != model:
        raise PermissionError('Model not allowed')
    messages = value.get('messages')
    if not isinstance(messages, list) or not 1 <= len(messages) <= 8:
        raise ValueError('Invalid messages')
    for item in messages:
        if not isinstance(item, dict) or set(item) != {'role', 'content'}:
            raise ValueError('Invalid message fields')
        if item['role'] not in {'system', 'user', 'assistant'} or not isinstance(item['content'], str):
            raise ValueError('Invalid message')
    if sum(len(m['content']) for m in messages) > 24000:
        raise ValueError('Prompt too long')
    if value.get('stream') is not False or not isinstance(value.get('format'), dict):
        raise ValueError('Non-streaming schema output required')
    # Server owns generation limits, regardless of user-supplied options.
    return {'model': model, 'messages': messages, 'format': value['format'],
            'stream': False, 'options': {'temperature': 0.1, 'num_predict': 1536, 'num_ctx': 8192}}


def make_server(address, token, model, upstream):
    if len(token) < 32 or not token.isascii():
        raise ValueError('Set GATEWAY_TOKEN to a random ASCII secret of at least 32 characters.')
    busy = threading.BoundedSemaphore(1)
    recent = deque()
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(15)

        def log_message(self, *args):
            pass  # Do not log credentials, prompt contents, or arbitrary paths.

        def reply(self, status, value):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def authorized(self):
            expected = ('Bearer ' + token).encode()
            return hmac.compare_digest(self.headers.get('Authorization', '').encode(), expected)

        def do_GET(self):
            if not self.authorized():
                return self.reply(401, {'error': 'Unauthorized'})
            if self.path != '/health':
                return self.reply(404, {'error': 'Not found'})
            self.reply(200, {'gateway': 'ready', 'model': model,
                             'note': 'Gateway only; does not verify Ollama generation.'})

        def do_POST(self):
            if not self.authorized():
                return self.reply(401, {'error': 'Unauthorized'})
            if self.path != '/api/chat':
                return self.reply(404, {'error': 'Not found'})
            if self.headers.get('Transfer-Encoding'):
                return self.reply(400, {'error': 'Unsupported encoding'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= MAX_BODY:
                    return self.reply(413, {'error': 'Invalid body size'})
                payload = validate_payload(json.loads(self.rfile.read(size)), model)
            except PermissionError:
                return self.reply(403, {'error': 'Model not allowed'})
            except (ValueError, TypeError, OSError):
                return self.reply(400, {'error': 'Invalid request'})
            with lock:
                now = time.monotonic()
                while recent and now - recent[0] >= 60:
                    recent.popleft()
                if len(recent) >= 20:
                    return self.reply(429, {'error': 'Request limit reached'})
                recent.append(now)
            if not busy.acquire(blocking=False):
                return self.reply(429, {'error': 'Model busy'})
            try:
                req = request.Request(upstream, data=json.dumps(payload).encode(),
                                      headers={'Content-Type': 'application/json'})
                with request.urlopen(req, timeout=90) as response:
                    result = json.loads(response.read(1048576))
                content = result['message']['content']
                if not isinstance(content, str):
                    raise ValueError('Invalid model output')
                self.reply(200, {'message': {'role': 'assistant', 'content': content}})
            except Exception:
                self.reply(502, {'error': 'Ollama request failed'})
            finally:
                busy.release()

    return ThreadingHTTPServer(address, Handler)


if __name__ == '__main__':
    server = make_server(('0.0.0.0', 8787), os.getenv('GATEWAY_TOKEN', ''),
                         os.getenv('GATEWAY_MODEL', 'qwen2.5:7b'),
                         'http://host.docker.internal:11434/api/chat')
    print('Authenticated demo gateway listening on port 8787.', flush=True)
    server.serve_forever()
