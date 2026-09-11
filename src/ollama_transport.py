"""Ollama transport with optional gateway authentication and no redirects."""
import json
import os
from urllib import request, error, parse


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def chat(payload: bytes) -> str:
    endpoint = os.getenv('OLLAMA_URL', 'http://localhost:11434/api/chat')
    token = os.getenv('OLLAMA_GATEWAY_TOKEN', '').strip()
    location = parse.urlsplit(endpoint)
    if token and location.scheme != 'https' and location.hostname not in {'localhost', '127.0.0.1'}:
        raise RuntimeError('Authenticated remote Ollama requires HTTPS.')
    headers = {'Content-Type': 'application/json'}
    if token:
        headers['Authorization'] = 'Bearer ' + token
    try:
        with request.build_opener(NoRedirect).open(
            request.Request(endpoint, data=payload, headers=headers), timeout=120
        ) as response:
            return json.loads(response.read())['message']['content']
    except error.HTTPError as exc:
        messages = {401: 'Gateway token missing or incorrect.',
                    403: 'Gateway rejected the requested model.',
                    429: 'Gateway busy or request limit reached. Try again later.',
                    502: 'Gateway could not complete the Ollama request.',
                    400: 'Gateway rejected the request format.'}
        raise RuntimeError(messages.get(exc.code, f'Ollama HTTP error {exc.code}.')) from None
    except Exception:
        raise RuntimeError('Could not reach Ollama. Check the endpoint, tunnel, and local model.') from None
