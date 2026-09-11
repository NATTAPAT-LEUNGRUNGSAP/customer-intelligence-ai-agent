import json
import threading
from urllib import request, error
from unittest.mock import MagicMock

import pytest
import ollama_gateway as gateway
from src import ollama_transport

TOKEN = 'test-token-' + 'x' * 40


def payload():
    return {'model': 'qwen2.5:7b', 'messages': [{'role': 'user', 'content': 'hello'}],
            'stream': False, 'format': {'type': 'object'},
            'options': {'num_predict': 999999}}


@pytest.fixture
def running_gateway():
    server = gateway.make_server(('127.0.0.1', 0), TOKEN, 'qwen2.5:7b', 'http://unused/api/chat')
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{server.server_port}'
    server.shutdown()
    server.server_close()
    thread.join()


@pytest.mark.parametrize('token', ['', 'wrong'])
def test_gateway_rejects_bad_token(running_gateway, token):
    req = request.Request(running_gateway + '/api/chat', data=json.dumps(payload()).encode(),
                          headers={'Authorization': 'Bearer ' + token})
    with pytest.raises(error.HTTPError) as result:
        request.build_opener().open(req)
    assert result.value.code == 401


def test_gateway_blocks_admin_routes(running_gateway):
    req = request.Request(running_gateway + '/api/pull', data=b'{}',
                          headers={'Authorization': 'Bearer ' + TOKEN})
    with pytest.raises(error.HTTPError) as result:
        request.build_opener().open(req)
    assert result.value.code == 404


def test_gateway_forces_limits_and_model():
    assert gateway.validate_payload(payload(), 'qwen2.5:7b')['options']['num_predict'] == 1536
    with pytest.raises(PermissionError):
        gateway.validate_payload(payload(), 'other-model')
    bad = payload()
    bad['stream'] = True
    with pytest.raises(ValueError):
        gateway.validate_payload(bad, 'qwen2.5:7b')


def test_authenticated_transport_round_trip(running_gateway, monkeypatch):
    captured = []
    def fake_upstream(req, timeout):
        captured.append(req)
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"message":{"content":"{\\"ok\\":true}"}}'
        return response
    monkeypatch.setattr(gateway.request, 'urlopen', fake_upstream)
    monkeypatch.setenv('OLLAMA_URL', running_gateway + '/api/chat')
    monkeypatch.setenv('OLLAMA_GATEWAY_TOKEN', TOKEN)
    assert ollama_transport.chat(json.dumps(payload()).encode()) == '{"ok":true}'
    assert len(captured) == 1
    assert not captured[0].has_header('Authorization')
    assert json.loads(captured[0].data)['options']['num_predict'] == 1536


def test_transport_refuses_token_over_remote_http(monkeypatch):
    monkeypatch.setenv('OLLAMA_URL', 'http://example.com/api/chat')
    monkeypatch.setenv('OLLAMA_GATEWAY_TOKEN', TOKEN)
    with pytest.raises(RuntimeError, match='HTTPS'):
        ollama_transport.chat(b'{}')


def test_gateway_refuses_empty_secret():
    with pytest.raises(ValueError):
        gateway.make_server(('127.0.0.1', 0), '', 'model', 'unused')
