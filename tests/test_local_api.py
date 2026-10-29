"""Local edition works without credentials; upload boundaries stay enforced."""
import io
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import server


class Processor:
    def process(self, source, directory, options, update):
        update(50, '处理')
        (directory / 'result.psd').write_bytes(b'8BPS')
        return {'files': {'psd': 'result.psd'}}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'model_status', lambda: [{'name': 'test-model', 'ready': True}])
    with TestClient(server.create_app(tmp_path / 'jobs', Processor())) as session:
        yield session


def png():
    stream = io.BytesIO()
    Image.new('RGB', (100, 80), 'orange').save(stream, format='PNG')
    return stream.getvalue()


def test_no_credentials_are_needed_when_models_are_ready(client):
    health = client.get('/api/health').json()
    assert health['ready'] is True, health
    assert health['edition'] == 'opensource'
    assert 'license' not in health


def test_home_is_the_workbench_and_has_no_activation_entry(client):
    page = client.get('/')
    assert page.status_code == 200
    assert 'uploadForm' in page.text
    assert '/activation' not in page.text
    assert client.get('/api/license').status_code == 404
    assert client.get('/activation').status_code == 404


def test_upload_finishes_and_downloads_without_activation(client):
    response = client.post('/api/jobs', files={'files': ('../../image.png', png(), 'image/png')})
    assert response.status_code == 200, response.text
    job = response.json()['jobs'][0]
    assert job['name'] == 'image.png'
    for _ in range(100):
        job = client.get('/api/jobs/' + job['id']).json()
        if job['status'] in ('completed', 'failed'):
            break
        time.sleep(.01)
    assert job['status'] == 'completed', job
    assert client.get(job['result']['files']['psd']).content == b'8BPS'
    assert client.get(f"/api/jobs/{job['id']}/files/status.json").status_code == 404


def test_foreign_origin_invalid_image_and_oversized_upload_are_rejected(client):
    assert client.post('/api/jobs', files={'files': ('x.png', png())},
                       headers={'Origin': 'https://untrusted.invalid'}).status_code == 403
    assert client.post('/api/jobs', files={'files': ('bad.png', b'not an image')}).status_code == 400
    assert client.post('/api/jobs', content=b'x',
                       headers={'Content-Length': str(81 * 1024 * 1024)}).status_code == 413
    assert client.get('/api/jobs').json()['jobs'] == []


def test_missing_models_are_reported_without_authorization_checks(tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'model_status', lambda: [{'name': 'missing', 'ready': False}])
    with TestClient(server.create_app(tmp_path / 'jobs', Processor())) as client:
        health = client.get('/api/health').json()
        assert health['ready'] is False
        assert 'missing' in health['message']


def test_concurrent_upload_reserves_capacity_before_image_validation(client, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    entered, release = threading.Event(), threading.Event()
    calls = []
    original = server.load_image
    def slow_first(path):
        calls.append(path)
        if len(calls) == 1:
            entered.set()
            assert release.wait(5)
        return original(path)
    monkeypatch.setattr(server, 'load_image', slow_first)
    store = client.app.state.store
    with store.lock:
        for i in range(29):
            ident = f'{i:032x}'
            store.jobs[ident] = {'id': ident, 'status': 'queued', 'created_at': '2026-01-01'}
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(client.post, '/api/jobs', files={'files': ('first.png', png())})
        try:
            assert entered.wait(3)
            second = client.post('/api/jobs', files={'files': ('second.png', png())})
            assert second.status_code == 429, second.text
        finally:
            release.set()
        assert first.result(timeout=3).status_code == 200
