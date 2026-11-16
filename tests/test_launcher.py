import urllib.error

import pytest

from scripts import run


@pytest.mark.parametrize("token", [None, "test-key"])
def test_browser_opens_only_after_health_response(monkeypatch, token):
    opened = []
    waits = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    class Opener:
        calls = 0

        def open(self, url, timeout):
            assert url == "http://127.0.0.1:8000/api/health" and timeout == 1
            self.calls += 1
            if self.calls == 1:
                raise urllib.error.URLError("server not ready")
            return Response()

    monkeypatch.setattr(run.urllib.request, "build_opener", lambda *_: Opener())
    monkeypatch.setattr(run.webbrowser, "open", opened.append)
    monkeypatch.setattr(run.time, "sleep", waits.append)
    run.open_browser_when_ready("http://127.0.0.1:8000", token=token)
    expected = (
        "http://127.0.0.1:8000/#key=test-key&role=viewer" if token else "http://127.0.0.1:8000"
    )
    assert opened == [expected]
    assert waits == [0.25]
