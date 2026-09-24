"""No test may touch the network.

A suite that reaches the live API passes or fails depending on whether
Hyperliquid is up, whether the sandbox allows egress, and how many requests
the last run used. That is not a test suite, it is a weather report -- and
the failure lands on whichever test happened to make the call, which is
almost never the one with the bug.

This turns any outbound connection into an immediate, clearly named error
that points at the test making it. A test that needs market data should use
a fixture or a fake client; there is no case where the real answer is a
live request.

Unix-domain sockets are left alone: that is how a local TestClient talks to
itself.
"""

from __future__ import annotations

import socket

import pytest

_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex


class NetworkUsedInTest(AssertionError):
    pass


def _blocked(self, address, *a, **kw):
    # AF_UNIX carries no (host, port) tuple and never leaves the machine.
    if self.family == getattr(socket, "AF_UNIX", object()):
        return _real_connect(self, address, *a, **kw)
    host = address[0] if isinstance(address, tuple) else address
    if host in ("127.0.0.1", "::1", "localhost"):
        return _real_connect(self, address, *a, **kw)
    raise NetworkUsedInTest(
        f"this test tried to reach {address!r}. Tests must not use the "
        f"network — seed the feed or inject a fake client instead.")


def _blocked_ex(self, address, *a, **kw):
    try:
        _blocked(self, address, *a, **kw)
    except NetworkUsedInTest:
        raise
    return 0


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked_ex)
    yield
