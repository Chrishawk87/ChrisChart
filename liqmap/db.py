"""One SQLite connection per thread, because sharing one is not safe.

`sqlite3.connect(check_same_thread=False)` does not make a connection thread
safe. It only removes the check that would have told you. The connection
carries a prepared-statement cache and per-cursor state, and two threads
using it at once corrupt each other's results.

That is not theoretical. This service runs a background worker that sweeps
and settles while FastAPI serves dashboard polls from a threadpool, and
hammering a shared connection from readers and writers for twelve seconds
produces:

    DatabaseError: another row available
    DatabaseError: no more rows available
    SystemError: error return without exception set
    OperationalError: cannot start a transaction within a transaction
    IndexError: tuple index out of range

The last one is the nastiest, because it surfaces far from its cause. A
`sqlite3.Row` holds the column description separately from the value tuple;
when another thread resets the shared statement mid-flight the two disagree,
and the failure appears as `dict(row)` raising IndexError inside whatever
innocent query happened to be running. Chasing that back to threading is a
bad afternoon, and the endpoint just 500s with a message that names the
wrong module.

The fix is one connection per thread against the same file, which is how
SQLite is designed to be used concurrently:

    WAL          readers do not block the writer and the writer does not
                 block readers. Without it, any read during a write gets
                 SQLITE_BUSY.
    busy_timeout waits for a lock instead of failing instantly, so a slow
                 sweep does not fail a dashboard poll.

Threads that come and go leave their connections behind; that is fine, as
they are closed when the thread's locals are collected, and this service has
a small fixed set of long-lived threads anyway.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path


class ThreadedDB:
    """A SQLite file with a connection per thread.

    Subclasses (or holders) use `self._conn` exactly as they used a shared
    connection before; it resolves to the calling thread's own.
    """

    def __init__(self, path: str | Path, schema: str = "") -> None:
        self.path = str(path)
        parent = Path(self.path).parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)
        self._schema = schema
        self._local = threading.local()
        # Build one now so a bad path or a broken schema fails at startup
        # rather than on the first request from some worker thread.
        self._bootstrap(self._conn)

    def _new_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False,
                               timeout=30.0)
        conn.row_factory = sqlite3.Row
        # WAL is a property of the database file, not the connection, so
        # setting it on each one is harmless and makes no assumption about
        # which connection got there first.
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.Error:
            # A filesystem that cannot do WAL (some network mounts) still
            # works, just with less concurrency. Not worth refusing to start.
            pass
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _bootstrap(self, conn: sqlite3.Connection) -> None:
        """Run once per connection. Overridden where migrations are needed."""
        if self._schema:
            conn.executescript(self._schema)
            conn.commit()

    @property
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._new_connection()
            self._local.conn = conn
            # A thread that appears after startup still needs the schema
            # applied to its own connection: `CREATE TABLE IF NOT EXISTS` is
            # cheap and idempotent, and skipping it would break a fresh
            # thread against a fresh file.
            self._bootstrap(conn)
        return conn

    def close(self) -> None:
        """Close this thread's connection.

        Other threads' connections are left alone -- closing them from here
        is exactly the cross-thread use this class exists to prevent. They
        are released when their threads end.
        """
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
