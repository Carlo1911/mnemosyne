"""
Tests for episodic forget (issue #959).

Pre-fix, ``Mnemosyne.forget()`` only deleted from the legacy ``memories``
table and ``working_memory``: an episodic row ID always resolved to
``not_found`` through ``mnemosyne_forget``/``forget()``, leaving
audit-surfaced episodic rows unmanageable by ID.

Post-fix, ``forget()`` falls back to ``BeamMemory.forget_episodic()``
when neither working_memory nor the legacy mirror claim the ID in the
caller's session scope. The fallback keeps the same trust boundary as
``forget_working`` (see E6.a there): the session-scoped episodic DELETE
(``session_id = ? OR scope = 'global'``) authorizes the cascade, so a
foreign session's private row is left untouched while a global row may
be removed cross-session.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mnemosyne.core.beam import BeamMemory, _vec_available, _vec_insert
from mnemosyne.core.memory import Mnemosyne


def _seed_episodic(conn, mem_id: str, session_id: str, scope: str = "session") -> int:
    """Insert a session/global episodic row; return its rowid."""
    conn.execute(
        "INSERT INTO episodic_memory "
        "(id, content, source, timestamp, session_id, importance, scope) "
        "VALUES (?, 'episodic content', 'test', datetime('now'), ?, 0.6, ?)",
        (mem_id, session_id, scope),
    )
    conn.commit()
    return conn.execute(
        "SELECT rowid FROM episodic_memory WHERE id = ?", (mem_id,)
    ).fetchone()[0]


def _seed_cascade(conn, mem_id: str, rowid: int | None = None) -> None:
    """Attach annotation + embedding + gist rows (and a vec row when given)."""
    conn.execute(
        "INSERT INTO annotations (memory_id, kind, value) VALUES (?, 'mentions', 'test')",
        (mem_id,),
    )
    conn.execute(
        "INSERT INTO memory_embeddings (memory_id, embedding_json) VALUES (?, '[]')",
        (mem_id,),
    )
    conn.execute(
        "INSERT INTO gists (id, text, memory_id) VALUES (?, 'gist text', ?)",
        (f"gist-{mem_id}", mem_id),
    )
    if rowid is not None:
        _vec_insert(conn, rowid, [0.1] * 384)
    conn.commit()


def _gist_count(conn, mem_id: str) -> int:
    """Count gists rows for a memory (-1 when the table is unavailable)."""
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM gists WHERE memory_id = ?", (mem_id,)
        ).fetchone()[0]
    except Exception:
        return -1


def _counts(conn, mem_id: str):
    """Count the episodic row and its cascade rows (-1 for missing vec)."""
    row = conn.execute(
        "SELECT COUNT(*) FROM episodic_memory WHERE id = ?", (mem_id,)
    ).fetchone()[0]
    ann = conn.execute(
        "SELECT COUNT(*) FROM annotations WHERE memory_id = ?", (mem_id,)
    ).fetchone()[0]
    emb = conn.execute(
        "SELECT COUNT(*) FROM memory_embeddings WHERE memory_id = ?", (mem_id,)
    ).fetchone()[0]
    try:
        vec = conn.execute("SELECT COUNT(*) FROM vec_episodes").fetchone()[0]
    except Exception:
        vec = -1
    return row, ann, emb, vec


def test_forget_deletes_own_episodic_row_and_cascade(tmp_path: Path):
    """Own episodic row goes with its annotations/embeddings/gists/vec."""
    db = tmp_path / "forget_ep.db"
    mem = Mnemosyne(session_id="sess-a", db_path=db)
    rowid = _seed_episodic(mem.conn, "em-1", "sess-a")
    _seed_cascade(mem.conn, "em-1", rowid if _vec_available(mem.conn) else None)

    assert mem.forget("em-1") is True

    row, ann, emb, vec = _counts(mem.conn, "em-1")
    assert row == 0
    assert ann == 0
    assert emb == 0
    if vec != -1:
        assert vec == 0
    if _gist_count(mem.conn, "em-1") != -1:
        assert _gist_count(mem.conn, "em-1") == 0


def test_forget_global_episodic_row_cross_session(tmp_path: Path):
    """A global episodic row may be removed from another session."""
    db = tmp_path / "forget_ep.db"
    writer = Mnemosyne(session_id="sess-a", db_path=db)
    _seed_episodic(writer.conn, "em-global", "sess-a", scope="global")

    other = Mnemosyne(session_id="sess-b", db_path=db)
    assert other.forget("em-global") is True
    assert _counts(other.conn, "em-global")[0] == 0


def test_forget_foreign_private_episodic_row_keeps_everything(tmp_path: Path):
    """A foreign session's private row (and its cascade) survives forget."""
    db = tmp_path / "forget_ep.db"
    writer = Mnemosyne(session_id="sess-a", db_path=db)
    rowid = _seed_episodic(writer.conn, "em-priv", "sess-a", scope="session")
    _seed_cascade(writer.conn, "em-priv", rowid if _vec_available(writer.conn) else None)

    other = Mnemosyne(session_id="sess-b", db_path=db)
    assert other.forget("em-priv") is False

    row, ann, emb, vec = _counts(other.conn, "em-priv")
    assert row == 1
    assert ann == 1
    assert emb == 1
    if vec != -1:
        assert vec == 1
    if _gist_count(other.conn, "em-priv") != -1:
        assert _gist_count(other.conn, "em-priv") == 1


def test_forget_unknown_id_returns_false(tmp_path: Path):
    """Unknown IDs still resolve to False without side effects."""
    mem = Mnemosyne(session_id="sess-a", db_path=tmp_path / "forget_ep.db")
    assert mem.forget("does-not-exist") is False


def test_beam_forget_episodic_cascade_is_atomic_on_miss(tmp_path: Path):
    """A miss must not disturb neighboring rows or their cascade data."""
    beam = BeamMemory(session_id="sess-a", db_path=tmp_path / "forget_ep.db")
    rowid = _seed_episodic(beam.conn, "em-keep", "sess-a")
    _seed_cascade(beam.conn, "em-keep", rowid if _vec_available(beam.conn) else None)

    assert beam.forget_episodic("em-missing") is False

    row, ann, emb, vec = _counts(beam.conn, "em-keep")
    assert (row, ann, emb) == (1, 1, 1)
    if vec != -1:
        assert vec == 1


def test_forget_episodic_emits_event_only_on_success(tmp_path: Path, monkeypatch):
    """MEMORY_INVALIDATED fires for a deleted row, never for a miss."""
    mem = Mnemosyne(session_id="sess-a", db_path=tmp_path / "forget_ep.db")
    events = []
    monkeypatch.setattr(
        mem, "_emit_wrapper", lambda *args, **kwargs: events.append((args, kwargs)))
    _seed_episodic(mem.conn, "em-evt", "sess-a")

    assert mem.forget("em-evt") is True
    assert mem.forget("em-missing") is False
    assert events == [(("MEMORY_INVALIDATED", "em-evt"), {})]


def test_forget_episodic_cascade_failure_rolls_back(tmp_path: Path):
    """A mid-cascade failure aborts the whole delete; every row survives."""
    beam = BeamMemory(session_id="sess-a", db_path=tmp_path / "forget_ep.db")
    rowid = _seed_episodic(beam.conn, "em-rb", "sess-a")
    vec_seeded = _vec_available(beam.conn)
    _seed_cascade(beam.conn, "em-rb", rowid if vec_seeded else None)
    before = _counts(beam.conn, "em-rb")
    gist_before = _gist_count(beam.conn, "em-rb")
    beam.conn.execute(
        "CREATE TRIGGER fail_ann_delete BEFORE DELETE ON annotations "
        "BEGIN SELECT RAISE(ABORT, 'forced annotations failure'); END"
    )
    beam.conn.commit()

    with pytest.raises(Exception, match="forced annotations failure"):
        beam.forget_episodic("em-rb")

    assert _counts(beam.conn, "em-rb") == before
    assert _gist_count(beam.conn, "em-rb") == gist_before
    assert before[0] == 1
    if vec_seeded:
        assert before[3] == 1
