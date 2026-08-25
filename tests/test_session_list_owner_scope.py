"""Session-list owner isolation and stale-incognito purge regressions."""

import sys
import tempfile
import types
import uuid
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

import core.database as cdb
from core.database import ChatMessage as DbMessage
from core.database import Session as DbSession
from routes import session_routes
from src.retrieval_policy import BEEPY_SERVICE_TOKEN_PROFILE, BEEPY_SHARED_RAG_SCOPE


_TMPDB = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_ENGINE = create_engine(
    f"sqlite:///{_TMPDB.name}",
    connect_args={"check_same_thread": False},
    poolclass=NullPool,
)
cdb.Base.metadata.create_all(_ENGINE)
_TS = sessionmaker(bind=_ENGINE, autoflush=False, autocommit=False)


def _stub_multipart_if_missing(monkeypatch):
    try:
        import python_multipart  # noqa: F401
        return
    except ImportError:
        pass
    stub = types.ModuleType("python_multipart")
    stub.__version__ = "0.0.20"
    monkeypatch.setitem(sys.modules, "python_multipart", stub)


def test_list_sessions_excludes_other_users_sessions(monkeypatch):
    import routes.session_routes as sr
    from unittest.mock import MagicMock

    _stub_multipart_if_missing(monkeypatch)
    monkeypatch.setattr(sr, "SessionLocal", _TS)
    monkeypatch.setattr(sr, "effective_user", lambda request: "alice")

    alice_id = str(uuid.uuid4())
    bob_id = str(uuid.uuid4())
    db = _TS()
    try:
        db.query(DbSession).delete()
        db.add(DbSession(id=alice_id, owner="alice", name="alice session",
                         endpoint_url="http://localhost", model="gpt-4", archived=False))
        db.add(DbSession(id=bob_id, owner="bob", name="bob session",
                         endpoint_url="http://localhost", model="gpt-4", archived=False))
        db.commit()
    finally:
        db.close()

    alice_session = MagicMock(id=alice_id, name="alice session",
                              model="gpt-4", endpoint_url="http://localhost",
                              rag=False, archived=False)
    sm = MagicMock()
    sm.get_sessions_for_user.return_value = {alice_id: alice_session}
    router = sr.setup_session_routes(sm, {})
    endpoint = next(r.endpoint for r in router.routes
                    if getattr(r, "path", "") == "/api/sessions"
                    and "GET" in getattr(r, "methods", set()))

    result = endpoint(request=MagicMock())
    returned_ids = {s["id"] for s in result}
    assert alice_id in returned_ids
    assert bob_id not in returned_ids


def test_auto_sort_skip_llm_cleans_owner_stamped_sessions_when_auth_disabled(monkeypatch):
    import routes.session_routes as sr
    from unittest.mock import MagicMock

    _stub_multipart_if_missing(monkeypatch)
    monkeypatch.setenv("AUTH_ENABLED", "false")
    monkeypatch.setattr(sr, "SessionLocal", _TS)
    monkeypatch.setattr(sr, "effective_user", lambda request: None)

    sid = str(uuid.uuid4())
    old_time = cdb.utcnow_naive() - timedelta(hours=2)
    db = _TS()
    try:
        db.query(DbMessage).delete()
        db.query(DbSession).delete()
        db.add(DbSession(
            id=sid,
            owner="alice",
            name="New chat",
            endpoint_url="http://localhost",
            model="gpt-4",
            archived=False,
            message_count=1,
            created_at=old_time,
            updated_at=old_time,
            last_message_at=old_time,
            last_accessed=old_time,
        ))
        db.add(DbMessage(
            id="m-" + uuid.uuid4().hex,
            session_id=sid,
            role="user",
            content="hi",
            timestamp=old_time,
        ))
        db.commit()
    finally:
        db.close()

    session = MagicMock(id=sid, name="New chat", model="gpt-4", endpoint_url="http://localhost", rag=False, archived=False)
    sm = MagicMock()
    sm.get_sessions_for_user.return_value = {sid: session}
    router = sr.setup_session_routes(sm, {})
    endpoint = next(r.endpoint for r in router.routes
                    if getattr(r, "path", "") == "/api/sessions/auto-sort"
                    and "POST" in getattr(r, "methods", set()))

    result = endpoint(request=MagicMock(), skip_llm=True)

    assert result["deleted_throwaway"] == 1
    db = _TS()
    try:
        assert db.query(DbSession).filter(DbSession.id == sid).first() is None
    finally:
        db.close()


class Column:
    def __init__(self, name):
        self.name = name

    def __eq__(self, value):
        return lambda row: getattr(row, self.name) == value

    def __lt__(self, value):
        return lambda row: getattr(row, self.name) < value

    def in_(self, values):
        return lambda row: getattr(row, self.name) in values


class SessionRow:
    owner = Column("owner")
    name = Column("name")
    created_at = Column("created_at")


class MessageRow:
    session_id = Column("session_id")


class Query:
    def __init__(self, rows):
        self.rows = rows
        self.predicates = []
        self.deleted = False

    def filter(self, *predicates):
        self.predicates.extend(predicates)
        return self

    def all(self):
        return [row for row in self.rows if all(predicate(row) for predicate in self.predicates)]

    def delete(self):
        self.deleted = True
        return len(self.all())


class FakeDb:
    def __init__(self, sessions, messages):
        self.sessions = sessions
        self.messages = messages
        self.pending_delete = []
        self.committed = False
        self.rolled_back = False

    def query(self, model):
        return Query(self.sessions if model is SessionRow else self.messages)

    def delete(self, row):
        self.pending_delete.append(row)

    def commit(self):
        for row in self.pending_delete:
            self.sessions.remove(row)
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        pass


def _request(*, service=False, owner="alice"):
    state = SimpleNamespace(api_token=service, current_user=owner)
    if service:
        state.api_token_owner = "service:beepy"
        state.api_token_scopes = ["chat", BEEPY_SHARED_RAG_SCOPE]
        state.api_token_profile = BEEPY_SERVICE_TOKEN_PROFILE
    return SimpleNamespace(state=state)


def test_stale_cleanup_deletes_only_authenticated_owners_rows(monkeypatch):
    now = datetime.now()
    alice_old = SimpleNamespace(id="alice-old", owner="alice", name="Incognito", created_at=now - timedelta(hours=1))
    bob_old = SimpleNamespace(id="bob-old", owner="bob", name="Nobody", created_at=now - timedelta(hours=1))
    alice_recent = SimpleNamespace(id="alice-live", owner="alice", name="Incognito", created_at=now)
    sessions = [alice_old, bob_old, alice_recent]
    db = FakeDb(sessions, [])

    monkeypatch.setattr(session_routes, "DbSession", SessionRow)
    monkeypatch.setattr(session_routes, "SessionLocal", lambda: db)
    monkeypatch.setattr("core.database.ChatMessage", MessageRow)
    monkeypatch.setattr(session_routes, "utcnow_naive", lambda: now)
    manager = SimpleNamespace(sessions={row.id: row for row in sessions})

    count = session_routes._purge_stale_incognito_sessions(_request(), manager, "alice")

    assert count == 1
    assert db.committed is True
    assert [row.id for row in sessions] == ["bob-old", "alice-live"]
    assert set(manager.sessions) == {"bob-old", "alice-live"}


def test_service_token_cannot_trigger_user_session_cleanup(monkeypatch):
    def forbidden_db():
        raise AssertionError("service cleanup must not open the session database")

    monkeypatch.setattr(session_routes, "SessionLocal", forbidden_db)
    manager = SimpleNamespace(sessions={"foreign": object()})
    count = session_routes._purge_stale_incognito_sessions(
        _request(service=True), manager, "service:beepy"
    )
    assert count == 0
    assert set(manager.sessions) == {"foreign"}


def test_missing_owner_fails_closed_without_cleanup(monkeypatch):
    monkeypatch.setattr(
        session_routes,
        "SessionLocal",
        lambda: (_ for _ in ()).throw(AssertionError("database must not open")),
    )
    assert session_routes._purge_stale_incognito_sessions(
        _request(), SimpleNamespace(sessions={}), None
    ) == 0
