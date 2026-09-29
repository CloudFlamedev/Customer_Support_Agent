"""SQLite storage for tickets and their Flow results.

Deliberately simple: stdlib sqlite3, one table, JSON text columns for the
structured bits (trace, reasons, sources). Good enough for a portfolio
project; swap for Postgres + SQLAlchemy for real concurrent load.
"""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from support_triage.config import settings
from support_triage.flow import FlowResult
from support_triage.schemas import Ticket

DB_PATH = Path(settings.chroma_path).parent / "support_triage.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id       TEXT PRIMARY KEY,
    customer_id     TEXT NOT NULL,
    subject         TEXT NOT NULL,
    body            TEXT NOT NULL,
    status          TEXT NOT NULL,      -- pending | auto_sent | approved | rejected
    action          TEXT NOT NULL,      -- auto_send | human_review (from the Flow)
    confidence      REAL NOT NULL,
    category        TEXT,
    priority        INTEGER,
    sentiment       TEXT,
    draft_reply     TEXT,
    sources         TEXT,               -- JSON list
    reasons         TEXT,               -- JSON list
    trace           TEXT,               -- JSON list
    final_reply     TEXT,               -- set on approve (edited or original)
    review_note     TEXT,               -- set on reject
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
"""


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _conn() as conn:
        conn.execute(SCHEMA)


def save_result(ticket: Ticket, result: FlowResult) -> None:
    now = datetime.now(timezone.utc).isoformat()
    status = "auto_sent" if result.decision.action == "auto_send" else "pending"
    with _conn() as conn:
        conn.execute(
            """INSERT INTO tickets
               (ticket_id, customer_id, subject, body, status, action, confidence,
                category, priority, sentiment, draft_reply, sources, reasons, trace,
                final_reply, review_note, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(ticket_id) DO UPDATE SET
                 status=excluded.status, action=excluded.action, confidence=excluded.confidence,
                 category=excluded.category, priority=excluded.priority, sentiment=excluded.sentiment,
                 draft_reply=excluded.draft_reply, sources=excluded.sources, reasons=excluded.reasons,
                 trace=excluded.trace, updated_at=excluded.updated_at""",
            (
                ticket.ticket_id, ticket.customer_id, ticket.subject, ticket.body,
                status, result.decision.action, result.decision.confidence,
                result.triage.category.value if result.triage else None,
                result.triage.priority if result.triage else None,
                result.triage.sentiment if result.triage else None,
                result.draft.reply if result.draft else None,
                json.dumps(result.draft.sources) if result.draft else "[]",
                json.dumps(result.decision.reasons),
                json.dumps(result.trace),
                None, None, now, now,
            ),
        )


def list_tickets(status: str | None = None) -> list[dict]:
    with _conn() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM tickets WHERE status=? ORDER BY created_at DESC", (status,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM tickets ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]


def get_ticket(ticket_id: str) -> dict | None:
    with _conn() as conn:
        row = conn.execute("SELECT * FROM tickets WHERE ticket_id=?", (ticket_id,)).fetchone()
        return dict(row) if row else None


def approve_ticket(ticket_id: str, edited_reply: str | None) -> dict | None:
    with _conn() as conn:
        row = conn.execute("SELECT draft_reply FROM tickets WHERE ticket_id=?", (ticket_id,)).fetchone()
        if not row:
            return None
        final = edited_reply if edited_reply else row["draft_reply"]
        conn.execute(
            "UPDATE tickets SET status='approved', final_reply=?, updated_at=? WHERE ticket_id=?",
            (final, datetime.now(timezone.utc).isoformat(), ticket_id),
        )
    return get_ticket(ticket_id)


def reject_ticket(ticket_id: str, note: str | None) -> dict | None:
    with _conn() as conn:
        exists = conn.execute("SELECT 1 FROM tickets WHERE ticket_id=?", (ticket_id,)).fetchone()
        if not exists:
            return None
        conn.execute(
            "UPDATE tickets SET status='rejected', review_note=?, updated_at=? WHERE ticket_id=?",
            (note, datetime.now(timezone.utc).isoformat(), ticket_id),
        )
    return get_ticket(ticket_id)
