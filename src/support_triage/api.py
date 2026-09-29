"""FastAPI app: submit tickets, list/view results, human review actions.

process_ticket() is synchronous and calls several LLMs in sequence, so a
POST /tickets request can take 20-40 seconds under Groq's free-tier limits.
Accepted trade-off for this project; a real deployment would push ticket
processing to a background queue and return "queued" immediately.
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from support_triage.db import approve_ticket, get_ticket, init_db, list_tickets, reject_ticket, save_result
from support_triage.flow import process_ticket
from support_triage.schemas import Ticket

app = FastAPI(title="Customer Support Triage API")
init_db()


class TicketIn(BaseModel):
    ticket_id: str
    customer_id: str
    subject: str
    body: str


class ApproveIn(BaseModel):
    edited_reply: str | None = None


class RejectIn(BaseModel):
    note: str | None = None


@app.post("/tickets")
def submit_ticket(payload: TicketIn):
    ticket = Ticket(**payload.model_dump())
    result = process_ticket(ticket)
    save_result(ticket, result)
    return get_ticket(ticket.ticket_id)


@app.get("/tickets")
def get_tickets(status: str | None = None):
    return list_tickets(status)


@app.get("/tickets/{ticket_id}")
def get_one_ticket(ticket_id: str):
    row = get_ticket(ticket_id)
    if not row:
        raise HTTPException(404, "ticket not found")
    return row


@app.post("/tickets/{ticket_id}/approve")
def approve(ticket_id: str, payload: ApproveIn):
    row = approve_ticket(ticket_id, payload.edited_reply)
    if not row:
        raise HTTPException(404, "ticket not found")
    return row


@app.post("/tickets/{ticket_id}/reject")
def reject(ticket_id: str, payload: RejectIn):
    row = reject_ticket(ticket_id, payload.note)
    if not row:
        raise HTTPException(404, "ticket not found")
    return row


_REVIEW_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Support Review Queue</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; display: flex; height: 100vh; }
  #list { width: 340px; border-right: 1px solid #ddd; overflow-y: auto; }
  #list div.item { padding: 12px; border-bottom: 1px solid #eee; cursor: pointer; }
  #list div.item:hover { background: #f5f5f5; }
  #detail { flex: 1; padding: 20px; overflow-y: auto; }
  textarea { width: 100%; min-height: 120px; font-family: inherit; font-size: 14px; }
  button { padding: 8px 16px; margin-right: 8px; cursor: pointer; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 12px; color: white; background: #1a73e8; }
  pre { white-space: pre-wrap; background: #f5f5f5; padding: 10px; border-radius: 4px; }
  .empty { color: #888; padding: 20px; }
</style>
</head>
<body>
  <div id="list"><div class="empty">Loading...</div></div>
  <div id="detail"><div class="empty">Select a ticket on the left</div></div>

<script>
let selected = null;

async function loadList() {
  const res = await fetch('/tickets?status=pending');
  const tickets = await res.json();
  const list = document.getElementById('list');
  if (tickets.length === 0) {
    list.innerHTML = '<div class="empty">No tickets pending review 🎉</div>';
    return;
  }
  list.innerHTML = tickets.map(t => `
    <div class="item" onclick="loadDetail('${t.ticket_id}')">
      <strong>${t.ticket_id}</strong> — ${t.subject}<br>
      <span class="badge">${t.category || '?'}</span>
      <small>confidence: ${(t.confidence ?? 0).toFixed(2)}</small>
    </div>
  `).join('');
}

async function loadDetail(ticketId) {
  selected = ticketId;
  const res = await fetch(`/tickets/${ticketId}`);
  const t = await res.json();
  const reasons = JSON.parse(t.reasons || '[]');
  const trace = JSON.parse(t.trace || '[]');
  document.getElementById('detail').innerHTML = `
    <h2>${t.ticket_id} — ${t.subject}</h2>
    <p><strong>Customer:</strong> ${t.customer_id}</p>
    <p><strong>Body:</strong></p><pre>${t.body}</pre>
    <p><strong>Category:</strong> ${t.category || '-'} &nbsp;
       <strong>Priority:</strong> P${t.priority || '-'} &nbsp;
       <strong>Sentiment:</strong> ${t.sentiment || '-'}</p>
    <p><strong>Confidence:</strong> ${(t.confidence ?? 0).toFixed(3)} &nbsp;
       <strong>Reasons:</strong> ${reasons.join(', ')}</p>
    <p><strong>Trace:</strong></p><pre>${trace.join(' -> ')}</pre>
    <p><strong>Draft reply (editable):</strong></p>
    <textarea id="replyBox">${t.draft_reply || ''}</textarea>
    <div style="margin-top:12px;">
      <button onclick="approve('${t.ticket_id}')">✅ Approve & Send</button>
      <button onclick="reject('${t.ticket_id}')">❌ Reject</button>
    </div>
  `;
}

async function approve(ticketId) {
  const reply = document.getElementById('replyBox').value;
  await fetch(`/tickets/${ticketId}/approve`, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({edited_reply: reply})
  });
  document.getElementById('detail').innerHTML = '<div class="empty">Select a ticket on the left</div>';
  loadList();
}

async function reject(ticketId) {
  const note = prompt('Reason for rejecting (optional):') || '';
  await fetch(`/tickets/${ticketId}/reject`, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({note})
  });
  document.getElementById('detail').innerHTML = '<div class="empty">Select a ticket on the left</div>';
  loadList();
}

loadList();
setInterval(loadList, 15000);
</script>
</body>
</html>
"""


@app.get("/review", response_class=HTMLResponse)
def review_page():
    return _REVIEW_HTML
