import time

from support_triage.crews import run_writer
from support_triage.schemas import Ticket

tests = [
    ("W1", "Refund timing", "I sent my item back last week. How long until I get my money?"),
    ("W2", "Gift card refund", "Can I get a refund on an unused gift card I bought?"),
    ("W3", "Off topic", "Do you sell pizza and can I order one through this chat?"),
]

for tid, subj, body in tests:
    ticket = Ticket(ticket_id=tid, customer_id="C1", subject=subj, body=body)
    r = run_writer(ticket)
    print(f"\n{tid} — self_score={r.self_score}  sources={r.sources}")
    print(f"  reply: {r.reply}")
    time.sleep(15)  # stay under the 8,000 TPM budget on gpt-oss-120b