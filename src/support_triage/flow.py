"""Orchestration: ties guardrails, triage, and research/writer together into
one decision per ticket.

Each individual step (classification, retrieval, drafting) is a CrewAI Crew
-- that's where the agentic reasoning happens. The orchestration and the
final auto_send/human_review decision are plain Python, so the decision is
deterministic, testable, and never left to an LLM's self-assessment. Every
failure path below defaults to human_review, never to guessing.
"""
from dataclasses import dataclass, field

from support_triage.config import settings
from support_triage.crews import TriageError, WriterError, run_triage, run_writer
from support_triage.guardrails import check_input, check_output
from support_triage.schemas import Decision, DraftReply, GuardrailResult, Ticket, TriageResult
from support_triage.tools.kb_search import search_kb


@dataclass
class FlowResult:
    ticket_id: str
    decision: Decision
    triage: TriageResult | None = None
    draft: DraftReply | None = None
    input_guardrail: GuardrailResult | None = None
    output_guardrail: GuardrailResult | None = None
    retrieval_score: float = 0.0
    trace: list[str] = field(default_factory=list)


def _human(ticket_id: str, reason: str, trace: list[str], **extra) -> FlowResult:
    trace.append(reason)
    return FlowResult(
        ticket_id=ticket_id,
        decision=Decision(action="human_review", confidence=0.0, reasons=[reason]),
        trace=trace,
        **extra,
    )


def _retrieval_confidence(raw_score: float) -> float:
    """Maps a raw cosine score to 0..1 using the floor/good bounds
    calibrated in Step 2 against real KB queries (config.py)."""
    floor, good = settings.retrieval_floor, settings.retrieval_good
    if raw_score <= floor:
        return 0.0
    if raw_score >= good:
        return 1.0
    return (raw_score - floor) / (good - floor)


def process_ticket(ticket: Ticket) -> FlowResult:
    trace: list[str] = []

    # 1. Input guardrail — nothing proceeds unchecked
    input_result = check_input(ticket.body)
    trace.append(f"input_guardrail:{'pass' if input_result.passed else 'fail'}")
    if not input_result.passed:
        return _human(ticket.ticket_id, f"input_guardrail_failed:{','.join(input_result.reasons)}",
                      trace, input_guardrail=input_result)

    masked_ticket = ticket.model_copy(update={"body": input_result.masked_text or ticket.body})

    # 2. Triage
    try:
        triage = run_triage(masked_ticket)
    except TriageError as e:
        return _human(ticket.ticket_id, f"triage_failed:{e}", trace, input_guardrail=input_result)
    trace.append(f"triage:{triage.category.value}/P{triage.priority}/{triage.sentiment}")

    # 3. High-risk categories go straight to a human -- no draft is even attempted
    if triage.category.value in settings.always_escalate_categories:
        return _human(ticket.ticket_id, f"high_risk_category:{triage.category.value}", trace,
                      input_guardrail=input_result, triage=triage)

    # 4. Research + Writer
    try:
        draft = run_writer(masked_ticket)
    except WriterError as e:
        return _human(ticket.ticket_id, f"writer_failed:{e}", trace,
                      input_guardrail=input_result, triage=triage)
    trace.append(f"draft:self_score={draft.self_score}")

    # 5. Output guardrail
    output_result = check_output(masked_ticket.subject, draft.reply)
    trace.append(f"output_guardrail:{'pass' if output_result.passed else 'fail'}")
    if not output_result.passed:
        return _human(ticket.ticket_id, f"output_guardrail_failed:{','.join(output_result.reasons)}", trace,
                      input_guardrail=input_result, triage=triage, draft=draft, output_guardrail=output_result)

    # 6. Retrieval score computed directly in code, NOT trusted from the
    #    research agent's own tool call -- same rule as every other decision.
    hits = search_kb(f"{masked_ticket.subject} {masked_ticket.body}", k=1)
    raw_retrieval_score = hits[0].score if hits else 0.0
    retrieval_confidence = _retrieval_confidence(raw_retrieval_score)
    trace.append(f"retrieval_score={raw_retrieval_score:.2f}")

    # 7. Combine into one confidence number
    confidence = round(0.5 * draft.self_score + 0.5 * retrieval_confidence, 3)
    trace.append(f"confidence={confidence}")

    # 8. Final gate
    action = "auto_send" if confidence >= settings.confidence_threshold else "human_review"
    decision = Decision(action=action, confidence=confidence,
                         reasons=[f"confidence={confidence}", f"threshold={settings.confidence_threshold}"])
    trace.append(f"decision:{action}")

    return FlowResult(
        ticket_id=ticket.ticket_id,
        decision=decision,
        triage=triage,
        draft=draft,
        input_guardrail=input_result,
        output_guardrail=output_result,
        retrieval_score=raw_retrieval_score,
        trace=trace,
    )
