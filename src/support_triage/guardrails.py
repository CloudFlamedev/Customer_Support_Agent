"""Guardrails: input safety checks (PII masking + injection detection) and
output safety checks (policy compliance).

These run as plain code + a single LLM call each, NOT as CrewAI crews. Same
rule as triage_crew.py and research_writer_crew.py: the LLM supplies a
signal, Python decides pass/fail. Any internal error fails CLOSED
(passed=False), so a broken check routes a ticket to a human rather than
silently letting it through.
"""
import json
import re

from crewai import LLM
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

from support_triage.config import settings
from support_triage.schemas import GuardrailResult

# ---------------------------------------------------------------------------
# PII masking — deterministic, free, always runs
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")               # check before phone (longer digit runs)
_PHONE_RE = re.compile(r"(?<!\d)(\+?\d[\d\-\s]{7,}\d)(?!\d)")


def mask_pii(text: str) -> tuple[str, list[str]]:
    """Replace emails/card numbers/phone numbers with placeholders.
    Returns (masked_text, list of PII types found). Heuristic regexes —
    good enough for a demo KB, would need tightening for production."""
    found: list[str] = []

    def _sub(pattern: re.Pattern, placeholder: str, label: str, s: str) -> str:
        if pattern.search(s):
            found.append(label)
        return pattern.sub(placeholder, s)

    masked = text
    masked = _sub(_EMAIL_RE, "[EMAIL]", "email", masked)
    masked = _sub(_CARD_RE, "[CARD]", "card_number", masked)
    masked = _sub(_PHONE_RE, "[PHONE]", "phone", masked)
    return masked, found


# ---------------------------------------------------------------------------
# Input guardrail: prompt-injection detection
# ---------------------------------------------------------------------------

# Primary, guaranteed signal: deterministic phrase matching. Zero cost,
# zero rate limits, cannot silently fail.
_INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) instructions",
    r"disregard (all |any )?(previous|prior|above)",
    r"you are now (a|an)",
    r"system prompt",
    r"new instructions?:",
    r"developer mode",
    r"jailbreak",
    r"reveal your (instructions|prompt|rules)",
    r"forget (your|all) (rules|instructions)",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def _keyword_injection_check(text: str) -> bool:
    return bool(_INJECTION_RE.search(text))


# Secondary, best-effort signal: Meta's purpose-built classifier via Groq.
# We don't have a confirmed spec for its exact Groq response format, so any
# unrecognized output is silently treated as "no extra signal" rather than
# breaking the pipeline. This can only ADD flags, never remove the ones above.
def _prompt_guard_check(text: str) -> bool | None:
    """Returns True if flagged, False if clear, None if unavailable/unclear."""
    try:
        llm = LLM(model=settings.guardrail_input_model, api_key=settings.groq_api_key, max_tokens=10)
        raw = (llm.call(text[:2000]) or "").strip().upper()
        if "JAILBREAK" in raw or "INJECTION" in raw:
            return True
        if "BENIGN" in raw:
            return False
        return None
    except Exception:
        return None


def check_input(ticket_body: str) -> GuardrailResult:
    """PII is masked and reported but never fails the check on its own —
    a customer's own email in their ticket is normal. Only injection signals
    fail it."""
    try:
        masked_text, pii_found = mask_pii(ticket_body)
        reasons: list[str] = []

        if _keyword_injection_check(ticket_body):
            reasons.append("injection_keyword_match")
        if _prompt_guard_check(ticket_body) is True:
            reasons.append("injection_model_flag")

        return GuardrailResult(
            passed=len(reasons) == 0,
            reasons=reasons,
            pii_types=pii_found,
            masked_text=masked_text,
        )
    except Exception as e:
        return GuardrailResult(passed=False, reasons=[f"guardrail_error:{e}"])


# ---------------------------------------------------------------------------
# Output guardrail: policy check on the drafted reply
# ---------------------------------------------------------------------------

_OUTPUT_GUARDRAIL_PROMPT = """You are a support-reply safety reviewer.

<ticket_subject>{subject}</ticket_subject>
<draft_reply>{reply}</draft_reply>

Check the draft reply for:
1. A specific refund amount, compensation, or legal promise that goes beyond
   a generic policy statement.
2. Any email, phone number, or card number left in the reply.
3. Unprofessional, rude, or unclear language.

Respond with ONLY one JSON object: {{"passed": true|false, "reasons": ["..."]}}
passed is false if ANY issue above is present. reasons lists which issues
were found (empty list if passed is true). No other text.
"""


def _is_retryable(e: BaseException) -> bool:
    if isinstance(e, ValueError):
        return True
    msg = str(e).lower()
    return "rate limit" in msg or "ratelimit" in msg or "429" in msg


def _extract_json(raw: str) -> str:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON in model output: {raw[:200]!r}")
    return match.group(0)


@retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(min=5, max=30),
    reraise=True,
)
def _check_output_once(subject: str, reply: str) -> GuardrailResult:
    llm = LLM(
        model=settings.guardrail_output_model,
        api_key=settings.groq_api_key,
        temperature=0,
        max_tokens=300,
        reasoning_effort="low",
    )
    raw = llm.call(_OUTPUT_GUARDRAIL_PROMPT.format(subject=subject, reply=reply))
    data = json.loads(_extract_json(raw))
    return GuardrailResult(passed=bool(data.get("passed", False)), reasons=list(data.get("reasons", [])))


def check_output(subject: str, reply: str) -> GuardrailResult:
    """Hard regex PII check runs first and fails closed immediately if it
    finds anything (no LLM call needed to catch an obvious leak). Otherwise
    the policy model reviews the reply. Any failure -> passed=False, so a
    broken guardrail routes to a human instead of letting an unreviewed
    reply go out."""
    _, pii_found = mask_pii(reply)
    if pii_found:
        return GuardrailResult(passed=False, reasons=[f"pii_in_reply:{','.join(pii_found)}"], pii_types=pii_found)
    try:
        return _check_output_once(subject, reply)
    except Exception as e:
        return GuardrailResult(passed=False, reasons=[f"guardrail_error:{e}"])
