"""Triage crew: reads a ticket, returns a validated TriageResult.

We do NOT use output_pydantic: it forces tool-calling, which gpt-oss on
Groq handles badly. Instead the model returns plain JSON and Pydantic
validates it in our own code.
"""
import re

from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

from support_triage.config import settings
from support_triage.schemas import Ticket, TriageResult


class TriageError(Exception):
    """Triage failed after retries. The Flow should send this ticket to a human."""


@CrewBase
class TriageCrew:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def triage_agent(self) -> Agent:
        llm = LLM(
            model=settings.triage_model,
            api_key=settings.groq_api_key,   # pydantic-settings does not fill os.environ
            temperature=0,
            max_tokens=1500,                 # covers thinking tokens + the JSON answer
            reasoning_effort="low",          # gpt-oss: think less, spend fewer tokens
        )
        return Agent(
            config=self.agents_config["triage_agent"],
            llm=llm,
            allow_delegation=False,
            verbose=False,
        )

    @task
    def triage_task(self) -> Task:
        return Task(config=self.tasks_config["triage_task"])

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            memory=False,
            verbose=False,
            tracing=False,
        )


def _extract_json(raw: str) -> str:
    """Models sometimes wrap JSON in text or code fences. Grab the {...} part."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON in model output: {raw[:200]!r}")
    return match.group(0)


def _is_retryable(e: BaseException) -> bool:
    # ValueError covers bad JSON and Pydantic ValidationError (both subclass it).
    if isinstance(e, ValueError):
        return True
    msg = str(e).lower()
    return "rate limit" in msg or "ratelimit" in msg or "429" in msg


# Retry only temporary problems. Setup errors (bad model, missing package) fail at once.
@retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(4),
    wait=wait_random_exponential(min=8, max=40),
    reraise=True,
)
def _run_once(ticket: Ticket) -> TriageResult:
    result = TriageCrew().crew().kickoff(
        inputs={
            "ticket_id": ticket.ticket_id,
            "subject": ticket.subject,
            "body": ticket.body[:4000],
        }
    )
    return TriageResult.model_validate_json(_extract_json(result.raw))


def run_triage(ticket: Ticket) -> TriageResult:
    try:
        return _run_once(ticket)
    except Exception as e:
        raise TriageError(f"Triage failed for {ticket.ticket_id}: {e}") from e