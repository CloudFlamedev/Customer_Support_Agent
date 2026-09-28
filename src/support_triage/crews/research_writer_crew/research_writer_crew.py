"""Research + Writer crew.

Research agent searches the KB with a tool. Writer agent sees only the
research output and the ticket, and must answer from that alone (this is
what makes it RAG rather than the model answering from memory).

Like triage_crew.py, we avoid output_pydantic (forces tool-calling, which
breaks on Groq's gpt-oss). The writer returns plain JSON that we validate
ourselves.
"""
import re

from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

from support_triage.config import settings
from support_triage.schemas import DraftReply, Ticket
from support_triage.tools.kb_search import KBSearchTool


class WriterError(Exception):
    """Research/writing failed after retries. The Flow should send this ticket to a human."""


@CrewBase
class ResearchWriterCrew:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def research_agent(self) -> Agent:
        llm = LLM(
            model=settings.writer_model,
            api_key=settings.groq_api_key,
            temperature=0,
            max_tokens=800,
            reasoning_effort="low",
        )
        return Agent(
            config=self.agents_config["research_agent"],
            llm=llm,
            tools=[KBSearchTool()],
            allow_delegation=False,
            verbose=False,
        )

    @agent
    def writer_agent(self) -> Agent:
        llm = LLM(
            model=settings.writer_model,
            api_key=settings.groq_api_key,
            temperature=0.2,
            max_tokens=900,
            reasoning_effort="low",
        )
        return Agent(
            config=self.agents_config["writer_agent"],
            llm=llm,
            allow_delegation=False,
            verbose=False,
        )

    @task
    def research_task(self) -> Task:
        return Task(config=self.tasks_config["research_task"])

    @task
    def writer_task(self) -> Task:
        return Task(config=self.tasks_config["writer_task"])

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
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON in model output: {raw[:200]!r}")
    return match.group(0)


def _is_retryable(e: BaseException) -> bool:
    if isinstance(e, ValueError):
        return True
    msg = str(e).lower()
    return "rate limit" in msg or "ratelimit" in msg or "429" in msg


@retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(min=8, max=40),
    reraise=True,
)
def _run_once(ticket: Ticket) -> DraftReply:
    result = ResearchWriterCrew().crew().kickoff(
        inputs={
            "subject": ticket.subject,
            "body": ticket.body[:3000],
        }
    )
    return DraftReply.model_validate_json(_extract_json(result.raw))


def run_writer(ticket: Ticket) -> DraftReply:
    try:
        return _run_once(ticket)
    except Exception as e:
        raise WriterError(f"Writer failed for {ticket.ticket_id}: {e}") from e