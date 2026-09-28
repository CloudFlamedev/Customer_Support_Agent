import os

# Must run before crewai/litellm do any work.
for _key in ("CREWAI_DISABLE_TELEMETRY", "CREWAI_DISABLE_TRACKING", "OTEL_SDK_DISABLED"):
    os.environ.setdefault(_key, "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

# --- Workaround for CrewAI bug #6789 -------------------------------------
# CrewAI auto-tags messages with an internal "cache_breakpoint" key for
# prompt-caching. Native providers strip it before sending; the generic
# LiteLLM path (used for Groq) does not, so Groq rejects the request with
# "property 'cache_breakpoint' is unsupported". Fix not yet merged upstream
# (https://github.com/crewAIInc/crewAI/pull/7176), so we strip it ourselves
# at the one place every model call passes through: litellm.completion().
import litellm  # noqa: E402

_original_completion = litellm.completion


def _completion_without_cache_breakpoint(*args, **kwargs):
    messages = kwargs.get("messages")
    if messages:
        for m in messages:
            if isinstance(m, dict):
                m.pop("cache_breakpoint", None)
    return _original_completion(*args, **kwargs)


litellm.completion = _completion_without_cache_breakpoint
# ---------------------------------------------------------------------------