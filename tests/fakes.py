from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class ScriptedLLM(BaseChatModel):
    """Returns the prepared AIMessages one by one, and records what it was sent."""
    script: list = Field(default_factory=list)
    seen: list = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])

    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def _llm_type(self):
        return "scripted"


def call(name, args, cid="c1"):
    return AIMessage(content="", tool_calls=[dict(name=name, args=args, id=cid)])