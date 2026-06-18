# tests/test_adapt_ai/test_providers.py
import pytest
from adapt_ai.llmops.providers import (
    CompletionResult, AnthropicProvider, OpenAICompatibleProvider, get_provider,
)


class _FakeAnthropicMessages:
    def create(self, **kw):
        class _U:
            input_tokens = 11
            output_tokens = 7
        class _Block:
            text = "anthropic-ok"
        class _Resp:
            content = [_Block()]
            usage = _U()
        self.last_kwargs = kw
        return _Resp()


class _FakeAnthropicClient:
    def __init__(self):
        self.messages = _FakeAnthropicMessages()


def test_anthropic_provider_maps_system_and_returns_tokens():
    prov = AnthropicProvider(client=_FakeAnthropicClient(), model="claude-haiku-4-5-20251001")
    res = prov.complete(system="sys", user="hi", max_tokens=128, temperature=0.3)
    assert isinstance(res, CompletionResult)
    assert res.text == "anthropic-ok"
    assert res.input_tokens == 11 and res.output_tokens == 7
    # Anthropic takes `system` as a top-level kwarg, not a message.
    assert prov._client.messages.last_kwargs["system"] == "sys"
    assert prov._client.messages.last_kwargs["messages"] == [{"role": "user", "content": "hi"}]


class _FakeOpenAICompletions:
    last_kwargs = {}

    @staticmethod
    def create(**kw):
        class _Msg:
            content = "ollama-ok"
        class _Choice:
            message = _Msg()
        class _U:
            prompt_tokens = 13
            completion_tokens = 5
        class _Resp:
            choices = [_Choice()]
            usage = _U()
        _FakeOpenAICompletions.last_kwargs = kw
        return _Resp()


class _FakeOpenAIChat:
    def __init__(self):
        self.completions = _FakeOpenAICompletions()


class _FakeOpenAIClient:
    def __init__(self):
        self.chat = _FakeOpenAIChat()


def test_openai_compatible_provider_folds_system_into_messages():
    prov = OpenAICompatibleProvider(client=_FakeOpenAIClient(), model="qwen2.5:7b-instruct")
    res = prov.complete(system="sys", user="hi", max_tokens=128, temperature=0.3)
    assert res.text == "ollama-ok"
    assert res.input_tokens == 13 and res.output_tokens == 5
    msgs = _FakeOpenAICompletions.last_kwargs["messages"]
    assert msgs[0] == {"role": "system", "content": "sys"}
    assert msgs[1] == {"role": "user", "content": "hi"}


def test_judge_refuses_same_model_as_sut():
    from evaluation.judge import build_judge
    with pytest.raises(ValueError, match="same model"):
        build_judge(provider="anthropic", model="claude-haiku-4-5-20251001",
                    sut_model="claude-haiku-4-5-20251001")


def test_judge_parses_numeric_score():
    from evaluation.judge import Judge
    from adapt_ai.llmops.providers import CompletionResult

    class _FakeJudgeProvider:
        model = "claude-opus-4-8"
        def complete(self, **kw):
            return CompletionResult("0.8", 1, 1, "claude-opus-4-8")

    j = Judge.from_provider(_FakeJudgeProvider(), sut_model="claude-haiku-4-5-20251001")
    score = j.score(prediction="some answer", reference="gold answer", query="the question")
    assert score == pytest.approx(0.8)
