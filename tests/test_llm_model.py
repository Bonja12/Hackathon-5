from types import SimpleNamespace

from provile import llm_model


class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, model, messages):
        self.calls.append(messages)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])


def make_llm():
    llm = llm_model.LLM(base_url="http://x", api_key="k", llm_model="m")
    fake = FakeCompletions()
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    return llm, fake


def test_invoke_without_system_prompt_sends_only_user_message():
    llm, fake = make_llm()
    assert llm.invoke("hi") == "ok"
    assert fake.calls[0] == [{"role": "user", "content": "hi"}]


def test_invoke_with_system_prompt_sends_system_first():
    llm, fake = make_llm()
    llm.invoke("hi", system_prompt="be nice")
    assert fake.calls[0] == [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "hi"},
    ]


def test_failed_calls_use_configured_tries_and_return_error(monkeypatch):
    monkeypatch.setenv("LLM_MAX_TRIES", "3")
    monkeypatch.setenv("LLM_RETRY_WAIT", "0")
    llm, _ = make_llm()
    attempts = []

    def boom(model, messages):
        attempts.append(1)
        raise RuntimeError("429")

    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=boom)))
    assert llm.invoke("hi") == llm_model.ERROR_RESPONSE
    assert len(attempts) == 3
