# Getting started from scratch

Build a working agent in about ten minutes: install the library into a fresh project,
connect a real LLM, and watch it call your tool.

At the end you will have a `my_agent.py` that asks Claude about the weather, and Claude
will decide on its own to call a Python function you wrote.

Commands are PowerShell (Windows). macOS/Linux differences are noted where they matter.

---

## Step 0 — Check your Python version

This library needs **Python 3.11 or newer**.

```powershell
python --version
```

If that prints 3.10 or lower, `pip install` will refuse with
`requires a different Python`. Install a newer Python from
[python.org/downloads](https://www.python.org/downloads/) first, or use the newer one you
already have — on Windows, `py -0` lists every installed version and
`py -3.13 -m venv .venv` picks one explicitly.

---

## Step 1 — Make a project folder

```powershell
mkdir weather-agent
```

```powershell
cd weather-agent
```

## Step 2 — Create and activate a virtual environment

A venv keeps this project's packages out of your system Python.

```powershell
python -m venv .venv
```

```powershell
.venv\Scripts\Activate.ps1
```

Your prompt now starts with `(.venv)`. That is how you know it worked.

- **macOS/Linux:** `source .venv/bin/activate`
- **PowerShell blocks the script?** Run
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and try again.
- **Skip activation entirely** by calling `.venv\Scripts\python.exe` instead of `python`
  in every later step.

## Step 3 — Install the library and a model client

```powershell
pip install agent-harnessed anthropic
```

Two separate installs, on purpose. `agent-harnessed` is the harness and depends only on
`pydantic`. `anthropic` is the model client — the harness deliberately ships no model
client, so you choose one.

Verify:

```powershell
python -c "import agent_harness; print(agent_harness.__version__)"
```

## Step 4 — Get an API key

1. Go to [console.anthropic.com](https://console.anthropic.com/) and sign in
2. **Settings → API keys → Create key**
3. Copy it now — it is shown only once

Set it for this terminal session:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-paste-yours-here"
```

That lasts until you close the window. To keep it permanently:

```powershell
setx ANTHROPIC_API_KEY "sk-ant-paste-yours-here"
```

`setx` only affects **new** terminals, so reopen your terminal afterwards.

- **macOS/Linux:** `export ANTHROPIC_API_KEY="sk-ant-..."`
- Never paste the key into your source file. Anything committed to git is public the
  moment you push.

## Step 5 — Write the agent

Create `my_agent.py` in the `weather-agent` folder:

```python
import asyncio

from anthropic import AsyncAnthropic

from agent_harness import (
    ContextEngine,
    GuardrailPipeline,
    InMemoryBackend,
    LLMTurnStep,
    LoopController,
    LoopLimits,
    MemoryStore,
    PromptCompiler,
    Tool,
    ToolGateway,
    TurnStepType,
)


# ---- 1. A tool the model is allowed to propose -----------------------------
async def get_weather(session_id: str, arguments: dict) -> dict:
    city = arguments["city"]
    print(f"   [tool ran] get_weather(city={city!r})")
    return {"city": city, "temp_c": 21, "sky": "clear"}


weather = Tool(
    name="get_weather",
    description="Get the current weather for a city.",
    parameters=["city"],
    execute=get_weather,
    read_only=True,
    max_calls_per_session=3,
)


# ---- 2. The adapter: CompiledPrompt -> Claude -> LLMTurnStep ---------------
def make_call_llm(client: AsyncAnthropic, tools: list[Tool]):
    schemas = [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": {
                "type": "object",
                "properties": {name: {"type": "string"} for name in tool.parameters},
                "required": list(tool.parameters),
                "additionalProperties": False,
            },
        }
        for tool in tools
    ]

    async def call_llm(prompt) -> LLMTurnStep:
        response = await client.messages.create(
            model="claude-opus-5",
            max_tokens=16000,
            messages=[{"role": "user", "content": prompt.render()}],
            tools=schemas,
        )
        for block in response.content:
            if block.type == "tool_use":
                return LLMTurnStep(
                    step_type=TurnStepType.TOOL_CALL,
                    tool_name=block.name,
                    tool_arguments=dict(block.input),
                )
        text = "".join(b.text for b in response.content if b.type == "text")
        return LLMTurnStep(step_type=TurnStepType.FINAL, text=text.strip())

    return call_llm


# ---- 3. Wire the harness together -----------------------------------------
async def main() -> None:
    memory = MemoryStore(InMemoryBackend())
    controller = LoopController(
        context_engine=ContextEngine(memory),
        prompt_compiler=PromptCompiler(role="You are a concise weather assistant."),
        tool_gateway=ToolGateway([weather]),
        guardrails=GuardrailPipeline(),
        memory=memory,
        call_llm=make_call_llm(AsyncAnthropic(), [weather]),
    )

    result = await controller.handle_turn(
        "session-1",
        "What's the weather in Paris?",
        limits=LoopLimits(max_turns=4, max_tool_calls=2),
    )

    print("stopped_reason:", result.stopped_reason)
    print("answer        :", result.final_text)
    print("turns_used    :", result.turns_used)
    for tool_result in result.tool_results:
        print("tool_result   :", tool_result.tool_name, "->", tool_result.data)


if __name__ == "__main__":
    asyncio.run(main())
```

`AsyncAnthropic()` takes no arguments — it reads `ANTHROPIC_API_KEY` from the environment
by itself.

## Step 6 — Run it

```powershell
python my_agent.py
```

Expect something close to:

```
   [tool ran] get_weather(city='Paris')
stopped_reason: final_answer
answer        : It's 21°C and clear in Paris.
turns_used    : 2
tool_result   : get_weather -> {'city': 'Paris', 'temp_c': 21, 'sky': 'clear'}
```

The exact wording of `answer` will differ every run. `[tool ran]` proves the model chose
to call your Python function.

---

## What actually happened

One `handle_turn` call, two trips to the model:

| | |
| --- | --- |
| 1 | `ContextEngine` builds a `TaskFrame` and pulls memory into a `ScopedContext` |
| 2 | `PromptCompiler` compiles that into a versioned `CompiledPrompt` |
| 3 | Your adapter sends it to Claude, which replies with a `tool_use` block |
| 4 | The adapter maps it to `TurnStepType.TOOL_CALL` |
| 5 | `ToolGateway` checks the allowlist and rate limit, **then** runs `get_weather` |
| 6 | The result is written to memory and folded into the prompt |
| 7 | Claude sees the evidence and replies with text → `TurnStepType.FINAL` |
| 8 | The loop stops and returns a `LoopResult` |

**The model never touched your function.** It proposed a call; the gateway decided. That
gap is the entire point of the library — it is where allowlisting, authorization, rate
limiting, and redaction live.

## Things to try next

Each of these takes one edit and shows a different guarantee.

**Watch the budget stop the loop.** Set `max_tool_calls=0` in `LoopLimits`. You get
`stopped_reason: max_tool_calls_exceeded` and `final_text: None` — no answer, because
there was no evidence to ground one.

**Watch a tool failure become evidence instead of a crash.** Add
`raise RuntimeError("weather service down")` as the first line of `get_weather`. The
process does not crash; the failure comes back as a `ToolResult` with `ok=False`, and the
model gets told about it and adapts.

**Watch redaction.** Add `redact_fields=["temp_c"]` to the `Tool` and rerun. The printed
`tool_result` shows `'temp_c': '[REDACTED]'`. The raw value never reaches memory or the
next prompt — it is scrubbed inside the gateway, before anything can observe it.

**Add a second tool.** Write another `async def`, wrap it in a `Tool`, and pass both to
`ToolGateway([weather, other])` *and* `make_call_llm(client, [weather, other])`. Both
lists must match — the gateway enforces policy, the adapter tells the model what exists.

## Using a different model

Nothing here is Claude-specific except the SDK call inside `call_llm`. To swap providers,
rewrite that one function so it returns an `LLMTurnStep`, and leave everything else alone.
Three mappings are all that any adapter needs:

- the model wants to use a tool → `TurnStepType.TOOL_CALL` with `tool_name` and `tool_arguments`
- the model needs to ask the user something → `TurnStepType.CLARIFICATION` with `text`
- the model is answering → `TurnStepType.FINAL` with `text`

For a model without native tool calling, have the adapter ask for JSON and parse it.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `requires a different Python: 3.10 not in '>=3.11'` | Your Python is too old. See Step 0. |
| `ModuleNotFoundError: No module named 'agent_harness'` | The venv is not active, or you installed into a different one. Re-run Step 2, then Step 3. Note the install name has a **d** (`agent-harnessed`); the import name does not (`agent_harness`). |
| `AuthenticationError` / `401` | Key missing, mistyped, or set in a different terminal. Check with `echo $env:ANTHROPIC_API_KEY`. If you used `setx`, open a new terminal. |
| `RateLimitError` / `429` | You are over your account's limit. Wait and retry; the SDK already retries twice on its own. |
| `credit balance is too low` | Add credits under **Settings → Billing** in the Anthropic console. |
| Runs, but `[tool ran]` never prints | The model answered from its own knowledge. Ask something it cannot know: `"What's the weather in Paris right now?"`, or sharpen the tool `description`. |
| `stopped_reason: max_turns_exceeded` | The loop hit its ceiling. Raise `max_turns` in `LoopLimits`, or check whether your adapter is returning `TOOL_CALL` forever without ever reaching `FINAL`. |
| `stopped_reason: repeated_call_detected` | The model proposed the identical call twice, so the loop cut it off. Usually the tool result was unhelpful — check what `get_weather` actually returned. |

## Where to go next

- [`examples/refund_agent.py`](examples/refund_agent.py) — the production-shaped
  version of everything here: a real database, an LLM intent classifier, three
  independent protections on a side effect that costs money, and `--trace` to see
  which stage of the pipeline stopped a turn
- [README](README.md) — module-by-module architecture, the three loop exits, and the
  guardrail payload contract
