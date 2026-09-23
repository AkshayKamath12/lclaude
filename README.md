# lclaude

Chat with local AI models in your terminal through Ollama. Keep your prompts on
your machine, without a cloud AI subscription.

Early stage: streaming chat with conversation history during the current session.
Coding tools and saved sessions are planned.

## Install and run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/),
[Git](https://git-scm.com/downloads), and [Ollama](https://ollama.com/download).
Requires Python 3.10+ (uv can install it automatically).

With Ollama running:

```sh
ollama pull qwen2.5:7b-instruct
uv tool install git+https://github.com/AkshayKamath12/lclaude.git
lclaude
```

Use another downloaded model with `lclaude --model <model-name>`.
