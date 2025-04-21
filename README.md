# ArcGIS Pro AI Toolbox

<p align="center">
  <img src="docs/logo.png" alt="ArcGIS Pro AI Toolbox logo" height="80"/>
</p>

<p align="center">
  <img src="https://img.shields.io/pypi/dw/arcgispro_ai" alt="PyPI - Downloads">
</p>

<p align="center">
  <b>Your GIS, now AI‑supercharged.</b>
</p>

---

## Quick Links
- [GitHub](https://github.com/danmaps/arcgispro-ai-toolbox)
- [Docs](https://danmaps.github.io/arcgispro_ai/)

---

## Why ArcGIS Pro AI Toolbox?

<blockquote>
ArcGIS Pro AI Toolbox is the only BYOK, open‑source plugin that brings conversational AI, code generation, and prompt‑driven geoprocessing natively into ArcGIS Pro—no cloud hop, no proprietary credits.
</blockquote>

- Install in minutes. Prompt, generate, map—directly inside ArcGIS Pro.
- Works with OpenAI, Azure, Claude, DeepSeek, local LLMs, and more.
- BYOK: Bring your own API key, keep your data private.
- No cloud detour, no extra Esri credits, no code required.

---

## Key Tools

- <b>Add AI Generated Field</b>: Create rich text from attributes using AI.
- <b>Get Map Info</b>: Extract map context to JSON for smarter prompts.
- <b>Generate Python Code</b>: ArcPy snippets tuned to your map.
- <b>Create AI Feature Layer</b>: Describe data, get a layer.
- <b>Convert Text to Numeric</b>: Standardize messy columns fast.

---

## Installation

`pip install arcgispro_ai`

Set up the required environment variables for your chosen AI provider(s):

   ```batch
   setx OPENAI_API_KEY "your-key-here"
   setx AZURE_OPENAI_API_KEY "your-key-here"
   setx ANTHROPIC_API_KEY "your-key-here"
   setx DEEPSEEK_API_KEY "your-key-here"
   ```

### For local LLM setup

   - Deploy a compatible LLM server that implements the OpenAI chat completions API
   - Configure the endpoint URL in the tool interface (defaults to `http://localhost:8000`)

## Usage

1. Select your preferred AI provider from the dropdown in each tool
2. Configure any provider-specific settings (model, endpoint, etc.)
3. Enter your prompt or query
4. Execute the tool

Each tool will use the selected provider to generate responses, with automatic fallback to OpenAI if the selected provider is not configured.

## Supported AI Providers

- <b>OpenAI</b>: GPT-4 and more (requires `OPENAI_API_KEY`)
- <b>Azure OpenAI</b>: Microsoft-hosted (requires `AZURE_OPENAI_API_KEY`)
- <b>Claude (Anthropic)</b>: (requires `ANTHROPIC_API_KEY`)
- <b>DeepSeek</b>: (requires `DEEPSEEK_API_KEY`)
- <b>Local LLM</b>: No API key needed, OpenAI-compatible API
- <b>Wolfram Alpha</b>: For math/computation (requires `WOLFRAM_ALPHA_API_KEY`)

## Contributing

Make an issue or create a branch for your feature or bug fix, and submit a pull request.

---

