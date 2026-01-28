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
- [Agent Instructions](AGENT_INSTRUCTIONS.md)

---

## Why ArcGIS Pro AI Toolbox?

<blockquote>
ArcGIS Pro AI Toolbox is the only BYOK, open‑source plugin that brings conversational AI, code generation, and prompt‑driven geoprocessing natively into ArcGIS Pro—no cloud hop, no proprietary credits.
</blockquote>

- Install in minutes. Prompt, generate, map—directly inside ArcGIS Pro.
- Preconfigured with OpenRouter out of the box, while still working with OpenAI, Azure, Claude, DeepSeek, local LLMs, and more.
- BYOK: Bring your own API key, keep your data private.
- No cloud detour, no extra Esri credits, no code required.

---

## Key Tools

- <b>Add AI Generated Field</b>: Create rich text from attributes using AI.
- <b>Get Map Info</b>: Extract map context to JSON for smarter prompts.
- <b>Generate Python Code</b>: ArcPy snippets tuned to your map.
- <b>Interpret Map</b>: Summarize the active view with an AI-driven interpretation that combines a screenshot and spatial context.
- <b>Create AI Feature Layer</b>: Describe data, get a layer.
- <b>Convert Text to Numeric</b>: Standardize messy columns fast.

---

## Installation

There are two ways to get started with the ArcGIS Pro AI Toolbox:

1. **The Simple Way (Recommended):**
   - Download the toolbox directly from [the arcgispro_ai website](https://danmaps.github.io/arcgispro_ai).
   - Set up the required environment variables for your chosen AI provider(s)
   - Add the downloaded `.pyt` file to ArcGIS Pro and start using the tools immediately.

2. **The Python Way (For Advanced Users):**
   - Install the package via pip:
     ```bash
     pip install arcgispro_ai
     ```
   - Set up the required environment variables for your chosen AI provider(s)
   - Use the tools programmatically or within ArcGIS Pro by referencing the installed package. This requires an import of the toolbox from a path like
   ```bash
      `C:\Users\<username>\AppData\Local\Programs\Python\Python<version>\Lib\site-packages\arcgispro_ai\toolboxes
   ```

## Environment Variables

Set up the required environment variables for your chosen AI provider(s):

   ```batch
   setx OPENROUTER_API_KEY "your-key-here"
   setx OPENAI_API_KEY "your-key-here"
   setx AZURE_OPENAI_API_KEY "your-key-here"
   setx ANTHROPIC_API_KEY "your-key-here"
   setx DEEPSEEK_API_KEY "your-key-here"
   ```

OpenRouter is the default provider for every tool, so configuring `OPENROUTER_API_KEY` is enough to start running prompts immediately. Set the other environment variables only if you plan to switch to those providers.

## For local LLM setup

- Deploy a compatible LLM server that implements the OpenAI chat completions API. That's up to you to figure out. idk, ask ChatGPT.
- Make sure to configure the endpoint URL to `http://localhost:8000` or you'll have to override it every time you want to run a tool.

---

## Usage

1. Leave the Source dropdown at OpenRouter (the default) or switch to another provider if you've configured its API key
2. Configure any provider-specific settings (model, endpoint, etc.)
3. Enter your prompt or query
4. Execute the tool

Each tool starts on OpenRouter, so you can run with zero extra configuration. When you choose another provider, make sure its API key and settings are in place before executing the tool.

## Supported AI Providers

- <b>OpenRouter (default)</b>: Unified API for multiple models including OpenAI, Gemini, Claude, Llama, and more (requires `OPENROUTER_API_KEY`)
- <b>OpenAI</b>: GPT-4 and more (requires `OPENAI_API_KEY`)
- <b>Azure OpenAI</b>: Microsoft-hosted (requires `AZURE_OPENAI_API_KEY`)
- <b>Claude (Anthropic)</b>: (requires `ANTHROPIC_API_KEY`)
- <b>DeepSeek</b>: (requires `DEEPSEEK_API_KEY`)
- <b>Local LLM</b>: No API key needed, OpenAI-compatible API
- <b>Wolfram Alpha</b>: For math/computation (requires `WOLFRAM_ALPHA_API_KEY`)

### OpenRouter Details

OpenRouter provides a single API that gives you access to dozens of AI models from various providers:
- OpenAI models (GPT-4, GPT-3.5, etc.)
- Google models (Gemini 2.0 Flash, etc.)
- Anthropic models (Claude 3.5 Sonnet, etc.)
- Meta models (Llama variants)
- DeepSeek models
- And many more

To use OpenRouter:
1. Sign up for an API key at [openrouter.ai](https://openrouter.ai)
2. Set your API key:
   ```batch
   setx OPENROUTER_API_KEY "your-openrouter-key"
   ```
3. OpenRouter is already selected in every tool—leave the Source dropdown on OpenRouter (or switch back to it) to start using it immediately
4. The Model dropdown shows a curated set of OpenRouter options; if you need something else, you can type any OpenRouter model ID manually. Refer to the full catalog at https://openrouter.ai/models?fmt=table

---

## Project Structure & Distribution Design

This project is organized for both maintainability and ease of distribution:

- **Modular Source Structure:**
  - The codebase is organized into multiple Python modules and packages (see `arcgispro_ai/` and `arcgispro_ai/toolboxes/`).
  - This modular design makes the code easy to maintain, test, and extend.
  - Utility functions, API clients, and tool logic are separated for clarity and reusability.

- **Monolithic `.pyt` for Distribution:**
  - For end users, a single-file, monolithic Python Toolbox (`.pyt`) is generated (`arcgispro_ai.pyt`).
  - This file contains all required code inlined—no dependencies on the rest of the repo structure.
  - Users can simply download the `.pyt` and add it to ArcGIS Pro, with no need to install Python packages or clone the repo.
  - The monolithic `.pyt` is auto-generated by the `build_monolithic_pyt.py` script, which inlines all code and strips out internal imports.
  - The version of the `.pyt` always matches the package version (from `setup.py`), ensuring consistency with PyPI releases.

- **Release Management:**
  - The `release.sh` script automates the version management and release process.
  - It automatically increments the patch version in `setup.py`, commits the change, and tags the release.
  - The script creates a Git tag with the new version number and pushes changes to the repository.
  - After pushing, it attempts to open the GitHub release page to facilitate release note creation.
  - This ensures consistent versioning between the codebase, PyPI releases, and GitHub tags.

**Summary:**

- Developers benefit from a clean, modular codebase.
- Users benefit from a simple, single-file download for ArcGIS Pro.
- Releases are managed systematically with automated versioning.

See `build_monolithic_pyt.py` for details on how the monolithic `.pyt` is built.

---

## Project Layout

The codebase is organized to keep Esri toolbox glue minimal while centralizing reusable logic:

```text
arcgispro_ai/
├─ toolboxes/
│  ├─ arcgispro_ai_tools.pyt        # Esri Python Toolbox (UI, params, execute orchestration)
│  └─ arcgispro_ai/
│     └─ arcgispro_ai_utils.py      # Shared utilities and helpers used by tools
├─ core/
│  └─ api_clients.py                # Provider abstraction + client factory (`get_client`)
├─ docs/                            # Static docs site content
├─ build_monolithic_pyt.py          # Builds single-file .pyt for distribution
├─ arcgispro_ai.pyt                 # Generated monolithic toolbox (checked in for convenience)
└─ README.md
```

### Separation of Concerns (Toolbox vs. Logic)

- Toolbox file focuses on: parameter definitions, validation messages, and high-level `execute()` orchestration.
- Implementation details live in utilities and core modules to keep the `.pyt` readable and maintainable.
- Examples of logic moved out of the toolbox into utilities:
  - Map context capture and screenshot handling
  - Interpretation instructions (system prompt) for the AI
  - Markdown → HTML rendering for rich, theme-friendly output
  - Feature counting, API key resolution, model parameter wiring, and vision capability checks

Key modules:

- Utilities: `arcgispro_ai/toolboxes/arcgispro_ai/arcgispro_ai_utils.py`
  - `get_interpretation_instructions()` – central system prompt for Interpret Map
  - `render_markdown_to_html()` – tasteful Markdown renderer (headings, lists, tables, code, links) with minimal, theme-inheriting styles
  - `get_feature_count_value()`, `resolve_api_key()`, `update_model_parameters()`, `model_supports_images()`
  - Map and layer helpers used across tools
- Provider abstraction: `arcgispro_ai/core/api_clients.py`
  - `get_client(source, api_key, **kwargs)` – returns the right provider client (OpenRouter, OpenAI, Azure OpenAI, Anthropic/Claude, DeepSeek, local)

### Providers, Models, and Environment Variables

- Environment variables: configure the provider(s) you use
  - `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `AZURE_OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`
- Toolbox UI wiring:
  - `update_model_parameters()` enables/disables fields (e.g., `endpoint`, `deployment`) and dynamically populates `model` options (OpenRouter catalog supported).
  - `resolve_api_key()` maps the selected provider to the appropriate environment variable and surfaces helpful errors.
  - `model_supports_images()` gates use of map screenshots only when a provider/model can accept images.
- Vision usage:
  - If enabled and supported, Interpret Map sends both the textual context and a low-detail screenshot to multimodal models.

### Theming and Markdown (Interpret Map)

- Interpret Map responses can include tasteful Markdown (headings, lists, tables, links, fenced code).
- Output is rendered to HTML via `render_markdown_to_html()` with minimal CSS that inherits ArcGIS Pro’s theme—no hardcoded text colors—so it looks good in both light and dark modes.
- A screenshot preview (if captured and supported) is shown with neutral borders/shadows that work across themes.

 

## Contributing

Make an issue or create a branch for your feature or bug fix, and submit a pull request.

---
