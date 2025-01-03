# ArcGIS Pro AI Toolbox

## Overview

The ArcGIS Pro AI Toolbox is a Python-based toolbox designed to enhance the functionality of ArcGIS Pro by integrating AI capabilities. This toolbox allows users to interact with AI models, process geospatial data, and generate insights directly within the ArcGIS Pro environment. This is a BYOK (bring your own key) implementation.

## Tools

**Add AI Generated Field**: Add new fields with AI-generated text based on existing attributes and user-defined prompts.

**Get Map Info**: Extract information about your current map into a JSON file, providing context for other AI tools.

**Generate Python Code**: Create Python snippets tailored to your workflows and data context.

**Create AI Feature Layer**: Generate new feature layers using prompts to describe the data you need.

**Convert Text to Numeric**: Quickly clean up inconsistent text formats and turn them into numeric fields.

## Supported AI Providers

The toolbox supports multiple AI providers:

### OpenAI
- Default provider
- Supports GPT-4 and other OpenAI models
- Requires: `OPENAI_API_KEY` environment variable
- Configuration:
  - Model: Select the OpenAI model to use (default: gpt-4)

### Azure OpenAI
- Microsoft's Azure-hosted OpenAI service
- Requires: `AZURE_OPENAI_API_KEY` environment variable
- Configuration:
  - Model: The deployed model name
  - Endpoint: Your Azure OpenAI endpoint URL
  - Deployment Name: The deployment name for your model

### Claude (Anthropic)
- Anthropic's Claude models
- Requires: `ANTHROPIC_API_KEY` environment variable
- Configuration:
  - Model: Select Claude model (default: claude-3-opus-20240229)

### DeepSeek
- DeepSeek's language models
- Requires: `DEEPSEEK_API_KEY` environment variable
- Configuration:
  - Model: Select DeepSeek model (default: deepseek-chat)

### Local LLM
- Run against a local LLM server
- No API key required
- Configuration:
  - Endpoint: Server URL (default: http://localhost:8000)
  - Must implement OpenAI-compatible chat completions API

### Wolfram Alpha
- For mathematical and computational queries
- Requires: `WOLFRAM_ALPHA_API_KEY` environment variable
- No additional configuration needed

## Installation

1. Set up the required environment variables for your chosen AI provider(s):
   ```batch
   setx OPENAI_API_KEY "your-key-here"
   setx AZURE_OPENAI_API_KEY "your-key-here"
   setx ANTHROPIC_API_KEY "your-key-here"
   setx DEEPSEEK_API_KEY "your-key-here"
   ```

2. For local LLM setup:
   - Deploy a compatible LLM server that implements the OpenAI chat completions API
   - Configure the endpoint URL in the tool interface (defaults to http://localhost:8000)

## Usage

1. Select your preferred AI provider from the dropdown in each tool
2. Configure any provider-specific settings (model, endpoint, etc.)
3. Enter your prompt or query
4. Execute the tool

Each tool will use the selected provider to generate responses, with automatic fallback to OpenAI if the selected provider is not configured.

## Contributing

