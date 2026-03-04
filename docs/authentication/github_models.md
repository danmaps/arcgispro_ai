# GitHub Models Authentication

ArcGIS Pro AI Toolbox supports GitHub Models with a local token workflow.

## Setup

1. Open a terminal in the repository root.
2. Run:

```bat
tools\auth\auth_github_models.bat
```

3. The script opens GitHub's token page.
4. Create a token with `models` access (or fine-grained `models:read`).
5. Paste the token into the terminal prompt.

## Token Storage

The token is stored locally at:

```text
%USERPROFILE%\.arcgispro_ai\github_models_token.txt
```

The setup script applies restrictive file permissions so only your user account can read the token.

To revoke local access, delete that file.

## Toolbox Usage

Choose:

- `Source`: `GitHub Models`
- `Model`: for example `openai/gpt-4.1`

The toolbox reads the local token and sends chat completion requests to:

```text
POST https://models.github.ai/inference/chat/completions
```

## Troubleshooting

- `GitHub Models token not found...`
  - Run `tools\auth\auth_github_models.bat` again.
- `401 Unauthorized`
  - Verify token scope/permission, expiration, and pasted value.
