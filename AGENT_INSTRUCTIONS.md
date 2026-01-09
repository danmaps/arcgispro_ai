# Agent Instructions

## Architecture Overview
- The maintainable source lives in modular packages under arcgispro_ai/ and arcgispro_ai/toolboxes/, keeping utilities, API clients, and tool logic separated for clarity and reuse.
- A single distributable toolbox arcgispro_ai.pyt is produced for ArcGIS Pro users; it inlines all required code so no other repo files are needed when importing the toolbox in ArcGIS Pro.
- build_monolithic_pyt.py owns the bundling process, stripping internal imports and ensuring the generated .pyt mirrors the package version defined in setup.py.

## Separation of Concerns (Toolbox vs. Logic)
- Keep `arcgispro_ai/toolboxes/arcgispro_ai_tools.pyt` focused strictly on ESRI Python toolbox structure: toolbox class metadata, parameter definitions, ArcPy validation hooks, and `execute()` wiring.
- Move provider-agnostic logic out of the .pyt and import it instead: prompts, Markdown/HTML rendering, API client calls, serialization, parsing, and utility helpers belong in importable modules (e.g., `arcgispro_ai/…/arcgispro_ai_utils.py`, `arcgispro_ai/…/core/api_clients.py`).
- Do not embed large prompt strings, HTML builders, or complex business logic directly inside the .pyt. Expose them through small, well-named functions and import them.
- The monolithic `arcgispro_ai.pyt` is a generated artifact that flattens imports for distribution only. Do not mirror that style in the development .pyt; keep it lean and delegating.

## Agent Development Workflow
1. Edit only the modular sources (arcgispro_ai/toolboxes/arcgispro_ai_tools.pyt and any modules it imports). Treat these files as the source of truth.
2. **Never modify** the generated monolithic file C:\Users\danny\dev\arcgispro_ai\arcgispro_ai.pyt directly; changes will be overwritten the next time the build script runs and risk desynchronizing the distributed toolbox.
3. After making code changes, regenerate the distributable toolbox by running build_monolithic_pyt.py if users need an updated .pyt.
4. Keep utilities, API clients, and tool classes loosely coupled to preserve the repo's modular design and simplify testing.

## Release Automation Notes
- Run release.sh to bump the patch version in setup.py, commit the change, push the tag, and open the GitHub release page; this keeps PyPI, tags, and the generated .pyt version aligned.
- The release script expects the working tree to be clean apart from the version bump; ensure you have already regenerated arcgispro_ai.pyt so the distributable reflects the tagged code.

## Quick Reference
- Source of truth: arcgispro_ai/toolboxes/arcgispro_ai_tools.pyt plus imported modules inside arcgispro_ai/.
- Generated artifact: C:\Users\danny\dev\arcgispro_ai\arcgispro_ai.pyt (do not edit).
- Build script: build_monolithic_pyt.py.
- Release script: release.sh.
