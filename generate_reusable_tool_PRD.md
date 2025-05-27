# Product Requirements Document: Generate Tool

## Overview

"Generate Tool" is a new geoprocessing tool for ArcGIS Pro that transforms a rough Python code sample (or optionally a natural language prompt) into a fully functional, documented, and parameterized Python toolbox (`.pyt`). It serves as the final step in the AI-assisted geospatial workflow, helping users deploy their scripts as reusable tools for themselves or their teams.

If the user provides only a natural language prompt, this tool calls the existing **Generate Python Code** tool under the hood to produce a starting script before generating the `.pyt` file.

**Goal**
Automate the creation of `.pyt` toolboxes to reduce friction in sharing, scaling, and reusing GIS workflows authored in Python.

---

## Key Features

1. **Input Types**

   * Accept raw Python script (preferred input)
   * Optionally accept a high-level natural language prompt

     * If a prompt is provided, run **Generate Python Code** tool internally to convert it to a code snippet

2. **Toolbox Generation**

   * Output a `.pyt` file with:

     * Tool class
     * `getParameterInfo()` with inferred or user-supplied parameters
     * `execute()` function populated with logic
     * Class-level docstrings and metadata

3. **Optional Parameter Control**

   * UI section titled “Parameter Details” (collapsible)
   * Users can define:

     * Parameter names
     * Data types (e.g., Field, Feature Layer, String, Integer)
     * Default values
     * Direction (Input/Output)
     * Multi-value support

4. **Auto-Inference (Fallback)**

   * If no parameter info is supplied, the tool uses LLM-based inference to:

     * Identify potential parameters
     * Guess data types and defaults
     * Generate basic validation logic

5. **Modes**

   * **Simple Mode:** Designed for low-code users, only requires a working code snippet
   * **Advanced Mode:** For Python power users, exposes full control of toolbox config

6. **User Feedback and Logging**

   * Warnings if inference seems uncertain
   * Suggestions for improvement after tool generation
   * Save log of code transformations

---

## User Stories

* *As a GIS analyst with limited coding experience,* I want to turn an AI-generated script into a Pro toolbox so I can reuse it later.
* *As a developer,* I want control over how parameters are named and typed, so my tools are robust and maintainable.
* *As a team lead,* I want junior analysts to package and share their tools cleanly without writing boilerplate.

---

## Out of Scope (v1)

* Support for `.tbx` or script tool XML conversion
* Real-time conversational refinement (follow-up prompts)
* GUI for drawing flowcharts or block-based workflows

---

## Open Questions

* Should it validate the generated `.pyt` file automatically?
* Should it suggest example values/test cases?
* Should we allow embedding of test datasets?

---

## Success Metrics

* Time saved vs manual `.pyt` creation
* % of tools generated that run successfully on first try
* Adoption by internal GIS teams and feedback collected
