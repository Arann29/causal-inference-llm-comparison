# app/utils/prompts.py
"""Dynamic prompt generation for LLM causal analysis experiments."""

from typing import Optional, List, Dict
from app.utils.llm_config import ExperimentConfig


def _build_step2_explicit(config: Optional[ExperimentConfig] = None, threshold_var: Optional[str] = None, threshold_value: Optional[float] = None, df_columns: Optional[List[str]] = None) -> str:
    """Build Step 2 prompt for regime characterization.
    
    Args:
        config: Experiment configuration (optional)
        threshold_var: Name of the threshold variable (always provided)
        threshold_value: Numeric threshold value (provided when explicit_threshold=True)
        df_columns: First two column names from dataframe for anonymization mapping
        
    Returns:
        Formatted prompt string for Step 2
    """
    # Determine threshold variable display name
    threshold_display = threshold_var
    if config and config.raw_data_options.get('anonymize_names', False) and df_columns and threshold_var:
        # Map actual variable name to anonymized name (Variable_X or Variable_Y)
        if len(df_columns) >= 2:
            if threshold_var == df_columns[0]:
                threshold_display = 'Variable_X'
            elif threshold_var == df_columns[1]:
                threshold_display = 'Variable_Y'
            else:
                # Threshold variable doesn't match first two columns - use generic name
                threshold_display = 'the segmentation variable'
    
    # Build prompt for explicit threshold case
    if threshold_var and threshold_value is not None:
        instructions = [
            "Role: Data Scientist — Analyze Causal Relationships Across Regimes",
            "",
            "# Dataset Summary",
            "{summary}",
            "",
            "# Your Previous Analysis (Step 1)",
            "{history}",
            "",
            "# Segmentation from Domain Literature",
            f"The data is segmented at **{threshold_display} = {threshold_value}**, yielding two regimes:",
            f"- **Regime 0 (Low):** {threshold_display} <= {threshold_value}",
            f"- **Regime 1 (High):** {threshold_display} > {threshold_value}",
            "",
            "# Instructions",
            "1. For each regime, summarize defining patterns or behaviors.",
            "2. Assess if the causal **direction** reverses between regimes:",
            "    - Reversal means X→Y in one regime switches to Y→X in the other.",
            "    - Only direction changes count as a reversal, not just changes in strength.",
            "3. Briefly explain why the mechanism could differ or stay consistent between regimes."
        ]
    else:
        # Fallback for when threshold is not explicitly provided
        instructions = [
            "Role: Data Scientist — Analyze Causal Relationships Across Regimes",
            "",
            "# Dataset Summary",
            "{summary}",
            "",
            "# Your Previous Analysis (Step 1)",
            "{history}",
            "",
            "# Given Information",
            f"The segmentation variable is **{threshold_display}**. Examine how the causal relationship changes across its range.",
            "",
            "# Instructions",
            "1. For each regime, summarize defining patterns or behaviors.",
            "2. Assess if the causal **direction** reverses between regimes:",
            "    - Reversal means X→Y in one regime switches to Y→X in the other.",
            "    - Only direction changes count as a reversal, not just changes in strength.",
            "3. Briefly explain why the mechanism could differ or stay consistent across the range."
        ]
    
    return '\n'.join(instructions)


def get_prompt_chain(config: Optional[ExperimentConfig] = None, threshold_var: Optional[str] = None, threshold_value: Optional[float] = None, df_columns: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """
    Returns the optimized 4-step prompt chain for causal reasoning.
    
    Args:
        config: Optional experiment configuration for dynamic prompt modification
        threshold_var: Name of the threshold variable (optional, from ground truth)
        threshold_value: Numeric threshold value (optional, from ground truth)
        df_columns: First two column names from dataframe for anonymization mapping
        
    Returns:
        List of prompt steps with name and prompt text
    """
    return [
        {
            "name": "Step 1: Domain Understanding & Hypothesis Generation",
            "prompt": """Role: Data Scientist — Explore Potential Causal Directions in a Bivariate Dataset

Dataset Summary:
{summary}

Instructions:
1. Statistical Overview:
   - Summarize the marginal distributions of both variables.
   - Describe their correlation and note any significant patterns or anomalies.

2. Competing Causal Hypotheses:
   - For X → Y: Propose one plausible scenario where X could cause Y, specifying context and reasoning.
   - For Y → X: Propose one plausible scenario where Y could cause X, specifying context and reasoning.

IMPORTANT: Do not assess or prefer one direction. Present both causal directions as viable and balanced possibilities."""
        },
        {
            "name": "Step 2: Regime Identification",
            "prompt": _build_step2_explicit(config, threshold_var, threshold_value, df_columns)
        },
        {
            "name": "Step 3: Regime-Specific Causal Analysis",
            "prompt": """Role: Data Scientist — Determine Causal Directions and Mechanisms Within Regimes

**Dataset Summary:**
{summary}

**Your Previous Analysis (Steps 1-2):**
---
{history}
---

**Instructions:**
For each regime (Low and High), based on your prior analysis:

1. **Causal Direction:** State if causality runs X→Y or Y→X.
2. **Mechanism:** In 2–3 sentences, explain why this direction applies in the regime.
3. **Confidence:** Choose High, Moderate, or Low.
4. **Evidence:** Briefly note supporting data patterns or examples.

**Direction Switching:**
- Does the causal direction REVERSE between regimes (X→Y ↔ Y→X)? Reply 'Yes' or 'No'.
- If yes: Explain briefly why the mechanism flips.
- If no: Explain briefly why it remains consistent.

Keep responses concise and direct."""
        },
        {
            "name": "Step 4: JSON Output",
            "prompt": """Role: Data Scientist — Extract and Standardize Causal Analysis Results

**Dataset Summary:**
{summary}

**Your Previous Analysis (Steps 1-3):**
---
{history}
---

**Instructions:**
Based on your complete analysis above, output ONLY valid JSON (no text before or after).

Format:
{{
  "threshold_variable": "<actual variable name>",
  "threshold_value": <numeric threshold>,
  "regimes": 2,
  "directions": [
    {{"regime": 0, "direction": "<X->Y or Y->X>", "confidence": "<High|Moderate|Low>"}},
    {{"regime": 1, "direction": "<X->Y or Y->X>", "confidence": "<High|Moderate|Low>"}}
  ]
}}

Rules:
- Use the actual given variable names (not "X->Y")
- regime 0 = Low regime, regime 1 = High regime
- Confidence: High, Moderate, or Low

OUTPUT JSON NOW:"""
        }
    ]
