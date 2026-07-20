"""
ai_reporter.py
Sends EDA + ML results to Groq (LLaMA) and returns a structured analytical report.
"""

import json
import os
from groq import Groq
from dotenv import load_dotenv

# Always load .env from the same folder as this script
_base_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_base_dir, ".env"), override=True)


def _build_prompt(eda: dict, ml_results: dict, filename: str) -> str:
    """Construct the full prompt sent to the AI model."""

    correlations = eda.get("correlations") or []
    strong_corrs = (ml_results.get("strong_correlations") or [])[:5]
    anomalies = ml_results.get("anomalies") or {}

    eda_summary = {
        "shape": eda.get("shape"),
        "numeric_columns": eda.get("numeric_columns") or [],
        "categorical_columns": eda.get("categorical_columns") or [],
        "date_columns": eda.get("date_columns") or [],
        "missing_values": eda.get("missing_values") or {},
        "descriptive_stats": eda.get("descriptive_stats") or {},
        "top_correlations": correlations[:5],
        "categorical_stats": eda.get("categorical_stats") or {},
        "outlier_counts": eda.get("outlier_counts") or {},
        "skewness": eda.get("skewness") or {},
    }

    ml_summary = {
        "anomaly_count": anomalies.get("anomaly_count", 0),
        "anomaly_pct": anomalies.get("anomaly_pct", 0.0),
        "trends": ml_results.get("trends") or {},
        "strong_correlations": strong_corrs,
        "distributions": ml_results.get("distributions") or {},
        "pca_variance": ml_results.get("pca") or {},
    }

    prompt = f"""
You are a senior data analyst and business intelligence expert.
A user has uploaded a dataset named "{filename}".

Below is the complete automated analysis:

=== EDA SUMMARY ===
{json.dumps(eda_summary, indent=2, default=str)}

=== MACHINE LEARNING ANALYSIS ===
{json.dumps(ml_summary, indent=2, default=str)}

Based on this data, write a professional, detailed analytical investigation report with the following sections.
Use clear headings, bullet points where appropriate, and business-friendly language.

---

## Executive Summary
A concise 3-5 sentence overview of what the dataset contains, its key characteristics, and the most important finding.

## Key Findings
List 5-8 specific, data-backed findings. Include actual numbers where available. Each finding should be a separate bullet.

## Trend Analysis
Describe the direction and significance of trends found in numeric columns. Note which columns are increasing, decreasing, or flat.

## Anomalies and Data Quality Issues
Describe the anomalies detected, their count and percentage. Highlight any data quality concerns.

## Business Recommendations
Provide 4-6 actionable business recommendations based on the data findings.

## Future Risks and Opportunities
Identify 3-4 potential future risks the data suggests. Identify 2-3 opportunities for growth or improvement.

## Methodology Note
Briefly explain the analysis pipeline used.

---

Be specific, insightful, and professional. Do not make up numbers that are not in the analysis above.
"""
    return prompt


def generate_report(eda: dict, ml_results: dict, filename: str, api_key: str) -> str:
    """
    Call Groq API and return the full markdown report string.
    """
    client = Groq(api_key=api_key)

    prompt = _build_prompt(eda, ml_results, filename)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a senior data analyst who writes clear, professional business reports based on data analysis results.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.4,
        max_tokens=4096,
    )
    return response.choices[0].message.content


def generate_quick_summary(df_shape: tuple, eda: dict, api_key: str) -> str:
    """
    Generate a short 3-bullet quick take shown at the top of the dashboard.
    """
    client = Groq(api_key=api_key)

    correlations = eda.get("correlations") or []
    prompt = f"""
Dataset: {df_shape[0]} rows x {df_shape[1]} columns.
Numeric columns: {eda.get('numeric_columns') or []}.
Categorical columns: {eda.get('categorical_columns') or []}.
Top correlations: {correlations[:3]}.
Outlier counts: {eda.get('outlier_counts') or {}}.

Write exactly 3 sharp, one-sentence bullet-point insights a business executive would care about.
Start each bullet with an emoji. Be concise. No headers.
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=256,
    )
    return response.choices[0].message.content
