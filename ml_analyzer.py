"""
ml_analyzer.py
Trend detection, anomaly detection, and basic ML insights.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats


# ---------------------------------------------------------------------------
# Anomaly Detection
# ---------------------------------------------------------------------------

def detect_anomalies(df: pd.DataFrame, contamination: float = 0.05) -> dict:
    """
    Use Isolation Forest on all numeric columns.
    Returns labels (-1 = anomaly, 1 = normal) and summary stats.
    """
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols) == 0:
        return {"labels": pd.Series(dtype=int), "anomaly_count": 0,
                "anomaly_pct": 0.0, "primary_column": None}

    X = df[num_cols].fillna(df[num_cols].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
    labels = model.fit_predict(X_scaled)
    label_series = pd.Series(labels, index=df.index, name="anomaly")

    anomaly_count = int((labels == -1).sum())
    anomaly_pct = round(anomaly_count / len(labels) * 100, 2)

    return {
        "labels": label_series,
        "anomaly_count": anomaly_count,
        "anomaly_pct": anomaly_pct,
        "primary_column": num_cols[0],
        "numeric_columns_used": num_cols,
    }


# ---------------------------------------------------------------------------
# Trend Detection
# ---------------------------------------------------------------------------

def detect_trends(df: pd.DataFrame) -> dict:
    """
    For each numeric column, compute linear trend (slope, r², p-value).
    Classify as increasing / decreasing / flat.
    """
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    trends = {}
    x = np.arange(len(df))

    for col in num_cols:
        y = df[col].fillna(df[col].median()).values
        if np.std(y) == 0:
            trends[col] = {"direction": "flat", "slope": 0.0, "r_squared": 0.0, "p_value": 1.0}
            continue
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        direction = "increasing" if slope > 0 else "decreasing"
        if p_value > 0.05:
            direction = "flat (not significant)"
        trends[col] = {
            "direction": direction,
            "slope": round(float(slope), 6),
            "r_squared": round(float(r_value ** 2), 4),
            "p_value": round(float(p_value), 6),
        }

    return trends


# ---------------------------------------------------------------------------
# Correlation Insights
# ---------------------------------------------------------------------------

def find_strong_correlations(df: pd.DataFrame, threshold: float = 0.7) -> list[dict]:
    """Return pairs of columns with |correlation| >= threshold."""
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols) < 2:
        return []

    corr = df[num_cols].corr()
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if abs(val) >= threshold:
                pairs.append({
                    "col1": cols[i],
                    "col2": cols[j],
                    "correlation": round(float(val), 4),
                    "strength": "strong positive" if val > 0 else "strong negative",
                })
    return sorted(pairs, key=lambda x: abs(x["correlation"]), reverse=True)


# ---------------------------------------------------------------------------
# Distribution Analysis
# ---------------------------------------------------------------------------

def analyze_distributions(df: pd.DataFrame) -> dict:
    """
    Test each numeric column for normality (Shapiro-Wilk on sample).
    Returns normality flag and skew interpretation.
    """
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    dist_info = {}
    for col in num_cols:
        series = df[col].dropna()
        skew = float(series.skew())

        # Shapiro-Wilk on sample (max 5000 rows)
        sample = series.sample(min(5000, len(series)), random_state=42)
        try:
            _, p_val = stats.shapiro(sample)
            is_normal = p_val > 0.05
        except Exception:
            is_normal = False
            p_val = None

        skew_label = (
            "roughly symmetric" if abs(skew) < 0.5
            else ("moderately skewed" if abs(skew) < 1.0
                  else "highly skewed")
        )
        dist_info[col] = {
            "skewness": round(skew, 4),
            "skew_label": skew_label,
            "is_normal": is_normal,
            "normality_p_value": round(float(p_val), 6) if p_val is not None else None,
        }
    return dist_info


# ---------------------------------------------------------------------------
# PCA Summary
# ---------------------------------------------------------------------------

def pca_summary(df: pd.DataFrame, n_components: int = 3) -> dict:
    """
    Run PCA on numeric columns and return explained variance ratios.
    Useful for high-dimensional datasets.
    """
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols) < 2:
        return {}

    X = df[num_cols].fillna(df[num_cols].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n = min(n_components, len(num_cols), X_scaled.shape[0])
    pca = PCA(n_components=n, random_state=42)
    pca.fit(X_scaled)

    return {
        f"PC{i+1}_variance_explained": round(float(v) * 100, 2)
        for i, v in enumerate(pca.explained_variance_ratio_)
    }


# ---------------------------------------------------------------------------
# Aggregate ML Results
# ---------------------------------------------------------------------------

def run_full_analysis(df: pd.DataFrame) -> dict:
    """Run all ML analyses and return a unified results dict."""
    results = {}
    results["anomalies"] = detect_anomalies(df)
    results["trends"] = detect_trends(df)
    results["strong_correlations"] = find_strong_correlations(df)
    results["distributions"] = analyze_distributions(df)
    results["pca"] = pca_summary(df)
    return results
