"""
data_processor.py
Handles data ingestion, cleaning, and exploratory data analysis (EDA).
"""

import pandas as pd
import numpy as np
from io import BytesIO


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def load_dataset(uploaded_file) -> pd.DataFrame:
    """Read CSV or Excel file uploaded via Streamlit."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")
    return df


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Auto-clean the dataframe.
    Returns cleaned dataframe and a cleaning report dict.
    """
    report = {}
    original_shape = df.shape

    # 1. Drop fully-duplicate rows
    dupes = df.duplicated().sum()
    df = df.drop_duplicates()
    report["duplicate_rows_removed"] = int(dupes)

    # 2. Strip whitespace from string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # 3. Attempt to parse object columns that look like numbers
    for col in str_cols:
        try:
            converted = pd.to_numeric(df[col], errors="raise")
            df[col] = converted
        except (ValueError, TypeError):
            pass

    # 4. Attempt to parse object columns that look like dates
    str_cols_remaining = df.select_dtypes(include="object").columns
    for col in str_cols_remaining:
        if any(kw in col.lower() for kw in ["date", "time", "dt", "year", "month"]):
            try:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
            except Exception:
                pass

    # 5. Report & fill missing values
    missing_before = df.isnull().sum()
    missing_cols = missing_before[missing_before > 0].to_dict()
    report["missing_values_before"] = {k: int(v) for k, v in missing_cols.items()}

    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include="object").columns

    # Fill numeric NaNs with median
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Fill categorical NaNs with mode
    for col in cat_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val[0] if len(mode_val) else "Unknown")

    report["final_shape"] = df.shape
    report["original_shape"] = original_shape
    report["rows_after_cleaning"] = df.shape[0]
    report["columns"] = df.shape[1]

    return df, report


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------

def perform_eda(df: pd.DataFrame) -> dict:
    """
    Compute key EDA statistics.
    Returns a structured dict consumed by the visualizer and AI reporter.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    date_cols = df.select_dtypes(include=["datetime64[ns]", "datetime"]).columns.tolist()

    eda = {
        "shape": df.shape,
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "date_columns": date_cols,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "descriptive_stats": {},
        "categorical_stats": {},
        "correlations": {},
        "skewness": {},
        "outlier_counts": {},
    }

    # Descriptive statistics for numeric columns
    if num_cols:
        desc = df[num_cols].describe().to_dict()
        eda["descriptive_stats"] = {
            col: {k: round(float(v), 4) for k, v in stats.items()}
            for col, stats in desc.items()
        }

        # Correlation matrix (top pairs)
        if len(num_cols) > 1:
            corr = df[num_cols].corr()
            pairs = (
                corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                .stack()
                .reset_index()
            )
            pairs.columns = ["col1", "col2", "correlation"]
            pairs["abs_corr"] = pairs["correlation"].abs()
            top_pairs = pairs.nlargest(10, "abs_corr")[["col1", "col2", "correlation"]]
            eda["correlations"] = top_pairs.to_dict(orient="records")

        # Skewness
        eda["skewness"] = {
            col: round(float(df[col].skew()), 4) for col in num_cols
        }

        # Outlier counts via IQR
        for col in num_cols:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            outliers = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
            eda["outlier_counts"][col] = int(outliers)

    # Categorical stats
    for col in cat_cols:
        vc = df[col].value_counts()
        eda["categorical_stats"][col] = {
            "unique_values": int(df[col].nunique()),
            "top_value": str(vc.index[0]) if len(vc) else "N/A",
            "top_count": int(vc.iloc[0]) if len(vc) else 0,
            "top_5": vc.head(5).to_dict(),
        }

    return eda
