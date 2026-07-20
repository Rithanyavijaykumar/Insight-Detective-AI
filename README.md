# 🔍 Insight Detective AI

> **"Upload any dataset and receive a complete AI-generated analytical investigation report."**

A full-stack data storytelling application that acts like a **junior data analyst** — automatically cleaning data, performing EDA, detecting anomalies, identifying trends, and generating a professional business report using LLaMA 3.3 70B via Groq.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧹 Auto Data Cleaning | Removes duplicates, fills missing values, fixes data types |
| 📊 Exploratory Data Analysis | Descriptive stats, distributions, correlations, skewness |
| 📈 Trend Detection | Linear regression per column with significance testing |
| 🚨 Anomaly Detection | Isolation Forest ML model highlights suspicious rows |
| 📐 Distribution Analysis | Shapiro-Wilk normality test, skewness classification |
| 🔗 Correlation Analysis | Pearson correlation heatmap + strong pair detection |
| 📉 Interactive Charts | Plotly histograms, box plots, heatmaps, scatter matrix, time series |
| 🤖 AI Report Generation | LLaMA 3.3 writes Executive Summary, Key Findings, Recommendations, Risks |
| 📥 Export | Download cleaned data, anomaly-flagged data, report, EDA stats |

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOURUSERNAME/insight-detective-ai.git
cd insight-detective-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get a free Groq API key

- Go to **https://console.groq.com/keys**
- Sign up free (Google login works)
- Click **Create API Key** — copy the key (starts with `gsk_...`)

### 4. Set up your API key

```bash
copy .env.example .env
```

Open `.env` and add your key:
```
GROQ_API_KEY=gsk_your_key_here
```

### 5. Run the app

```bash
python -m streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📁 Project Structure

```
insight-detective-ai/
├── app.py                  ← Main Streamlit application
├── data_processor.py       ← Data loading, cleaning, EDA
├── visualizer.py           ← All Plotly chart functions
├── ml_analyzer.py          ← Anomaly detection, trends, correlations, PCA
├── ai_reporter.py          ← Groq API integration & prompt engineering
├── requirements.txt        ← Python dependencies
├── .env.example            ← API key template (copy to .env)
└── sample_data/
    └── customer_data.csv   ← Sample dataset to test the app
```

---

## 🧰 Tech Stack

| Library | Purpose |
|---|---|
| **Streamlit** | Web UI |
| **Pandas + NumPy** | Data manipulation |
| **Plotly** | Interactive visualizations |
| **Scikit-Learn** | Isolation Forest, PCA, StandardScaler |
| **SciPy** | Shapiro-Wilk test, linear regression |
| **Groq API** | LLaMA 3.3 70B for AI report generation |

---

## 💡 Example

Upload `sample_data/customer_data.csv` and the app will:

1. Clean 40 rows of customer data
2. Detect correlations between tenure and spend
3. Flag high-support-call customers as anomalies
4. Generate an executive report recommending a loyalty program

---

## ⚠️ Important

- Never commit your `.env` file — it is in `.gitignore`
- The app works without an API key for all EDA/ML features
- AI report requires a valid Groq API key
