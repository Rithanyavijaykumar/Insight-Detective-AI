# 🔍 Insight Detective AI

> **Upload any dataset and receive an AI-powered analytical investigation report.**

Insight Detective AI is a Streamlit-based web application that automates the complete data analysis workflow. Simply upload a CSV dataset, and the application cleans the data, performs exploratory data analysis (EDA), creates interactive visualizations, detects anomalies, and generates AI-powered business insights.

---

## 📌 Features

* 📂 Upload any CSV dataset
* 🧹 Automatic data cleaning & preprocessing
* 📊 Exploratory Data Analysis (EDA)
* 📈 Interactive data visualizations
* 🔍 Trend and correlation analysis
* 🚨 Machine Learning-based anomaly detection
* 🤖 AI-generated Executive Summary
* 💡 Business Recommendations
* ⚠️ Future Risk Analysis
* 📥 Export processed data and reports

---

## 🛠️ Tech Stack

| Category         | Technologies           |
| ---------------- | ---------------------- |
| Programming      | Python                 |
| Data Analysis    | Pandas, NumPy          |
| Visualization    | Plotly, Matplotlib     |
| Machine Learning | Scikit-learn           |
| Web Framework    | Streamlit              |
| AI Integration   | Groq API (Llama Model) |

---

## 📷 Application Workflow

```text
Upload Dataset
       │
       ▼
Data Cleaning
       │
       ▼
Exploratory Data Analysis
       │
       ▼
Interactive Visualizations
       │
       ▼
Trend & Anomaly Detection
       │
       ▼
AI Report Generation
       │
       ▼
Business Insights & Recommendations
```

---

## 📂 Project Structure

```text
Insight-Detective-AI/
│
├── app.py
├── data_processor.py
├── visualizer.py
├── ml_analyzer.py
├── ai_reporter.py
├── requirements.txt
├── README.md
├── customer_data.csv
└── .env.example
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Insight-Detective-AI.git
cd Insight-Detective-AI
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your API Key

Create a `.env` file and add your Groq API key.

```env
GROQ_API_KEY=your_api_key_here
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

---

## 📊 What You'll Get

After uploading a dataset, the application automatically provides:

* Dataset Overview
* Data Cleaning
* Summary Statistics
* Interactive Charts
* Correlation Analysis
* Trend Detection
* Anomaly Detection
* AI-Generated Summary
* Business Recommendations
* Future Risk Analysis

---

## 🎯 Project Objective

The objective of this project is to simplify data analysis by combining automation, machine learning, and AI into a single application. It enables users to transform raw datasets into meaningful insights without writing complex analytical code.

---

## 🔮 Future Improvements

* Predictive Analytics
* Time-Series Forecasting
* PDF Report Export
* Multi-file Analysis
* User Authentication
* Dashboard Customization
* Cloud Deployment

---
