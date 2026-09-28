# AI Sales & Customer Intelligence Platform

A simple data analytics and machine learning project that analyzes sales performance, groups customers into segments, predicts customer churn risk, and generates business insights using a locally running Ollama LLM.

## Features

- **Sales Dashboard:** View total revenue, orders, and customers.
- **Sales Analytics:** Explore revenue by category, product, region, and month.
- **Customer Segmentation:** Group customers by purchasing behavior using K-Means clustering.
- **Churn Risk Prediction:** Use a Random Forest model to estimate which customers may become inactive.
- **AI Business Insights:** Generate business summaries and recommendations using Ollama.

## Technologies Used

Python, Pandas, NumPy, Scikit-learn, Streamlit, Plotly, Joblib, and Ollama.

## Project Structure

```text
AI_Sales_Customer_Intelligence/
├── app.py
├── data/
│   └── processed/
│       ├── clean_sales_data.csv
│       ├── customer_features.csv
│       └── customer_segments.csv
├── models/
│   ├── churn_model.pkl
│   ├── churn_scaler.pkl
│   └── churn_features.pkl
├── reports/
│   └── ai_business_report.txt
├── requirements.txt
└── README.md
```

## Run Locally

### 1. Clone the repository

Replace the URL with your GitHub repository URL.

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

### 2. Create and activate a virtual environment (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up Ollama

Install Ollama from [ollama.com](https://ollama.com). Download the model used in your app. For example:

```bash
ollama pull llama3.2:3b
```

Keep Ollama running locally. If `app.py` uses another model name, download that model and ensure the name in the code matches.

### 5. Start the app

Run this command from the directory containing `app.py`:

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal (usually `http://localhost:8501`).

## Machine Learning Notes

- Customer segmentation is performed with K-Means clustering.
- The churn target is based on an inactivity threshold in the sample data. It is a **proxy label**, not verified historical churn data.
- Churn predictions are experimental estimates and should not be treated as production-ready predictions.

## Dataset

This project uses sample sales data for learning and demonstration. Results and insights depend on the data included in the repository.

## Future Improvements

- Connect a real sales database.
- Evaluate churn prediction using verified churn labels.
- Add model monitoring and retraining.
- Deploy the Streamlit app to a cloud platform.

## Author

**Pritam Mahamansingh**
