
# 📊 AI Sales & Customer Intelligence Platform

An AI-powered business analytics platform that helps businesses understand sales performance, segment customers, identify potential churn risks, and generate actionable business insights using Machine Learning and Generative AI.

The application is built using Python, Pandas, Scikit-learn, Streamlit, Plotly, and Ollama.

---

## 🚀 Project Overview

The AI Sales & Customer Intelligence Platform transforms sales transaction data into meaningful business insights.

It combines data analytics, machine learning, and a locally running Large Language Model (LLM) to support business decision-making and customer relationship management.

The platform provides interactive dashboards, customer segmentation, churn-risk prediction, and AI-generated business recommendations through a Streamlit web application.

---

## ✨ Key Features

### 1. Business Performance Dashboard
- Monitor total revenue, total orders, and customer count.
- Calculate average order value.
- Visualize monthly revenue trends.
- Analyze revenue by product category.

### 2. Sales Analytics
- Analyze revenue by category, product, and region.
- Identify top-performing products.
- Explore monthly sales trends.
- Compare customer purchasing behavior.

### 3. Customer Segmentation
- Apply K-Means clustering to customer purchasing data.
- Group customers into four segments:
  - Premium Customers
  - Regular Customers
  - At-Risk Customers
  - Loyal High-Value Customers
- Visualize customer distribution and revenue across segments.

### 4. Churn Prediction
- Use a Random Forest Classifier to estimate customer churn risk.
- Analyze customer purchase history and behavioral features.
- Display churn-risk predictions and probabilities.
- Support customer retention analysis.

**Note:** The current dataset uses an inactivity-based proxy label (more than 120 days since the last purchase) rather than verified historical churn records. Predictions should therefore be interpreted as estimates of this proxy target, not confirmed real-world customer churn.

### 5. AI Business Insights
- Generate business reports using a locally running LLM through Ollama.
- Summarize sales performance and customer behavior.
- Identify potential business opportunities.
- Generate practical recommendations for customer retention and sales improvement.

---

## 🖥️ Application Screenshots

### Business Performance Dashboard

The dashboard provides an overview of sales performance, revenue trends, customer activity, and category-wise revenue.

![Business Performance Dashboard](screenshots/dashboard.png)

### Customer Segmentation<img width="1917" height="1026" alt="Screenshot 2026-09-28 120814" src="https://github.com/user-attachments/assets/f15ce702-3ba0-4fe1-b2ca-1b8fd4d02a91" />
<img width="1915" height="965" alt="Screenshot 2026-09-28 122740" src="https://github.com/user-attachments/assets/39d70fa1-e2eb-4d60-b8ae-a93442d33a2b" />


Explore customer segments generated using K-Means clustering and compare their distribution and revenue contribution.

![Customer Segmentation](screenshots/customer-segmentation.png)

### Churn Prediction

Analyze customer churn-risk predictions based on customer purchase history and machine learning.

![Churn Prediction](screenshots/churn-prediction.png)

### AI Business Insights

Generate AI-powered business reports and actionable recommendations using the local Ollama language model.

![AI Business Insights](screenshots/ai-business-insights.png)

---

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Data Visualization | Plotly |
| Machine Learning | Scikit-learn |
| Customer Segmentation | K-Means Clustering |
| Churn Prediction | Random Forest Classifier |
| Generative AI | Ollama, Llama 3.2 (3B) |
| Web Application | Streamlit |
| Model Persistence | Joblib |
| Development Environment | VS Code, Jupyter Notebook / Google Colab |

---

## 📂 Project Structure

```text
AI-Sales-Customer-Intelligence/
│
├── app.py
│
├── data/
│   └── processed/
│       ├── clean_sales_data.csv
│       ├── customer_features.csv
│       └── customer_segments.csv
│
├── models/
│   ├── churn_model.pkl
│   ├── churn_scaler.pkl
│   └── churn_features.pkl
│
├── reports/
│   └── ai_business_report.txt
│
├── screenshots/
│   ├── dashboard.png
│   ├── customer-segmentation.png
│   ├── churn-prediction.png
│   └── ai-business-insights.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation and Setup

Follow these steps to run the project locally on your machine.

### Step 1: Clone the Repository

```bash
git clone https://github.com/Pritam-2002-git/ai-sales-customer-intelligence.git
```

Navigate to the project directory:

```bash
cd ai-sales-customer-intelligence
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the virtual environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, use Command Prompt instead:

```cmd
.venv\Scripts\activate.bat
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install and Run Ollama

Download Ollama from:

https://ollama.com/download

After installation, open a terminal and download the model:

```bash
ollama pull llama3.2:3b
```

Start the Ollama service if it is not already running:

```bash
ollama serve
```

Keep Ollama running while using the AI Business Insights feature.

### Step 5: Run the Streamlit Application

Open a new terminal in the project directory, activate the virtual environment, and execute:

```bash
python -m streamlit run app.py
```

The application will open in your browser at:

http://localhost:8501

---

## 📊 Machine Learning Workflow

The project follows a complete machine learning workflow:

1. Data collection and preprocessing
2. Exploratory Data Analysis (EDA)
3. Customer feature engineering
4. Customer segmentation using K-Means
5. Churn-risk modeling using Random Forest
6. Model evaluation and comparison
7. Model serialization using Joblib
8. Integration with the Streamlit application
9. AI-generated business insights using Ollama

---

## 📈 Business Insights

The synthetic sales dataset used during development contains:

| Metric | Value |
|---|---:|
| Total Revenue | ₹59,823,398.59 |
| Total Orders | 1,500 |
| Unique Customers | 298 |
| Top Revenue Category | Electronics |
| Top Revenue Product | Laptop |
| Top Revenue Region | North |
| Highest Revenue Month | August |

These figures represent the development dataset and are not actual business or customer records.

---

## 🔮 Future Enhancements

- Integrate real-time sales data from business databases.
- Improve churn modeling using verified customer churn labels.
- Add advanced customer lifetime value (CLV) analysis.
- Integrate cloud-based deployment.
- Add automated sales forecasting.
- Build interactive business reports and exportable PDF summaries.

---

## 👨‍💻 Author

**Pritam Mahamansingh**

Aspiring Data Analyst | Machine Learning & AI Enthusiast

GitHub: [Pritam-2002-git](https://github.com/Pritam-2002-git)

---

## 📄 License

This project is intended for educational and portfolio purposes.
<img width="1917" height="921" alt="Screenshot 2026-09-28 122710" src="https://github.com/user-attachments/assets/faf5da79-1543-4b0b-b8fe-ece77810369b" />
<img width="1917" height="911" alt="Screenshot 2026-09-28 120830" src="https://github.com/user-attachments/assets/736fe38f-b5e9-4147-98d5-bebdd91c0698" />
<img width="1917" height="1026" alt="Screenshot 2026-09-28 120814" src="https://github.com/user-attachments/assets/03afc28e-eff7-4168-b4f1-0c567c5f4fc7" />
