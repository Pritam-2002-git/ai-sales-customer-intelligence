
from pathlib import Path
import subprocess
import json

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Sales & Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Supports both the root folder and nested project folder.
PROJECT_DIR = (
    BASE_DIR
    if (BASE_DIR / "models").exists()
    else BASE_DIR / "AI_Sales_Customer_Intelligence"
)

DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"

SALES_PATH = DATA_DIR / "clean_sales_data.csv"
CUSTOMER_PATH = DATA_DIR / "customer_features.csv"
SEGMENTS_PATH = DATA_DIR / "customer_segments.csv"

MODEL_PATH = MODEL_DIR / "churn_model.pkl"
FEATURES_PATH = MODEL_DIR / "churn_features.pkl"

# Your screenshots show churn_scale.pkl.
# Also support churn_scaler.pkl if that is the actual filename.
SCALER_PATH = (
    MODEL_DIR / "churn_scale.pkl"
    if (MODEL_DIR / "churn_scale.pkl").exists()
    else MODEL_DIR / "churn_scaler.pkl"
)

REPORT_PATH = REPORT_DIR / "ai_business_report.txt"


# ============================================================
# 3. LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    """Load the processed CSV files."""

    sales = pd.read_csv(SALES_PATH)

    customers = pd.read_csv(CUSTOMER_PATH)

    segments = pd.read_csv(SEGMENTS_PATH)

    # Convert date columns safely
    if "Order_Date" in sales.columns:
        sales["Order_Date"] = pd.to_datetime(
            sales["Order_Date"],
            errors="coerce"
        )

    if "Last_Purchase_Date" in customers.columns:
        customers["Last_Purchase_Date"] = pd.to_datetime(
            customers["Last_Purchase_Date"],
            errors="coerce"
        )

    return sales, customers, segments


# ============================================================
# 4. LOAD TRAINED CHURN MODEL
# ============================================================

@st.cache_resource
def load_churn_model():
    """Load the trained Random Forest model and its features."""

    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)

    return model, features


# Check required files before loading
required_files = [
    SALES_PATH,
    CUSTOMER_PATH,
    SEGMENTS_PATH,
    MODEL_PATH,
    FEATURES_PATH
]

missing_files = [
    str(path)
    for path in required_files
    if not path.exists()
]

if missing_files:
    st.error(
        "The following required files were not found:\n\n"
        + "\n".join(missing_files)
    )

    st.info(
        "Check that app.py is in the correct project folder "
        "and that the data and models folders exist."
    )

    st.stop()


try:
    df, customer_df, segments_df = load_data()

    churn_model, churn_features = load_churn_model()

except Exception as e:
    st.error(f"Error loading project files: {e}")
    st.stop()


# ============================================================
# 5. VALIDATE AND PREPARE DATA
# ============================================================

required_sales_columns = [
    "Order_ID",
    "Customer_ID",
    "Order_Date",
    "Revenue",
    "Category",
    "Product",
    "Region",
    "Customer_Type"
]

missing_columns = [
    col for col in required_sales_columns
    if col not in df.columns
]

if missing_columns:
    st.error(
        f"Missing columns in sales data: {missing_columns}"
    )
    st.stop()


# Ensure numerical columns are numeric
df["Revenue"] = pd.to_numeric(
    df["Revenue"], errors="coerce"
).fillna(0)

df["Order_Date"] = pd.to_datetime(
    df["Order_Date"], errors="coerce"
)

# Ensure dates are available for monthly analytics
df["Month"] = df["Order_Date"].dt.strftime("%b %Y")


# ============================================================
# 6. BUSINESS METRICS
# ============================================================

total_revenue = df["Revenue"].sum()

total_orders = df["Order_ID"].nunique()

total_customers = df["Customer_ID"].nunique()

average_order_value = (
    total_revenue / total_orders
    if total_orders > 0
    else 0
)

top_category = (
    df.groupby("Category")["Revenue"]
    .sum()
    .idxmax()
)

top_product = (
    df.groupby("Product")["Revenue"]
    .sum()
    .idxmax()
)

top_region = (
    df.groupby("Region")["Revenue"]
    .sum()
    .idxmax()
)

monthly_sales = (
    df.dropna(subset=["Order_Date"])
    .groupby(df["Order_Date"].dt.to_period("M"))["Revenue"]
    .sum()
    .sort_index()
)

best_month = (
    monthly_sales.idxmax().strftime("%B %Y")
    if not monthly_sales.empty
    else "Not available"
)

# Calculate customer risk counts
if "Customer_Segment" in segments_df.columns:
    at_risk_customers = (
        segments_df["Customer_Segment"]
        == "At-Risk Customers"
    ).sum()
else:
    at_risk_customers = 0

if "Churn" in customer_df.columns:
    churn_risk_customers = (
        customer_df["Churn"] == 1
    ).sum()
else:
    churn_risk_customers = 0


# ============================================================
# 7. SIDEBAR
# ============================================================

st.sidebar.title("📊 AI Sales Intelligence")

st.sidebar.markdown(
    """
    **AI-powered business analytics**

    - Sales performance
    - Customer segmentation
    - Churn prediction
    - AI business insights
    """
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Sales Analytics",
        "Customer Segmentation",
        "Churn Prediction",
        "AI Business Insights"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Built with Python, Pandas, Scikit-learn, "
    "Streamlit, Plotly and Ollama."
)


# ============================================================
# 8. DASHBOARD
# ============================================================

if page == "Dashboard":

    st.title("📊 AI Sales & Customer Intelligence")
    st.markdown(
        "### Business Performance Dashboard"
    )

    st.write(
        "Monitor sales, customer behavior, and churn risk "
        "using your machine learning pipeline."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Revenue",
        f"₹{total_revenue:,.0f}"
    )

    col2.metric(
        "Total Orders",
        f"{total_orders:,}"
    )

    col3.metric(
        "Customers",
        f"{total_customers:,}"
    )

    col4.metric(
        "Average Order Value",
        f"₹{average_order_value:,.0f}"
    )

    st.divider()

    # Revenue by month
    left, right = st.columns(2)

    with left:
        st.subheader("Monthly Revenue Trend")

        monthly_df = (
            df.dropna(subset=["Order_Date"])
            .groupby(
                df["Order_Date"].dt.to_period("M")
            )["Revenue"]
            .sum()
            .reset_index()
        )

        monthly_df["Month"] = (
            monthly_df["Order_Date"].astype(str)
        )

        fig = px.line(
            monthly_df,
            x="Month",
            y="Revenue",
            markers=True,
            title="Monthly Sales Revenue"
        )

        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Revenue (₹)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:
        st.subheader("Revenue by Category")

        category_df = (
            df.groupby("Category")["Revenue"]
            .sum()
            .reset_index()
            .sort_values("Revenue", ascending=False)
        )

        fig = px.bar(
            category_df,
            x="Category",
            y="Revenue",
            color="Category",
            title="Category-wise Revenue"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # Business summary
    st.subheader("Key Business Drivers")

    c1, c2, c3 = st.columns(3)

    c1.info(f"🏆 Top Category: {top_category}")
    c2.info(f"📦 Top Product: {top_product}")
    c3.info(f"🌍 Top Region: {top_region}")

    st.subheader("Customer Risk Overview")

    r1, r2 = st.columns(2)

    r1.metric(
        "At-Risk Customer Segment",
        int(at_risk_customers)
    )

    r2.metric(
        "Inactivity-Based Churn Risk",
        int(churn_risk_customers)
    )


# ============================================================
# 9. SALES ANALYTICS
# ============================================================

elif page == "Sales Analytics":

    st.title("📈 Sales Analytics")

    st.write(
        "Explore sales performance by product, category, "
        "region, and customer type."
    )

    # Filters
    col1, col2 = st.columns(2)

    with col1:
        categories = sorted(
            df["Category"].dropna().unique().tolist()
        )

        selected_categories = st.multiselect(
            "Select Category",
            categories,
            default=categories
        )

    with col2:
        regions = sorted(
            df["Region"].dropna().unique().tolist()
        )

        selected_regions = st.multiselect(
            "Select Region",
            regions,
            default=regions
        )

    filtered_df = df[
        df["Category"].isin(selected_categories)
        & df["Region"].isin(selected_regions)
    ].copy()

    if filtered_df.empty:
        st.warning("No sales data matches these filters.")

    else:

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Filtered Revenue",
            f"₹{filtered_df['Revenue'].sum():,.0f}"
        )

        col2.metric(
            "Filtered Orders",
            filtered_df["Order_ID"].nunique()
        )

        col3.metric(
            "Filtered Customers",
            filtered_df["Customer_ID"].nunique()
        )

        st.divider()

        left, right = st.columns(2)

        with left:
            st.subheader("Revenue by Region")

            region_df = (
                filtered_df.groupby("Region")["Revenue"]
                .sum()
                .reset_index()
                .sort_values("Revenue", ascending=False)
            )

            fig = px.bar(
                region_df,
                x="Region",
                y="Revenue",
                color="Region",
                title="Regional Revenue"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with right:
            st.subheader("Top 10 Products")

            product_df = (
                filtered_df.groupby("Product")["Revenue"]
                .sum()
                .reset_index()
                .nlargest(10, "Revenue")
            )

            fig = px.bar(
                product_df.sort_values("Revenue"),
                x="Revenue",
                y="Product",
                orientation="h",
                title="Top Products by Revenue"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.subheader("Revenue by Customer Type")

        customer_type_df = (
            filtered_df.groupby("Customer_Type")["Revenue"]
            .sum()
            .reset_index()
        )

        fig = px.pie(
            customer_type_df,
            names="Customer_Type",
            values="Revenue",
            hole=0.4,
            title="Customer Type Revenue Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Sales Data")

        st.dataframe(
            filtered_df,
            use_container_width=True
        )

        st.download_button(
            "Download Filtered Sales CSV",
            data=filtered_df.to_csv(index=False),
            file_name="filtered_sales_data.csv",
            mime="text/csv"
        )


# ============================================================
# 10. CUSTOMER SEGMENTATION
# ============================================================

elif page == "Customer Segmentation":

    st.title("👥 Customer Segmentation")

    st.write(
        "Explore the KMeans customer segments generated "
        "during model development."
    )

    if "Customer_Segment" not in segments_df.columns:
        st.error(
            "Customer_Segment column is missing from "
            "customer_segments.csv."
        )
        st.stop()

    segment_counts = (
        segments_df["Customer_Segment"]
        .value_counts()
        .reset_index()
    )

    segment_counts.columns = [
        "Customer_Segment",
        "Customers"
    ]

    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(
            segment_counts,
            names="Customer_Segment",
            values="Customers",
            hole=0.4,
            title="Customer Segment Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        if (
            "Total_Revenue" in segments_df.columns
            and "Customer_Segment" in segments_df.columns
        ):

            segment_revenue = (
                segments_df.groupby("Customer_Segment")
                ["Total_Revenue"]
                .mean()
                .reset_index()
            )

            fig = px.bar(
                segment_revenue,
                x="Customer_Segment",
                y="Total_Revenue",
                color="Customer_Segment",
                title="Average Revenue per Segment"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    st.subheader("Segment Summary")

    st.dataframe(
        segment_counts,
        use_container_width=True
    )

    st.subheader("Customer Details")

    selected_segment = st.selectbox(
        "Select a Customer Segment",
        segments_df["Customer_Segment"].dropna().unique()
    )

    selected_customers = segments_df[
        segments_df["Customer_Segment"] == selected_segment
    ]

    st.dataframe(
        selected_customers,
        use_container_width=True
    )

    st.download_button(
        "Download Customer Segment CSV",
        data=selected_customers.to_csv(index=False),
        file_name="customer_segment_details.csv",
        mime="text/csv"
    )


# ============================================================
# 11. CHURN PREDICTION
# ============================================================

elif page == "Churn Prediction":

    st.title("⚠️ Customer Churn Risk Prediction")

    st.warning(
        "This model predicts the inactivity-based churn "
        "proxy used during training. It is not a verified "
        "real-world churn probability."
    )

    st.write(
        "Enter customer information to estimate whether "
        "the customer belongs to the churn-risk class."
    )

    with st.form("churn_prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            total_orders_input = st.number_input(
                "Total Orders",
                min_value=0,
                max_value=10000,
                value=5
            )

            total_revenue_input = st.number_input(
                "Total Revenue (₹)",
                min_value=0.0,
                value=100000.0,
                step=1000.0
            )

            average_order_value_input = st.number_input(
                "Average Order Value (₹)",
                min_value=0.0,
                value=20000.0,
                step=500.0
            )

        with col2:

            total_quantity_input = st.number_input(
                "Total Quantity",
                min_value=0,
                max_value=100000,
                value=10
            )

            average_discount_input = st.number_input(
                "Average Discount",
                min_value=0.0,
                max_value=1.0,
                value=0.10,
                step=0.01,
                help="Enter a decimal, for example 0.10 for 10%."
            )

        submitted = st.form_submit_button(
            "Predict Churn Risk",
            type="primary"
        )

    if submitted:

        input_values = {
            "Total_Orders": total_orders_input,
            "Total_Revenue": total_revenue_input,
            "Average_Order_Value": average_order_value_input,
            "Total_Quantity": total_quantity_input,
            "Average_Discount": average_discount_input
        }

        try:

            # Preserve the exact training feature order
            input_df = pd.DataFrame(
                [[input_values[feature]
                  for feature in churn_features]],
                columns=churn_features
            )

            prediction = churn_model.predict(input_df)[0]

            probability = churn_model.predict_proba(
                input_df
            )[0]

            # Find the probability associated with class 1
            class_list = list(churn_model.classes_)

            if 1 in class_list:
                churn_probability = probability[
                    class_list.index(1)
                ]
            else:
                churn_probability = 0.0

            st.divider()

            if prediction == 1:

                st.error(
                    "⚠️ Prediction: Churn Risk"
                )

                st.write(
                    "The model classifies this customer "
                    "as belonging to the inactivity-based "
                    "churn-risk class."
                )

            else:

                st.success(
                    "✅ Prediction: Active"
                )

                st.write(
                    "The model classifies this customer "
                    "as belonging to the active class."
                )

            st.metric(
                "Churn Proxy Class Probability",
                f"{churn_probability * 100:.2f}%"
            )

            st.progress(
                float(churn_probability)
            )

            st.caption(
                "This probability refers to the training "
                "target, defined as inactivity exceeding "
                "120 days. It is not a calibrated guarantee "
                "of future customer behavior."
            )

        except Exception as e:

            st.error(
                f"Prediction error: {e}"
            )


# ============================================================
# 12. AI BUSINESS INSIGHTS WITH OLLAMA
# ============================================================

elif page == "AI Business Insights":

    st.title("🤖 AI Business Insights")

    st.write(
        "Generate a natural-language business report "
        "using your local Ollama LLM."
    )

    st.info(
        "Make sure Ollama is running on your computer "
        "and the llama3.2:3b model is installed."
    )

    # Prepare compact business context
    business_context = f"""
    SALES PERFORMANCE
    Total Revenue: ₹{total_revenue:,.2f}
    Total Orders: {total_orders}
    Total Customers: {total_customers}
    Average Order Value: ₹{average_order_value:,.2f}

    BUSINESS DRIVERS
    Top Category: {top_category}
    Top Product: {top_product}
    Top Region: {top_region}
    Highest Revenue Month: {best_month}

    CUSTOMER RISK
    At-Risk Customer Segment Count: {at_risk_customers}
    Inactivity-Based Churn Risk Count: {churn_risk_customers}

    The churn label is a proxy based on inactivity
    exceeding 120 days, not an observed real-world churn label.
    """

    st.subheader("Business Context")

    with st.expander("View metrics sent to the LLM"):
        st.text(business_context)

    if st.button(
        "Generate AI Business Report",
        type="primary"
    ):

        prompt = f"""
        You are a professional sales and customer
        intelligence analyst.

        Analyze the following business data:

        {business_context}

        Generate a clear business report with these sections:

        1. SALES PERFORMANCE
        2. CUSTOMER BEHAVIOR
        3. CHURN RISK
        4. KEY INSIGHTS
        5. ACTIONABLE RECOMMENDATIONS

        Rules:
        - Use only the supplied metrics.
        - Do not invent statistics or facts.
        - Clearly identify possible interpretations.
        - Explain that churn is based on an inactivity proxy.
        - Keep recommendations practical and concise.
        - Write for a business manager.
        """

        with st.spinner(
            "Ollama is generating your report..."
        ):

            try:

                response = subprocess.run(
                    [
                        "ollama",
                        "run",
                        "llama3.2:3b",
                        prompt
                    ],
                    capture_output=True,
                    text=True,
                    timeout=240,
                    encoding="utf-8",
                    errors="replace"
                )

                if response.returncode == 0:

                    ai_report = response.stdout.strip()

                    if ai_report:

                        st.success(
                            "AI Business Report Generated!"
                        )

                        st.markdown(ai_report)

                        st.download_button(
                            "Download AI Report",
                            data=ai_report,
                            file_name="ai_business_insights.txt",
                            mime="text/plain"
                        )

                    else:

                        st.warning(
                            "Ollama returned an empty response."
                        )

                else:

                    st.error(
                        "Ollama returned an error."
                    )

                    st.code(response.stderr)

            except FileNotFoundError:

                st.error(
                    "Ollama was not found in the PATH. "
                    "Restart VS Code after installing Ollama "
                    "or use the full path to ollama.exe."
                )

            except subprocess.TimeoutExpired:

                st.error(
                    "Ollama took too long to respond. "
                    "Try again after confirming the model "
                    "is loaded."
                )

            except Exception as e:

                st.error(
                    f"AI report error: {e}"
                )


# ============================================================
# 13. FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Sales & Customer Intelligence Platform | "
    "Python • Machine Learning • Streamlit • Ollama"
)