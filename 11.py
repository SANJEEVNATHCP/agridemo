# msme_ews_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="MSME Early Warning System", layout="wide")

st.title("🚀 MSME Early Warning System (EWS)")

# -------------------
# Upload business data
# -------------------
st.header("Upload Business Data")
st.markdown("Upload a CSV file containing your business data. Example columns: Date, Sales, CashFlow, Inventory")

uploaded_file = st.file_uploader("msme_sample_data", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['Date'])
    st.success("File uploaded successfully!")
    
    # Show data
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # -------------------
    # Dashboard
    # -------------------
    st.header("📊 Business Dashboard")
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Total Sales", f"${df['Sales'].sum():,.0f}")
    col2.metric("Total Cash Flow", f"${df['CashFlow'].sum():,.0f}")
    col3.metric("Average Inventory", f"{df['Inventory'].mean():,.0f} units")
    
    # Sales over time
    fig_sales = px.line(df, x='Date', y='Sales', title="Sales Over Time")
    st.plotly_chart(fig_sales, use_container_width=True)
    
    # Cash flow over time
    fig_cash = px.line(df, x='Date', y='CashFlow', title="Cash Flow Over Time", line_color='green')
    st.plotly_chart(fig_cash, use_container_width=True)
    
    # Inventory over time
    fig_inv = px.line(df, x='Date', y='Inventory', title="Inventory Over Time", line_color='orange')
    st.plotly_chart(fig_inv, use_container_width=True)
    
    # -------------------
    # Problem Detection
    # -------------------
    st.header("⚠ Early Warning Alerts")
    
    alerts = []
    
    # Simple rule-based alerts
    if df['Sales'].iloc[-1] < df['Sales'].mean() * 0.7:
        alerts.append("Sales dropped significantly! 🚨")
    if df['CashFlow'].iloc[-1] < df['CashFlow'].mean() * 0.5:
        alerts.append("Cash flow is critically low! 💰")
    if df['Inventory'].iloc[-1] > df['Inventory'].mean() * 1.5:
        alerts.append("Inventory is too high! ⚠")
    if df['Inventory'].iloc[-1] < df['Inventory'].mean() * 0.5:
        alerts.append("Inventory is too low! ⚠")
    
    if alerts:
        for alert in alerts:
            st.error(alert)
    else:
        st.success("All business parameters are stable ✅")
    
    # -------------------
    # Recommendations
    # -------------------
    st.header("💡 Recommendations")
    
    for alert in alerts:
        if "Sales dropped" in alert:
            st.info("Consider running promotions, marketing campaigns, or reviewing sales strategy.")
        if "Cash flow is critically low" in alert:
            st.info("Check receivables, reduce unnecessary expenses, or arrange short-term financing.")
        if "Inventory is too high" in alert:
            st.info("Consider discounts, bundle sales, or reduce new stock orders.")
        if "Inventory is too low" in alert:
            st.info("Restock essential items to avoid sales loss.")
    
    # -------------------
    # Export Reports
    # -------------------
    st.header("📄 Export Report")
    if st.button("Download CSV Report"):
        df.to_csv("msme_report.csv", index=False)
        st.success("Report saved as msme_report.csv")
        
else:
    st.info("Please upload a CSV file to continue.")
