import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import HumanMessage, SystemMessage
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import re
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DEFAULT_CATEGORIES = [
    "Housing", "Food", "Transportation", "Utilities",
    "Healthcare", "Entertainment", "Savings", "Investments",
    "Debt", "Other"
]

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "transactions" not in st.session_state:
    st.session_state.transactions = []
if "financial_goals" not in st.session_state:
    st.session_state.financial_goals = []

platform_rules = {"max_length": 4096}  # You can adjust this value
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

def init_groq_chain():
    """Initialize Groq model with LangChain"""
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-specdec",
        temperature=0.7,
        max_tokens=min(platform_rules['max_length'], 1024)
    )
        

def financial_analyst_chain():
    """Create financial analysis chain"""
    system_prompt = """You are a certified financial advisor with 20+ years experience.
        Provide detailed, actionable advice about personal finance management including:
        - Budgeting techniques
        - Investment strategies
        - Debt management
        - Savings plans
        - Expense optimization
        Always include numbers and percentages where appropriate.
        Break down complex concepts using simple analogies.
        Current date: {current_date}"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    return prompt | init_groq_chain() | StrOutputParser()

def financial_overview():
    """Display financial dashboard"""
    st.sidebar.header("Financial Overview")
    df = pd.DataFrame(st.session_state.transactions)
    
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df["type"] = df["amount"].apply(lambda x: "Income" if x > 0 else "Expense")
        
        current_month = datetime.now().month
        monthly_income = df[(df["type"] == "Income") & (df["date"].dt.month == current_month)]["amount"].sum()
        monthly_expenses = df[(df["type"] == "Expense") & (df["date"].dt.month == current_month)]["amount"].sum()
        savings_rate = ((monthly_income - monthly_expenses) / monthly_income * 100) if monthly_income > 0 else 0

        st.sidebar.metric("Monthly Income", f"${monthly_income:,.2f}")
        st.sidebar.metric("Monthly Expenses", f"${monthly_expenses:,.2f}")
        st.sidebar.metric("Savings Rate", f"{savings_rate:.1f}%")
        
        st.sidebar.subheader("Expense Categories")
        expense_df = df[df["type"] == "Expense"]
        if not expense_df.empty:
            fig = px.pie(expense_df, names="category", values="amount", title="Expense Breakdown")
            st.sidebar.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(page_title="AI Finance Assistant", layout="wide")
    st.title("AI Personal Finance Assistant")
    financial_overview()
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "analysis" in msg:
                    st.dataframe(msg["analysis"], hide_index=True)
        
        if prompt := st.chat_input("Ask financial question or add transaction..."):
            user_msg = {"role": "user", "content": prompt}
            st.session_state.messages.append(user_msg)
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.spinner("Analyzing and crafting response..."):
                chain = financial_analyst_chain()
                response = chain.invoke({
                    "input": prompt,
                    "current_date": datetime.now().strftime("%Y-%m-%d")
                })
            
            assistant_msg = {"role": "assistant", "content": response}
            st.session_state.messages.append(assistant_msg)
            
            with st.chat_message("assistant"):
                st.markdown(response)
    
    with col2:
        st.subheader("Financial Tools")
        
        with st.expander("Budget Planner"):
            selected_category = st.selectbox("Category", DEFAULT_CATEGORIES)
            budget_amount = st.number_input("Monthly Budget", min_value=0.0)
            if st.button("Set Budget"):
                st.success(f"Budget set: ${budget_amount} for {selected_category}")
        
        with st.expander("Savings Calculator"):
            initial = st.number_input("Initial Amount", value=1000.0)
            monthly = st.number_input("Monthly Contribution", value=500.0)
            years = st.slider("Years", 1, 30, 5)
            rate = st.slider("Annual Return (%)", 1.0, 15.0, 7.0)
            
            if st.button("Calculate"):
                months = years * 12
                monthly_rate = rate / 100 / 12
                future_value = initial * (1 + monthly_rate)**months
                future_value += monthly * ((1 + monthly_rate)**months - 1) / monthly_rate
                st.metric("Projected Value", f"${future_value:,.2f}")
        
        with st.expander("Debt Payoff Calculator"):
            debt_amount = st.number_input("Debt Balance", value=10000.0)
            interest_rate = st.number_input("Interest Rate (%)", value=7.0)
            monthly_payment = st.number_input("Monthly Payment", value=500.0)
            
            if debt_amount and interest_rate and monthly_payment:
                monthly_interest = interest_rate / 100 / 12
                months = -1 * (np.log(1 - (debt_amount * monthly_interest) / monthly_payment) / np.log(1 + monthly_interest))
                st.metric("Months to Payoff", f"{months:.1f} months")

if __name__ == "__main__":
    main()
