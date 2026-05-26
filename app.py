import streamlit as st
from src.predict import run_inference
from datetime import datetime

st.set_page_config(page_title="Fair Price Auto", page_icon="🚘", layout="centered")

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("🚘 FairPrice Auto Evaluator")
st.markdown("Enter a vehicle's basic details to calculate its fair market value and avoid getting ripped off.")
st.markdown("---")

# User-Friendly Inputs
col1, col2 = st.columns(2)

with col1:
    # Most common brands as a default list
    brand = st.selectbox("Make / Brand", ["Toyota", "Honda", "Ford", "Chevrolet", "BMW", "Mercedes-Benz", "Audi", "Hyundai", "Nissan", "Volkswagen"])
    
    # Limit years to realistic used-car ranges
    current_year = datetime.now().year
    year = st.selectbox("Manufacturing Year", list(range(current_year, 1999, -1)))
    
    fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid", "Electric"])

with col2:
    # Instead of an arbitrary slider, let them type the exact mileage or use a sensible slider
    mileage = st.number_input("Odometer (Miles/Km Driven)", min_value=0, max_value=300000, value=50000, step=1000)
    
    transmission = st.radio("Transmission", ["Automatic", "Manual"], horizontal=True)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("Calculate Market Value"):
    user_data = {
        'brand': brand,
        'year': year,
        'km_driven': mileage,
        'fuel': fuel,
        'transmission': transmission
    }
    
    predicted_price = run_inference(user_data)
    
    # Display the result like a premium financial tool
    st.success(f"### 💰 Estimated Fair Market Value\n# **${predicted_price:,.2f}**")
    
    st.info("💡 **Negotiation Tip:** If a dealer is asking for more than 10% above this price, ask them to justify the markup with service records or a certified warranty.")