from collections import namedtuple
import streamlit as st

PriceCO2 = namedtuple('PriceCO2', ['price', 'co2'])

@st.cache_data
def get_steel_price_co2_dict():
    """
    Create and return a dictionary of PriceCO2 objects for different rebar types.

    Returns:
    dict: A dictionary where the keys are rebar type and the values are PriceCO2 objects.
    """
    steel_price_co2_dict = {
        # (Armeringstype): (Pris [DKK/kg], CO2 [kgCO2eq/m3])
        'Y':  PriceCO2(15, 0.45),
        'K':  PriceCO2(14, 0.45),
    }
    return steel_price_co2_dict