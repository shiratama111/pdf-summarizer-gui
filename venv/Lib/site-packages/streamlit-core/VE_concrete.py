from collections import namedtuple

import numpy as np
import streamlit as st

def get_concrete_params(fc_str, gc):
    """
    Calculate and return concrete parameters.

    Parameters:
    fc_str (str)  : Concrete strength class as a string.
    gc     (float): Safety factor for concrete.

    Returns:
    dict: A dictionary containing concrete parameters.
    """
    fck = float(fc_str[1:])
    fctm = 0.3 * fck ** (2 / 3)
    fctk = 0.7 * fctm
    fcd = fck / gc
    fctd = fctk / gc
    fbd = 2.25 * fctd

    return {'fck': fck, 'fctm': fctm, 'fctk': fctk, 'fcd': fcd, 'fctd': fctd, 'fbd': fbd}


def get_rebar_params(YK_str, gs):
    """
    Calculate and return rebar parameters.

    Parameters:
    YK_str (str)  : Rebar strength class as a string.
    gs     (float): Safety factor for steel.

    Returns:
    dict: A dictionary containing rebar parameters.
    """
    fyk = 500 if YK_str == 'K' else 550  # MPa
    fyd = fyk / gs

    return {'fyk': fyk, 'fyd': fyd}


def get_mRd_rc_plate(h, c, fcd, ds, a, fyd):
    """
    Calculate and return the moment capacity of a reinforced concrete plate.

    Parameters:
    h   (int)  : Total height of the concrete plate.
    c   (int)  : Cover to the reinforcement.
    fcd (float): Design value of concrete compressive strength.
    ds  (int)  : Diameter of the reinforcement.
    a   (int)  : Distance between the reinforcements.
    fyd (int)  : Design value of steel yield strength.

    Returns:
    float: The moment capacity of the reinforced concrete plate.
    """
    As = np.pi / 4 * ds ** 2 / a  # mm2/mm
    d = h - (c + ds)  # mm
    omega = As * fyd / (d * fcd)  # enhedsløs
    mu = omega * (1 - omega / 2)  # enhedsløs
    mR = mu * d ** 2 * fcd  # momentbæreevne, Nmm/mm
    mR = mR / 1000  # momentbæreevne, kNm/m

    return mR

def get_As_min(fctm, fyk, d):
    """
    Calculate and return the minimum reinforcement area.

    Parameters:
    fctm (float) [Pa]: Middel trækstyrke.
    fyk  (float) [Pa]: Krakteristisk flydespændning.
    d    (int)   [m] : Effektive højde.

    Returns:
    float: The minimum reinforcement area.
    """
    As_min = max(0.26 * fctm / fyk * d, 0.0013 * d)  # m2/m
    return As_min


PriceCO2 = namedtuple('PriceCO2', ['price', 'co2'])

@st.cache_data
def get_concrete_price_co2_dict():
    """
    Create and return a dictionary of PriceCO2 objects for different concrete strength classes and environmental classes.

    Returns:
    dict: A dictionary where the keys are tuples of concrete strength class and environmental class, and the values are PriceCO2 objects.
    """
    concrete_price_co2_dict = {
        # (Betonstyrke, Miljøklasse): (Pris [DKK/m3], CO2 [kgCO2eq/m3])
        ('C12', 'P'):  PriceCO2(1345, 175),
        ('C16', 'P'):  PriceCO2(1382, 205),
        ('C20', 'P'):  PriceCO2(1420, 220),
        ('C25', 'P'):  PriceCO2(1459, 245),
        ('C30', 'P'):  PriceCO2(1499, 275),
        ('C35', 'P'):  PriceCO2(1539, 305),
        ('C40', 'P'):  PriceCO2(1585, 335),
        ('C45', 'P'):  PriceCO2(1625, 360),
        ('C30', 'M'):  PriceCO2(1686, 290),
        ('C35', 'M'):  PriceCO2(1770, 315),
        ('C40', 'M'):  PriceCO2(1855, 345),
        ('C45', 'M'):  PriceCO2(1940, 370),
        ('C35', 'A'):  PriceCO2(1935, 370),
        ('C40', 'A'):  PriceCO2(2020, 395),
        ('C45', 'A'):  PriceCO2(2105, 420),
        ('C40', 'EA'): PriceCO2(2042, 440),
        ('C45', 'EA'): PriceCO2(2125, 465)
    }
    return concrete_price_co2_dict