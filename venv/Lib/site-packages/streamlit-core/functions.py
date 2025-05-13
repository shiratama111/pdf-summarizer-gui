import math
import numpy as np
import shapely.geometry
import pandas as pd
import streamlit as st

from functools import cache
from scipy.optimize import fsolve, minimize_scalar, root_scalar
from shapely.ops import unary_union

def get_hori_rebar(params):
    elementtype = params.elementtype # [-] Søjle eller væg
    st_ash = params.st_ash           # [-] Forudstæning af tværarmering

    L_geo = params.L_geo  # [mm] faktisk søjlelængde

    h = params.h  # [mm] højde af tværsnit
    b = params.b  # [mm] bredde af tværsnit

    nc  = params.nc         # [-]  antal tryk-armeringsstænger
    dsc = params.dsc        # [mm] diameter tryk-armeringsstænger
    nt  = params.nt         # [-]  antal træk-armeringsstænger bs
    dst = params.dst        # [mm] diameter træk-armeringsstænger

    Asc = nc * (math.pi / 4 * dsc ** 2)  # [mm^2] tryk-armeringsareal
    Ast = nt * (math.pi / 4 * dst ** 2)  # [mm^2] træk-armeringsareal
    Ac = h * b  # [mm^2] beton tværsnitsareal

    # Tværarmering
    if st_ash == 'Minimum':  # Tværarmering som minimum
        if elementtype == 'Søjle':  # For søjler
            dist_h = min(20 * min(dsc, dst), min(b, h), 400)  # [mm] tværarmeringsafstand
            dsh = max(6, math.ceil(max(dsc, dst) / 4 / 2) * 2)  # [mm] diameter tværarmering

            ash0 = 2 * (math.pi / 4 * dsh ** 2) / dist_h  # [mm^2/mm] tværarmeringsareal søjle midt
            ash1 = 2 * (math.pi / 4 * dsh ** 2) / (0.6 * dist_h)  # [mm^2/mm] tværarmeringsareal søjle ender
            l_ender = 2 * max(b, h)
            ash = (ash0 * (L_geo - l_ender) + ash1 * l_ender) / L_geo  # [mm^2/mm] tværarmeringsareal søjle vægtet

        else:  # For vægge
            ash = max(0.25 * (Asc + Ast) / b, 0.001 * Ac / b)  # [mm^2/mm] tværarmeringsareal
            if ash < 0.500:  # [mm^2/mm]
                dsh = 6
            elif ash < 1.000:  # [mm^2/mm]
                dsh = 8
            elif ash < 1.500:  # [mm^2/mm]
                dsh = 10
            elif ash < 2.000:  # [mm^2/mm]
                dsh = 12
            else:
                dsh = 16
            dist_h = 2 * (math.pi / 4 * dsh ** 2) / ash  # [mm] tværarmeringsafstand

    else:  # Tværarmering som længdearmering
        ash = (Ast + Asc) / b  # [mm^2/mm] tværarmeringsareal
        if ash < 0.500:  # [mm^2/mm]
            dsh = 6
        elif ash < 1.000:  # [mm^2/mm]
            dsh = 8
        elif ash < 1.500:  # [mm^2/mm]
            dsh = 10
        elif ash < 2.000:  # [mm^2/mm]
            dsh = 12
        else:
            dsh = 16
        dist_h = 2 * (math.pi / 4 * dsh ** 2) / ash  # [mm] tværarmeringsafstand

    return dsh, dist_h, ash

def out_hori_rebar_st(params):
    dsh, dist_h, ash = get_hori_rebar(params)
    try:
        return f"Ø{dsh}/{round(dist_h)}"
    except:
        return "N/A"


def out_hori_rebar_st_ends(params):
    try:
        dsh, dist_h, ash = get_hori_rebar(params)

        if params.elementtype == 'Søjle' and params.st_ash == 'Minimum':
            return f"Ø{dsh}/{round(dist_h * 0.6)}"

        return "-"
    except:
        return "N/A"

def get_NM_data(params):
    user_error = None

    elementtype = params.elementtype
    st_ash = params.st_ash

    # Beton materialeparametre

    gc = params.gc  # partialkoefficient beton, enhedsløs

    fc_str = params.fc_str
    fck = float(fc_str[1:])  # kar.-styrke beton, MPa

    c_cost_dict = get_cost_dict(params.env, params)
    c_CO2_dict = get_CO2_dict(params.env, params)
    try:
        cost_c = c_cost_dict[fc_str]  # pris per m3 beton
        CO2_c = c_CO2_dict[fc_str]  # kg CO2eq per m3 beton
    except:
        user_error = 'Fejl: Vælg en tilladelig kombination af betonstyrke og miljøpåvirkning.'
        return {'user_error': user_error}

    # Stål materialeparametre

    gs = params.gs  # partialkoefficient stål, enhedsløs
    rho_s = 7850  # kg/m3

    if params.YK_str == 'K':  # K-stål
        fyk = 500  # kar.-styrke, MPa
        cost_s = params.cost_s_k  # Pris per kg stål
        CO2_s = params.CO2_s_k  # kg CO2eq per kg stål

    else:  # Y-stål
        fyk = 550  # kar.-styrke, MPa
        cost_s = params.cost_s_y  # Pris per kg stål
        CO2_s = params.CO2_s_y  # kg CO2eq per kg stål

    L_geo = params.L_geo  # faktisk søjlelængde, mm
    beta = params.beta  # betafaktor
    Ls_eff = L_geo * beta  # effektiv søjlelængde, mm

    h = params.h  # højde af tværsnit, mm
    b = params.b  # bredde af tværsnit, mm
    c = params.c  # afstand center armeringsstænger til kant, mm

    Nc = params.nc  # antal tryk-armeringsstænger, enhedsløs
    dsc = params.dsc  # diameter tryk-armeringsstænger, mm
    Nt = params.nt  # antal træk-armeringsstænger bs, enhedsløs
    dst = params.dst  # diameter træk-armeringsstænger, mm

    bool_cc = params.bool_cc
    bool_tc = params.bool_tc

    NEd_step = params.NEd_step * 1000  # steplængde mellem punkter på N-M diagram, N

    # Afhængige konstanter

    fcd = fck / gc  # regn.-styrke beton, MPa
    fyd = fyk / gs  # regn.-styrke stål, MPa
    fcm = fck + 8  # middelstyrke beton, MPa
    ec1 = 0.7 * fcm ** 0.31 / 1000  # tøjning i beton ved højeste spænding, enhedsløs
    ec3 = 0.00175  # trykbrudtøjning
    ecu3 = 0.0035  # trykbrudtøjning
    elcm = 22 * (fcm / 10) ** 0.3 * 1000  # sekant-elasticitetsmodul beton, MPa
    elcd = elcm / gc  # regn.-elasticitetsmodul beton, MPa

    els = 200000  # elasiticitetsmodul stål, MPa
    eyd = fyd / els  # regn. flydetøjning stål, enhedsløs

    Asc = Nc * (math.pi / 4 * dsc ** 2)  # tryk-armeringsareal, mm^2
    Ast = Nt * (math.pi / 4 * dst ** 2)  # træk-armeringsareal, mm^2
    Ac = h * b  # beton tværsnitsareal, mm^2

    dist_c = get_dist_c(params)  # tryk-armeringsafstand, mm
    dist_t = get_dist_t(params)  # træk-armeringsafstand, mm

    i = h / 6 * math.sqrt(3)
    lam = Ls_eff / i
    rho = (Asc + Ast) / Ac

    # Tværarmering
    dsh, dist_h, ash = get_hori_rebar(params)

    # Mængder, pris og CO2 summer
    vol_c = h * b * L_geo / 1e9  # m3
    mass_sct = (Asc + Ast) * L_geo / 1e9 * rho_s  # kg
    mass_sh = ash * L_geo * (b + h) / 1e9 * rho_s if elementtype == 'Søjle' else ash * L_geo * b / 1e9 * rho_s  # kg

    if elementtype == 'Søjle':
        sums = {'sum_c': vol_c, 'sum_sct': mass_sct, 'sum_sh': mass_sh}
        cost = {'cost_c': vol_c * cost_c,
                'cost_sct': mass_sct * cost_s, 'cost_sh': mass_sh * cost_s}  # DKK
        CO2 = {'CO2_c': vol_c * CO2_c,
               'CO2_sct': mass_sct * CO2_s, 'CO2_sh': mass_sh * CO2_s}  # kg
    else:
        sums = {'sum_c': vol_c * 1000 / b, 'sum_sct': mass_sct * 1000 / b, 'sum_sh': mass_sh * 1000 / b}
        cost = {'cost_c': vol_c * cost_c * 1000 / b,
                'cost_sct': mass_sct * cost_s * 1000 / b, 'cost_sh': mass_sh * cost_s * 1000 / b}  # DKK/m
        CO2 = {'CO2_c': vol_c * CO2_c * 1000 / b,
               'CO2_sct': mass_sct * CO2_s * 1000 / b, 'CO2_sh': mass_sh * CO2_s * 1000 / b}  # kg/m

    # Krybetal
    RH = params.RH
    t0 = params.t0
    p_mm = params.p_mm  # krybetal-korrektionsfaktor M_0Eqp / M_0Ed, enhedsløs
    h0 = 2 * b * h / (2 * (b + h))
    a1 = (35 / fcm) ** 0.7
    a2 = (35 / fcm) ** 0.2
    if fcm <= 35:
        prh = 1 + (1 - RH / 100) / (0.1 * h0 ** (1 / 3))
    else:
        prh = (1 + (1 - RH / 100) / (0.1 * h0 ** (1 / 3)) * a1) * a2
    bfcm = 16.8 / math.sqrt(fcm)
    bt0 = 1 / (0.1 + t0 ** 0.2)
    p0 = prh * bfcm * bt0  # slutkrybetal
    pef = p_mm * p0  # effektivt krybetal, enhedsløs

    # NM diagrammer fra funktioner

    x_gen, y_gen = get_NM_general(fcd, fyd, ec1, elcd, els, pef, h, b, c, Ls_eff, Asc, Ast,
                                  bool_cc, bool_tc, NEd_step)
    x_nom, y_nom = get_NM_nominel(fck, fcd, fyd, ec3, ecu3, eyd, elcd, els, pef, h, b, c, Ls_eff, Asc, Ast,
                                  bool_cc, bool_tc, NEd_step)

    # Polygoner

    poly_gen_points = [(0, 0)]
    for x, y in zip(x_gen, y_gen):
        poly_gen_points.append((x, y))
    poly_nom_points = [(0, 0)]
    for x, y in zip(x_nom, y_nom):
        poly_nom_points.append((x, y))

    poly_gen = shapely.geometry.Polygon(poly_gen_points)
    poly_nom = shapely.geometry.Polygon(poly_nom_points)
    try:
        poly_env = shapely.ops.unary_union([poly_gen, poly_nom])
    except:
        poly_env = poly_gen

    # Polygon as_min_søjle
    NEd_max = max(x_gen[-1], x_nom[-1])  # N
    MRd_max = max(max(y_gen), max(y_nom))  # Nmm

    # Minimumsarmering for søjler afhænging af ned:
    if elementtype == 'Søjle' and 0.1 * NEd_max / fyd > (Asc + Ast):  # NEd_max i mm
        NEd_asmin = (Asc + Ast) * fyd / 0.1  # N
        poly_asmin_rect_points = [(NEd_asmin, -MRd_max),
                                  (NEd_asmin, 2 * MRd_max),
                                  (2 * NEd_max, 2 * MRd_max),
                                  (2 * NEd_max, -MRd_max)]
        poly_asmin_rect = shapely.geometry.Polygon(poly_asmin_rect_points)

        # Opdeling af polygon af envelope
        poly_asmin = poly_env.intersection(poly_asmin_rect)
        poly_env = poly_env.difference(poly_asmin_rect)

        x_asmin_array, y_asmin_array = poly_asmin.exterior.coords.xy
        x_asmin, y_asmin = x_asmin_array.tolist(), y_asmin_array.tolist()

    else:
        x_asmin = None
        y_asmin = None

    x_env_array, y_env_array = poly_env.exterior.coords.xy
    x_env = x_env_array.tolist()
    y_env = y_env_array.tolist()

    # Konstruktive regler inkl minimumsarmering

    as_vmin = 0.002 * Ac
    as_vmax = 0.04 * Ac

    if as_vmin / 2 > Asc:
        user_error = f"Fejl: Trykarmering overholder ikke minimumsarmering: " \
                     f"As,c = {round(Asc)} mm2 < As,vmin / 2 = {round(as_vmin / 2)} mm2"
    elif as_vmin / 2 > Ast:
        user_error = f"Fejl: Trækarmering overholder ikke minimumsarmering: " \
                     f"As,t = {round(Ast)} mm2 < As,vmin / 2 = {round(as_vmin / 2)} mm2"
    elif as_vmax < Asc + Ast:
        user_error = f"Fejl: Armering overholder ikke maksimumsarmering: " \
                     f"As,tot = {round(Asc + Ast)} mm2 > As,vmax = {round(as_vmax)} mm2"
    elif x_nom[1] < 0:
        user_error = "Fejl: Tværsnit er overarmeret."

    elif elementtype == 'Søjle':
        if Nc % 1 != 0 or Nt % 1 != 0:
            user_error = "Fejl: For søjler bør antallet af armeringsstænger være hele tal."
        elif dsc < 8 or dst < 8:
            user_error = "Fejl: Længdearmeringsstænger i søjler skal være mindst ø8."
    #    elif Nc < 2 or Nt < 2:
    #    user_error = "Fejl: Søjler bør have mindst to stænger i hver side."

    elif elementtype == 'Væg':
        if max(dist_c, dist_t) > min(3 * h, 400):
            user_error = "Fejl: For vægge må armeringsafstanden ikke overstige 3 x h eller 400"

    return {'x_gen': x_gen, 'y_gen': y_gen, 'x_nom': x_nom, 'y_nom': y_nom,
            'x_env': x_env, 'y_env': y_env, 'x_asmin': x_asmin, 'y_asmin': y_asmin, 'user_error': user_error,
            'p0': p0, 'pef': pef, 'Asc': Asc, 'Ast': Ast, 'as_vmin': as_vmin, 'as_vmax': as_vmax,
            'dsh': dsh, 'rho': rho, 'lam': lam, 'Ls_eff': Ls_eff, 'sums': sums, 'cost': cost, 'CO2': CO2}

def calculate_Nac_Nat(fcd, fyd, ec1, elcd, els, pef, h, b, c, ls, Asc, Ast, bool_cc, bool_tc, NEd_step, x, e0):
    if bool_cc:  # Medregn asc i tryk?
        Nac = np.maximum(np.minimum((x - c) / x * (1 + pef) * e0 * Asc * els, Asc * fyd), -Asc * fyd)
    else:
        Nac = np.maximum(np.minimum((x - c) / x * (1 + pef) * e0 * Asc * els, 0), -Asc * fyd)
    if bool_tc:  # Medregn ast i tryk?
        Nat = np.maximum(np.minimum((h - x - c) / x * (1 + pef) * e0 * Ast * els, Ast * fyd), -Ast * fyd)
    else:
        Nat = np.maximum(np.minimum((h - x - c) / x * (1 + pef) * e0 * Ast * els, Ast * fyd), 0)
    return Nac, Nat

def get_NM_general(fcd, fyd, ec1, elcd, els, pef, h, b, c, ls, Asc, Ast, bool_cc, bool_tc, NEd_step):
    k = 1.05 * elcd * ec1 / fcd

    NEd = 0
    NEd_list = []
    M0Rd_list = []

    guess = np.ndarray((1, 1))
    guess[0, 0] = h / 4  # Gæt for nullinjehøjde

    # while-løkke giver ét N-M-punkt og opdaterer Ned.
    # break når M0Ed bliver negativ

    while True:
        def get_M0Rd_neg(e0):  # Funktion der giver momentbæreevnen ud fra kanttøjning e0.
            A = e0 / (k * ec1)
            B = (2 - k) * e0 / ec1

            def horizontal_equilibrium(x):
                # Funktion der beregner fejl i vandret ligevægt ud fra nullinjehøjden x.

                if x > 1e6:
                    zeta = 0.999
                else:
                    zeta = 0 if x <= h else (x - h) / x

                Nc_1 = 1 / 2 * k * e0 / ec1 * (1 - zeta ** 2 + (A - B) / (B ** 3) *
                      (B ** 2 * (1 - zeta ** 2) + 2 * B * (1 - zeta) + 2 * np.log((1 - B) / (1 - zeta * B))))

                Nc = Nc_1 * b * x * fcd

                Nac, Nat = calculate_Nac_Nat(fcd, fyd, ec1, elcd, els, pef, h, b, c,
                                             ls, Asc, Ast, bool_cc, bool_tc, NEd_step, x, e0)

                err = NEd + Nat - Nc - Nac
                return err

            root = fsolve(horizontal_equilibrium, guess)  # Roden af fejl-funktion findes.
            x = root[0]  # nullinjehøjden x bestemmes som den, der giver vandret ligevægt.

            # Når krumning går mod 0, går x mod uendelig og zeta mod 1. If-betingelse modgår numerisk støj i
            # udregningen af Nc_1, hvor der multipliceres med (1 - zeta ** 2), der går med 0.
            if x > 1e6:
                zeta = 0.999
            else:
                zeta = 0 if x <= h else (x - h) / x

            Nc_1 = 1 / 2 * k * e0 / ec1 * (
                    1 - zeta ** 2 + (A - B) / (B ** 3) * (
                        B ** 2 * (1 - zeta ** 2) + 2 * B * (1 - zeta) + 2 * np.log((1 - B) / (1 - zeta * B)))
            )

            Nc = Nc_1 * b * x * fcd

            Nc_2 = 1 / 3 * k * e0 / ec1 * (
                    1 - zeta ** 3 - (A - B) / (2 * B ** 4) *
                    (-2 * B ** 3 * (1 - zeta ** 3) - 3 * B ** 2 * (1 - zeta ** 2) - 6 * B * (1 - zeta) - 6 *
                     np.log((1 - B) / (1 - zeta * B)))
            )

            y_ = x * Nc_2 / Nc_1

            Nac, Nat = calculate_Nac_Nat(fcd, fyd, ec1, elcd, els, pef, h, b, c,
                                         ls, Asc, Ast, bool_cc, bool_tc, NEd_step, x, e0)

            #    print('NEd =', NEd, ', Nc =', Nc, ', Nac =', Nac, ', Nat =', Nat, ', err =', NEd + Nat - Nc - Nac, ', x =',
            #          x)

            # Aflever None istedet for -M0Rd hvis ligevægt ikke tilfredsstillende
            err = NEd + Nat - Nc - Nac
            if abs(err) > 1:
                return None

            MRd = (h / 2 - x + y_) * Nc + (h / 2 - c) * Nac + (h / 2 - c) * Nat
            M0Rd = MRd - NEd / 10 * (1 + pef) * e0 / x * ls ** 2

            return -M0Rd  # Momentbæreevnen returneres som negativ, da funktionen minimeres.

        # Momentbæreevnen maksimeres (ved at minimere resultatet af funktionen) over kanttøjningen e0.
        try:
            res = minimize_scalar(get_M0Rd_neg, bounds=(0.0001, 0.0035), method='bounded')
            M0Rd = -res.fun  # Maksimal momentbæreevne
            e0 = res.x  # Kanttøjning ved maksimal momentbæreevne
        except:  # Hvis minimize ikke virker (fordi alle løsninger er None pga manglende ligevægt) afsluttes kurven.
            NEd_final = NEd_list[-1]
            M0Rd_final = 0

            NEd_list.append(NEd_final)  # N
            M0Rd_list.append(M0Rd_final)  # Nmm
            break

        # while-løkken afbrydes hvis momentbæreevne er negativ
        if M0Rd < 0:
            # Lineær interpolation til M0Rd = 0
            NEd_prior = NEd_list[-1]
            M0Rd_prior = M0Rd_list[-1]
            NEd_final = NEd_prior + M0Rd_prior * (NEd - NEd_prior) / (M0Rd_prior - M0Rd)
            M0Rd_final = 0

            NEd_list.append(NEd_final)  # N
            M0Rd_list.append(M0Rd_final)  # Nmm
            break

        # N-M-koordinatsæt tilføjes lister
        #    print(NEd)
        #    print(M0Rd)
        NEd_list.append(NEd)  # N
        M0Rd_list.append(M0Rd)  # Nmm

        # Normalkraften opdateres før næste iteration
        NEd += NEd_step

    return [NEd_list, M0Rd_list]


def get_NM_nominel(fck, fcd, fyd, ec3, ecu3, eyd, elcd, els, pef, h, b, c, ls, Asc, Ast, bool_cc, bool_tc, NEd_step):
    d = h - c

    # sit B, NEd = 0

    # løsning af 2. gradsligning -eyd < esc < eyd
    a2 = 0.8 * b * fcd
    b2 = Asc * els * ecu3 - Ast * fyd
    c2 = - Asc * els * ecu3 * c

    d2 = b2 ** 2 - 4 * a2 * c2
    xB = (- b2 + math.sqrt(d2)) / (2 * a2)

    escB = ecu3 * (xB - c) / xB
    esB = ecu3 * (d - xB) / xB

    NEdB = 0
    MRdB = 0.8 * xB * b * fcd * (d - 0.5 * 0.8 * xB) + Asc * els * escB * (d - c)

    if escB < -eyd:  # trækflydning i asc
        xB = (Ast * fyd + Asc * fyd) / (0.8 * b * fcd)
        MRdB = 0.8 * xB * b * fcd * (d - 0.5 * 0.8 * xB) - Asc * fyd * (d - c)

    elif escB > eyd:  # trykflydning i asc
        xB = (Ast * fyd - Asc * fyd) / (0.8 * b * fcd)
        MRdB = 0.8 * xB * (d - 0.5 * 0.8 * xB) * b * fcd + Asc * fyd

    # Meregn ikke asc i tryk
    if (bool_cc is not True) and (escB > 0):
        xB = (Ast * fyd) / (0.8 * b * fcd)
        MRdB = 0.8 * xB * (d - 0.5 * 0.8 * xB) * b * fcd

    # sit C, balanceret (es = eyd)
    xC = ecu3 * d / (ecu3 + eyd)
    escC = ecu3 * (xC - c) / xC
    if escC > eyd:  # trykflydning i asc
        NEdC = 0.8 * xC * b * fcd + Asc * fyd - Ast * fyd
        MRdC = 0.8 * xC * b * fcd * (h / 2 - 0.5 * 0.8 * xC) + Asc * fyd * (h / 2 - c) + Ast * fyd * (d - h / 2)

    else:  # under trykflydning i asc
        NEdC = 0.8 * xC * b * fcd + Asc * els * escC - Ast * fyd
        MRdC = 0.8 * xC * b * fcd * (h / 2 - 0.5 * 0.8 * xC) + Asc * els * escC * (h / 2 - c) + Ast * fyd * (d - h / 2)

    # Meregn ikke asc i tryk
    if (bool_cc is not True) and (escC > 0):
        NEdC = 0.8 * xC * b * fcd - Ast * fyd
        MRdC = 0.8 * xC * b * fcd * (h / 2 - 0.5 * 0.8 * xC) + Ast * fyd * (d - h / 2)

    # sit D, es = 0
    xD = d
    NEdD = 0.8 * d * b * fcd + Asc * fyd
    MRdD = 0.8 * d * b * fcd * (h / 2 - 0.8 * d / 2) + Asc * fyd * (d - h / 2)

    # Meregn ikke asc i tryk
    if bool_cc is not True:
        NEdD = 0.8 * d * b * fcd
        MRdD = 0.8 * d * b * fcd * (h / 2 - 0.8 * d / 2)

    # sit E, esc = es = ec3
    NEdE = b * h * fcd + (Asc + Ast) * ec3 * els
    MRdE = (Asc - Ast) * ec3 * els * (h / 2 - c)

    # Meregn ikke asc og/eller ast i tryk
    if (bool_cc is not True) and (bool_tc is not True):
        NEdE = b * h * fcd
        MRdE = 0
    elif bool_cc is not True:
        NEdE = b * h * fcd + Ast * ec3 * els
        MRdE = - Ast * ec3 * els * (h / 2 - c)
    elif bool_tc is not True:
        NEdE = b * h * fcd + Asc * ec3 * els
        MRdE = Asc * ec3 * els * (h / 2 - c)

    NEd_list = []
    MRd_list = []

    # print('B', NEdB, MRdB)
    # print('C', NEdC, MRdC)
    # print('D', NEdD, MRdD)
    # print('E', NEdE, MRdE)

    # Ligevægte mellem sit B og C, trækflydning i ast
    n_step = max(math.ceil((NEdC - NEdB) / NEd_step), 2)
    x_rangeBC = np.linspace(xB, xC, num=n_step)
    for x in x_rangeBC:
        escBC = ecu3 * (x - c) / x

        if escBC < -eyd:  # trækflydning i asc
            NEd = 0.8 * x * b * fcd + Asc * (-fyd) - Ast * fyd
            MRd = 0.8 * x * b * fcd * (h / 2 - 0.5 * 0.8 * x) + (- Asc + Ast) * fyd * (d - h / 2)

        elif escBC < eyd:  # under trykflydning i asc
            NEd = 0.8 * x * b * fcd + Asc * els * escBC - Ast * fyd
            MRd = 0.8 * x * b * fcd * (h / 2 - 0.5 * 0.8 * x) + (Asc * escBC * els + Ast * fyd) * (d - h / 2)

        else:  # trykflydning i asc
            NEd = 0.8 * x * b * fcd + Asc * fyd - Ast * fyd
            MRd = 0.8 * x * b * fcd * (h / 2 - 0.5 * 0.8 * x) + (Asc + Ast) * fyd * (d - h / 2)

        # Meregn ikke asc i tryk
        if (bool_cc is not True) and (escBC > 0):
            NEd = 0.8 * x * b * fcd - Ast * fyd
            MRd = 0.8 * x * b * fcd * (h / 2 - 0.5 * 0.8 * x) + Ast * fyd * (d - h / 2)

        NEd_list.append(NEd)
        MRd_list.append(MRd)

    # Ligevægte mellem sit C og D, ikke flydning i ast
    n_step = max(math.ceil((NEdD - NEdC) / NEd_step), 2)
    x_rangeCD = np.linspace(xC, xD, num=n_step)
    for x in x_rangeCD[1:]:
        escCD = ecu3 * (x - c) / x
        esCD = ecu3 * (d - x) / x

        if escCD < eyd:  # ikke flydning i asc
            NEd = 0.8 * x * b * fcd + Asc * els * escCD - Ast * els * esCD
            MRd = 0.8 * x * b * fcd * (h / 2 - 0.5 * 0.8 * x) + (Asc * escCD * els + Ast * esCD * els) * (d - h / 2)

        else:  # trykflydning i asc
            NEd = 0.8 * x * b * fcd + Asc * fyd - Ast * els * esCD
            MRd = 0.8 * x * b * fcd * (h / 2 - 0.5 * 0.8 * x) + (Asc * fyd + Ast * esCD * els) * (d - h / 2)

        # Meregn ikke asc i tryk
        if (bool_cc is not True) and (escCD > 0):
            NEd = 0.8 * x * b * fcd - Ast * els * esCD
            MRd = 0.8 * x * b * fcd * (h / 2 - 0.5 * 0.8 * x) + Ast * esCD * els * (d - h / 2)

        NEd_list.append(NEd)
        MRd_list.append(MRd)

    # Ligevægte mellem sit D og E
    n_step = max(math.ceil((NEdE - NEdD) / NEd_step), 2)
    NEd_range = np.linspace(NEdD, NEdE, num=n_step)
    for NEd in NEd_range[1:]:  # lineær interpolation mellem D og E
        MRd = MRdD + (MRdE - MRdD) / (NEdE - NEdD) * (NEd - NEdD)
        NEd_list.append(NEd)
        MRd_list.append(MRd)

    #   print('NEd = ', NEd, ', MRd =', MRd)

    # Reduktion 2.orden nominel stivhed

    NEd_list0 = []
    M0Rd_list = []

    for NEd, MRd in zip(NEd_list, MRd_list):
        ac = h * b
        i = h / 6 * math.sqrt(3)

        k1 = math.sqrt(fck / 20)
        k2 = min(NEd / (ac * fcd) * ls / i / 170,
                 0.20)
        ks = 1
        kc = k1 * k2 / (1 + pef)

        iner_s = (Asc + Ast) * (h / 2 - c) ** 2
        iner_c = 1 / 12 * h ** 3 * b

        ei = kc * elcd * iner_c + ks * els * iner_s

        Nb = math.pi ** 2 * ei / (ls ** 2)

        #   print('k2 =', k2, ', ei =', ei, ', Nb =', Nb, ', Nb/NEd =', Nb/NEd)

        beta = math.pi ** 2 / 9.6  # parabolsk m-kurve
        beta = 1  ###### efter NemStatik ##########

        if NEd == 0:
            M0Rd = MRd
        else:
            M0Rd = MRd / (1 + beta / ((Nb / NEd) - 1))

        if NEd > Nb:
            NEd_final = NEd_list0[-1]
            M0Rd_final = 0

            NEd_list0.append(NEd_final)  # N
            M0Rd_list.append(M0Rd_final)  # Nmm
            break

        if M0Rd < 0:
            NEd_prior = NEd_list0[-1]
            M0Rd_prior = M0Rd_list[-1]
            NEd_final = NEd_prior + M0Rd_prior * (NEd - NEd_prior) / (M0Rd_prior - M0Rd)
            M0Rd_final = 0

            NEd_list0.append(NEd_final)  # N
            M0Rd_list.append(M0Rd_final)  # Nmm
            break

        NEd_list0.append(NEd)  # N
        M0Rd_list.append(M0Rd)  # Nmm

    if M0Rd_list[-1] > 0:
        NEd_list0.append(NEd_list0[-1])
        M0Rd_list.append(0)

    return [NEd_list0, M0Rd_list]

def get_param_dict(env, params, prefix):
    param_dict = {
        'Ekstra aggressiv': {'C40': getattr(params, f"{prefix}_c40e"), 'C45': getattr(params, f"{prefix}_c45e")},
        'Aggressiv':        {'C35': getattr(params, f"{prefix}_c35a"), 'C40': getattr(params, f"{prefix}_c40a"),
                             'C45': getattr(params, f"{prefix}_c45a")},
        'Moderat':          {'C30': getattr(params, f"{prefix}_c30m"), 'C35': getattr(params, f"{prefix}_c35m"),
                             'C40': getattr(params, f"{prefix}_c40m"), 'C45': getattr(params, f"{prefix}_c45m")},
        'Passiv':           {'C12': getattr(params, f"{prefix}_c12p"), 'C16': getattr(params, f"{prefix}_c16p"),
                             'C20': getattr(params, f"{prefix}_c20p"), 'C25': getattr(params, f"{prefix}_c25p"),
                             'C30': getattr(params, f"{prefix}_c30p"), 'C35': getattr(params, f"{prefix}_c35p"),
                             'C40': getattr(params, f"{prefix}_c40p"), 'C45': getattr(params, f"{prefix}_c45p")}
    }
    return param_dict.get(env, param_dict['Passiv'])

def get_cost_dict(env, params):
    return get_param_dict(env, params, 'cost')

def get_CO2_dict(env, params):
    return get_param_dict(env, params, 'CO2')


def get_loads(params):
    Ls = params.L_geo
    name = []
    NEd = []
    MEd = []

    ecc1 = params.ecc1 if params.ecc1 is not None else 0  # mm
    ecc0 = params.ecc0 if params.ecc0 is not None else 0  # mm
    ecc2 = params.ecc2 if params.ecc2 is not None else 0  # mm

    # Loop over each row in the table
    for row in params.tab.iterrows():
        # Define load
        name.append(row[1]['Navn'])
        N0   = row[1]['N0 [kN]'] * 1000  # N
        N1   = row[1]['N1 [kN]'] * 1000  # N
        N2   = row[1]['N2 [kN]'] * 1000  # N
        w    = row[1]['w [kN/m]'] # N/mm
        NEd.append(N1 + N0 + N2)  # N
        MEd.append(N1 * ecc1 + N0 * ecc0 - N2 * ecc2 + 1 / 8 * w * Ls ** 2)  # Nmm

    return {'name': name, 'x_load': NEd, 'y_load': MEd}


def get_dist_c(params):
    nc = params.nc
    b = params.b
    c = params.c
    elementtype = params.elementtype
    if elementtype == 'Søjle':
        dist_c = (b - 2 * c) / (nc - 1)
    else:
        dist_c = b / nc
    return dist_c


def get_dist_t(params):
    nt = params.nt
    b = params.b
    c = params.c
    elementtype = params.elementtype
    if elementtype == 'Søjle':
        dist_t = (b - 2 * c) / (nt - 1)
    else:
        dist_t = b / nt
    return dist_t

def get_data(params, list_sort='Pris', **kwargs):
    # Liste af løsninger til dataframe (df)
    case_list = get_case_list(params)
    if case_list is None:
        return None

    df = pd.DataFrame(data=case_list)

    # Sortering af løsninger efter CO2
    df = df.sort_values(by=['CO2eq [kg/m2]'])
    df.iloc[0, 4] = 'MIN.CO2: ' + df.iloc[0, 4]  # name
    df.iloc[0, 5] = 5  # point_size
    index_CO2 = df['CO2eq [kg/m2]'].min()

    # Sortering af løsninger efter pris
    df = df.sort_values(by=['Pris [DKK/m2]'])
    df.iloc[0, 4] = 'MIN.PRIS: ' + df.iloc[0, 4]  # name
    df.iloc[0, 5] = 5  # point_size
    index_cost = df['Pris [DKK/m2]'].min()

    if list_sort == 'CO2':  # Sorter efter pris eller CO2
        df = df.sort_values(by=['CO2eq [kg/m2]'])

    if params.bool_comp:  # Anvend sammenligningstværsnit eller ej
        case_comp = [get_case_comp(params)]
        df_comp = pd.DataFrame(data=case_comp)
        df = pd.concat([df_comp, df])
        df.iloc[0, 4] = 'SAMMENLIGN: ' + df.iloc[0, 4]  # name
        df.iloc[0, 5] = 5  # point_size

        if params.bool_index_comp:  #Gør sammenligningstværsnit til index 100
            index_CO2 = df_comp['CO2eq [kg/m2]'].min()
            index_cost = df_comp['Pris [DKK/m2]'].min()

    # Beregning af index
    df['CO2-index [-]'] = (df['CO2eq [kg/m2]'] * 100 / index_CO2).round(decimals=1)
    df['Prisindex [-]'] = (df['Pris [DKK/m2]'] * 100 / index_cost).round(decimals=1)

    return df


def get_case_list(params):
    # Fire nestede for-løkker: fck, t, ds, a.
    # Stålparametre defineres først udenfor løkker

    t_array = range(params.t_min, params.t_max + params.t_step, params.t_step)
    ds_array = [float(x[1:]) for x in params.ds_str]
    a_array = range(params.a_min, params.a_max + params.a_step, params.a_step)

    case_list = []

    # Stål materialeparametre
    if params.st_YK == 'K':
        fyk = 500
        cost_s = params.cost_s_k
        CO2_s = params.CO2_s_k
    else:
        fyk = 550
        cost_s = params.cost_s_y
        CO2_s = params.CO2_s_y
    fyd = fyk / params.gamma_s  # MPa
    ep_syd = fyd / 200000

    # Beton materialeparametre
    for fc_str in params.fc_str:
        fck = float(fc_str[1:])
        fcd = fck / params.gamma_c
        fctm = get_fctm(fc_str)

        if params.env == 'Ekstra aggressiv' and fck < 40 or \
                params.env == 'Aggressiv' and fck < 35 or \
                params.env == 'Moderat' and fck < 30:
            continue

        c_cost_dict = get_cost_dict(params.env, params)
        c_CO2_dict = get_CO2_dict(params.env, params)

        cost_c = c_cost_dict[fc_str]
        CO2_c = c_CO2_dict[fc_str]

        for t in t_array:
            for ds in ds_array:
                for a in reversed(a_array):

                    case = get_calculation(fc_str, fcd, fctm, params.st_YK, fyk, fyd, ep_syd, params.bool_st_c,
                                           params.bool_sx, t, ds, a, params.c_lag, cost_c, cost_s, CO2_c, CO2_s,
                                           params.total_area)

                    # Tester momentbæreevne, As_y (begge sider) overholder minimumsarmering samt beta_bal
                    if all([case['mRd [kNm]'] > params.med,
                            case['As_y [mm2/m]'] > case['As_y_min'],
                            case['As_y [mm2/m]'] < case['As_y_max'],
                            case['beta'] < 700 / (fyd + 700)]):
                        case_list.append(case)
                        if params.simple_option:
                            break
    if not case_list:
        return None
    else:
        return case_list


def get_case_comp(params):
    ds_comp = float(params.ds_str_comp[1:])
    fck = float(params.fc_str_comp[1:])
    fcd = fck / params.gamma_c
    fctm = get_fctm(params.fc_str_comp)

    # Stål materialeparametre
    if params.st_YK_comp == 'K':
        fyk = 500
        cost_s = params.cost_s_k
        CO2_s = params.CO2_s_k
    else:
        fyk = 550
        cost_s = params.cost_s_y
        CO2_s = params.CO2_s_y
    fyd = fyk / params.gamma_s  # MPa
    ep_syd = fyd / 200000

    c_cost_dict = get_cost_dict(params.env_comp, params)
    c_CO2_dict = get_CO2_dict(params.env_comp, params)

    cost_c_comp = c_cost_dict[params.fc_str_comp]
    CO2_c_comp = c_CO2_dict[params.fc_str_comp]

    case_comp = get_calculation(params.fc_str_comp, fcd, fctm, params.st_YK, fyk, fyd, ep_syd, params.bool_st_c,
                                params.bool_sx, params.t_comp, ds_comp, params.a_comp, params.c_lag, cost_c_comp,
                                cost_s, CO2_c_comp, CO2_s, params.total_area)
    return case_comp


def get_calculation(fc_str, fcd, fctm, st_yk, fyk, fyd, ep_syd, bool_st_c, bool_sx, t, ds, a, c_lag,
                    cost_c, cost_s, CO2_c, CO2_s, total_area):
    if bool_sx == 'Minimum':
        case_name = f"{st_yk}{int(ds)}/{a} bs, {fc_str} t={t}mm"
    else:
        case_name = f"{st_yk}{int(ds)}/{a} bs br, {fc_str} t={t}mm"

    el_s = 200000  # E-modul armeringsstål [N/mm2]
    ep_cu3 = 0.0035  # Beton brudtøjning for normalstyrkebeton fck =< 50 MPa [-]

    d = t - c_lag - ds / 2
    d0 = c_lag + ds / 2  # Afstand fra overkant til TP trykarmering [mm]

    as_y_min = max(0.26 * fctm / fyk * 1000 * d, 0.0013 * 1000 * d)
    as_y_max = 0.04 * 1000 * t
    as_y = math.floor(math.pi / 4 * ds ** 2 * 1000 / a * 2)  # bs
    as_t = as_y / 2  # Armeringsareal, trækarmering  [mm2]

    as_x_min = 0.20 * as_y
    if bool_sx == 'Minimum':
        as_x = math.floor(as_x_min)
    else:
        as_x = math.floor(as_y)

    # Medregnet trykarmering
    if bool_st_c:
        as_c = as_y / 2  # Armeringsarel, trykarmering [mm2]
    else:
        as_c = 0

    # Nulhøjden x findes vha. 2.gradsligning (virker for både med og uden trykarmering)
    a2 = 0.8 * 1000 * fcd
    b2 = as_c * el_s * ep_cu3 - as_t * fyd
    c2 = - as_c * el_s * ep_cu3 * d0
    d2 = b2 ** 2 - 4 * a2 * c2

    x = (- b2 + math.sqrt(d2)) / (2 * a2)
    ep_sc = ep_cu3 * (x - d0) / x

    # Flydning i trykarmering? ##NB BØR REGNE NY X!!
    if ep_sc < ep_syd:
        sigma_sc = ep_sc * el_s
    else:
        sigma_sc = fyd

    beta = x / d
    mrd = math.floor((0.8 * x * (d - 0.5 * 0.8 * x) * 1000 * fcd + as_c * sigma_sc * (d - d0)) / 1e6 * 10) / 10

    #    Gammel beregning uden trykarmering
    #    beta = as_y/2 * fyd / (0.8 * 1000 * d * fcd)
    #    mu = 0.8 * beta * (1 - 0.4 * beta)
    #    mrd = math.floor(mu * 1000 * d ** 2 * fcd / 1e6 * 10) / 10

    cost = round(t / 1000 * cost_c + (as_y + as_x) / 1e6 * 7850 * cost_s)
    cost_total = cost * total_area / 1000

    CO2 = round((t / 1000 * CO2_c + (as_y + as_x) / 1e6 * 7850 * CO2_s) * 10) / 10
    CO2_total = CO2 * total_area / 1000

    as_mass = round((as_y + as_x) / 1e6 * 7850 * 10) / 10

    return {'As_y [mm2/m]': as_y, 'As_x [mm2/m]': as_x,'t [mm]': t, 'mRd [kNm]': mrd, 'case_name': case_name,
            'point_size': 1, 'Pris [DKK/m2]': cost, 'Total pris [tDKK]': cost_total, 'Prisindex [-]': 100,
            'CO2eq [kg/m2]': CO2, 'Total CO2eq [ton]': CO2_total, 'CO2-index [-]': 100, 'As_y_min': as_y_min,
            'As_y_max': as_y_max, 'As_x_min': as_x_min, 'beta': beta, 'Stål [kg/m2]': as_mass}


def get_fctm(fc_str):
    fctm_dict = {'C12': 1.6, 'C16': 1.9, 'C20': 2.2, 'C25': 2.6, 'C30': 2.9, 'C35': 3.2, 'C40': 3.5, 'C45': 3.8}
    return fctm_dict[fc_str]