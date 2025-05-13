import math

import numpy as np

from core.VE_concrete import *

# def get_input_error(params, **kwargs):
#     input_fields = dict(P=params.P, L=params.L, D=params.D, B=params.B, gc=params.gc, gs=params.gs,
#                         h=params.h, c=params.c, a=params.a, lb_rqd=params.lb_rqd)
#     violations = []
#     for key, value in input_fields.items():
#         if value is None:
#             violations.append(InputViolation("Type a value", fields=[key]))
#
#     # ds = float(params.ds_str[1:])
#     # if params.c > params.h/2 - ds:
#     #     violations.append(InputViolation("h >= 2*(c+ø_s) ikke overholdt", fields=["h", "c", "ds_str"]))
#
#     if violations:
#         raise UserError("You must fix invalid field values before a result can be obtained.",
#                         input_violations=violations)


def get_R1(params):
    L = params.L / 1000  # fundamentsbredde, m
    R1 = (L ** 2 / math.pi) ** .5

    return R1

def get_R0(params, **kwargs):
    if params.DB_str == 'Cirkulær':
        D = params.DB / 1000  # Diameter af søjle, m
        R0 = D / 2  # Radius af søjle, m
    else:
        B = params.DB / 1000  # Bredde af søjle, m
        R0 = (B ** 2 / math.pi) ** .5  # Radius af søjle, m

    return R0


def get_R(params, **kwargs):
    R0 = get_R0(params)  # Radius af søjle, m
    R1 = get_R1(params)  # Radius af fundament, m
    R = (R0 * R1 ** 2) ** (1 / 3)  # Radius ved overgang konstant til aftagende moment m_t

    return R


def get_m(params, **kwargs):
    P = params.P  # Søjlelast, kN

    R1 = get_R1(params)  # Radius af fundament, m
    R = get_R(params)  # Radius ved overgang konstant til aftagende moment m_t
    R0 = get_R0(params)  # Radius af søjle, m

    D = 2 * R0  # Diameter af søjle (Equivalent diameter ved kvadratisk søjle), m
    n_step = 100

    r_lst1 = list(np.linspace(0, R0, num=n_step, endpoint=False))
    r_lst2 = list(np.linspace(R0, R, num=n_step, endpoint=False))
    r_lst3 = list(np.linspace(R, R1, num=n_step))

    r_lst = r_lst1 + r_lst2 + r_lst3

    mt_lst = []
    mr_lst = []

    for r in r_lst1:
        mt = P / (2 * math.pi) * (1 - (D ** 2 / (4 * R1 ** 2)) ** (1 / 3))
        mr = P / (2 * math.pi) * (
                1 / 3 * (1 / (R1 ** 2) - 1 / ((0.5 * D) ** 2)) * r ** 2 + 1 - (D ** 2 / (4 * R1 ** 2)) ** (1 / 3))

        mt_lst.append(mt)
        mr_lst.append(mr)

    for r in r_lst2:
        mt = P / (2 * math.pi) * (1 - (D ** 2 / (4 * R1 ** 2)) ** (1 / 3))
        mr = P / (2 * math.pi) * \
             (1 / 3 * (r / R1) ** 2 - (D ** 2 / (4 * R1 ** 2)) ** (1 / 3) - (
                     1 / 3 * (R / R1) ** 2 - (D ** 2 / (4 * R1 ** 2)) ** (1 / 3)) * (R / r))

        mt_lst.append(mt)
        mr_lst.append(mr)

    for r in r_lst3:
        mt = P / (2 * math.pi) * (1 - (r / R1) ** 2)
        mr = 0

        mt_lst.append(mt)
        mr_lst.append(mr)

    return [r_lst, mt_lst, mr_lst]


def out_m_max(params, **kwargs):
    try:
        [r_lst, mt_lst, mr_lst] = get_m(params)
        m_max = max(mt_lst + mr_lst)
        m_max_round = round(m_max, 1)
    except:
        m_max_round = "N/A"

    return m_max_round


def get_mR(params, **kwargs):
    h = params.h  # mm
    c = params.c  # mm
    ds = params.ds  # mm
    a = params.a  # mm

    lb_rqd = params.lb / 1000  # Forankring af armering uden hensyntagen til opbuk, m
    R1 = get_R1(params)  # Radius af fundament, m

    # Materialeparametre
    fcd = get_concrete_params(params.fc_str, params.gc)['fcd']  # MPa
    fbd = get_concrete_params(params.fc_str, params.gc)['fbd']  # MPa
    fyk = get_rebar_params(params.YK_str, params.gs)['fyk']  # MPa
    fyd = get_rebar_params(params.YK_str, params.gs)['fyd']  # MPa

    # Fuld forankringslængde
    lb = fyk * ds / (4 * fbd)  # mm (forudsætter gode forankringsforhold (bunden af et fundament)).
    lb = lb / 1000  # m
    lb_1 = min(lb, lb_rqd)
    rb = max(min(R1 + lb_1 - lb, R1), 0)  # radius ud til fuld forankring start, m

    # Momentbæreevne plade
    mR = get_mRd_rc_plate(h, c, fcd, ds, a, fyd)  # kNm/m
    mR_edge = mR * lb_1 / lb
    mR_max = min(mR, mR * (R1 + lb_1) / lb)

    return [[rb, mR_max], [R1, mR_edge]]


def get_vE_vR(params, **kwargs):
    # Geometry
    h = params.h  # mm
    c = params.c  # mm
    DB = params.DB  # mm
    L = params.L  # mm
    P = params.P * 1000  # N
    gc = params.gc
    ds = params.ds  # mm
    a = params.a  # mm

    # Materialeparametre
    fck = get_concrete_params(params.fc_str, params.gc)['fck']  # MPa
    fcd = get_concrete_params(params.fc_str, params.gc)['fcd']  # MPa

    # as_min = max(0.26 * fctm / fyk * d, 0.0013 * d)   # mm2/mm
    # rho_min = as_min / d                              # mm2/mm2

    R1 = get_R1(params) * 1000  # mm
    As = np.pi / 4 * ds ** 2 / a  # mm2/mm

    d = h - c - ds  # mm
    Ac = L ** 2  # mm2
    rho = As / d  # mm2/mm2

    # Strengths
    ny = max(0.7 - fck / 200, 0.45)
    k = min(1 + (200 / d) ** .5, 2)

    vRd_min = 0.051 / gc * k ** 1.5 * fck ** .5
    vRd_c = max(vRd_min, 0.18 / gc * k * (100 * rho * fck) ** (1 / 3))
    vRd_max = .5 * ny * fcd

    if params.DB_str == 'Kvadratisk':
        a_max = min(get_eq_squircle_L1(L, DB), 2*d)
    else:
        a_max = min(R1 - DB / 2, 2*d)

    a_lst = np.linspace(0, a_max, 100)
    r_lst   = []
    vRd_lst = []
    vEd_lst = []

    for a in a_lst:

        if a == 0:
            vRd = vRd_max
        else:
            vRd = min(max(vRd_c * 2 * d / a, vRd_min), vRd_max)

        if params.DB_str == 'Cirkulær':
            r = (a + DB / 2) / 1000
            u = 2 * np.pi * (DB / 2 + a)
            A1 = np.pi * (DB / 2 + a) ** 2
        else:
            r = (a + DB / 2) / 1000
            u = 2 * np.pi * a + 4 * DB
            A1 = DB ** 2 + 4 * DB * a + np.pi * a ** 2

        VEd = P * (Ac - A1) / Ac
        vEd = VEd / d / u

        r_lst.append(r)
        vRd_lst.append(vRd)
        vEd_lst.append(vEd)

    return [r_lst, vEd_lst, vRd_lst]


def out_OK(params, **kwargs):
    try:
        [r_lst, mt_lst, mr_lst] = get_m(params)
        [r2_lst, mR_lst] = get_mR(params)
    except:
        return ""

    r1 = r2_lst[-3]
    mR1 = mR_lst[-3]
    r2 = r2_lst[-2]
    mR2 = mR_lst[-2]

    if mt_lst[0] > mR1:
        return "Bæreevne ikke OK!"

    for i, r in enumerate(r_lst):
        if r < r1:
            continue
        else:
            mR = mR1 + (mR2 - mR1) / (r2 - r1) * (r - r1)

            if round(mt_lst[i]) > round(mR):
                return "Bæreevne ikke OK!"

    return "Bæreevne OK!"


def vis_D(params, **kwargs):
    return params.DB_str == 'Cirkulær'


def vis_B(params, **kwargs):
    return params.DB_str == 'Kvadratisk'


def get_vol_c(params, **kwargs):
    L = params.L / 1000  # m
    h = params.h / 1000  # m
    vol_c = L ** 2 * h
    return vol_c


def get_mass_s(params, **kwargs):
    rho_s = 7850  # kg/m3
    L = params.L / 1000  # m
    h = params.h / 1000  # m
    ds = float(params.ds_str[1:]) / 1000  # m
    a = params.a / 1000  # m
    As = math.pi / 4 * ds ** 2 / a  # m2/m

    vol_s = 2 * As * L
    mass_s = vol_s * rho_s
    return mass_s


def out_mass_s(params, **kwargs):
    try:
        mass_s_round = round(get_mass_s(params), 1)
        return mass_s_round
    except:
        return "N/A"

def get_eq_squircle_L1(L, B):
    # Return the equivalent length L1 from column side with the same area as the square area
    # Squircle = Mix of square and circle

    # Formel kommer fra area af en squircle skal være lig med area af firkantet fundament
    L1 = (np.sqrt((L**2 - B**2) * np.pi + 4 * B**2) - 2 * B) / np.pi
    return L1