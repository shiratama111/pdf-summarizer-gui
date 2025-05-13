import pandas as pd
import streamlit as st

from core.LGH import *
from core.VE_concrete import *


# def get_input_error_opti(params, **kwargs):
#     input_fields = dict(P=params.P, L=params.L, D=params.D, B=params.B, gc=params.gc, gs=params.gs,
#                         h=params.h, c=params.c, a_min=params.a_min, a_max=params.a_max, lb_rqd=params.lb_rqd)
#     violations = []
#     for key, value in input_fields.items():
#         if value is None:
#             violations.append(InputViolation("Type a value", fields=[key]))
#
#     if violations:
#         raise UserError("You must fix invalid field values before a result can be obtained.",
#                         input_violations=violations)


def get_mR_opti(params, ds_str, a, **kwargs):
    """
    get_mR_opti returnerer koordinatsæt til plot af momentbæreevnen mR.
    """

    [[rb, mR_max], [R1, mR_edge]] = get_mR(params)

    r2_lst = [-R1, -R1, -rb, rb, R1, R1]
    mR_lst = [0, mR_edge, mR_max, mR_max, mR_edge, 0]

    return [r2_lst, mR_lst]


def get_ds_a_opti(params, **kwargs):
    """
    Sorterer dataframe med alle brugbare løsninger efter mass og udvælger den letteste (første i liste)
    """
    no_solutions_as_min = False
    
    case_lst = get_case_lst(params)

    if not case_lst:
        ds_max = max(params.ds_lst)
        return ds_max, params.a_min, 0, 0, True, False, no_solutions_as_min

    df = pd.DataFrame(data=case_lst)

    df = df.sort_values(by=['As'])
    df = df.reset_index(drop=True)

    # Get the index of first solution with As_min_bool = True
    try:
        idx = df.loc[df['As_min_bool'], :].index[0]
    except IndexError:
        idx = 0
        no_solutions_as_min = True

    As_min_warning = False
    if idx != 0:
        As_min_warning = True

    # Remove As_min_bool from results
    df = df.drop(columns=['As_min_bool'])

    # Return the first solution and warning
    return *df.iloc[idx].values, False, As_min_warning, no_solutions_as_min


def get_case_lst(params, **kwargs):
    """
    Beregner momentbæreevnen af alle kombonationer af ds og a. Frasorterer alle design der ikke for tilstrækkelig styrke.
    Returnerer alle OK kombinationer af ds, a og As.
    """
    [r_lst, mt_lst, mr_lst] = get_m(params)
    mE = max(mt_lst)

    ds_array = params.ds_lst  # mm
    a_array = np.arange(params.a_min, params.a_max, params.a_step)  # mm

    if a_array.size == 0:
        a_array = [params.a_min]

    case_lst = []

    for ds in ds_array:

        params.ds = ds
        # Bestem minimumsarmering
        fctm = get_concrete_params(params.fc_str, params.gc)['fctm'] * np.power(10, 6)  # Pa
        fyk = get_rebar_params(params.YK_str, params.gs)['fyk'] * np.power(10, 6)       # Pa

        h_ = params.h / 1000     # m
        c_ = params.c / 1000     # m
        ds_ = ds / 1000          # m

        d_ = h_ - (c_ + ds_)     # m

        As_min = get_As_min(fctm, fyk, d_) * 1000  # mm2/mm


        for a in a_array:
            params.a = a

            # Calculate As
            As = math.pi / 4 * ds ** 2 / a  # mm2/mm

            # Tjek minimumsarmering
            As_min_bool = As >= As_min

            [[rb, mR_max], [R1, mR_edge]] = get_mR(params)

            if mR_max < mE:
                continue

            # Tjek:
            OK = True
            for r, mt in zip(r_lst, mt_lst):
                if r <= rb:
                    continue
                else:
                    mR = mR_max + (mR_edge - mR_max) / (R1 - rb) * (r - rb)
                    if round(mt) > round(mR):
                        OK = False
            if OK is False:
                continue

            case = {'ds': ds, 'a': a, 'As': As, 'As_min': As_min, 'As_min_bool': As_min_bool}
            case_lst.append(case)

    return case_lst


def get_mass_s_opti(params):
    rho_s = 7850              # kg/m3
    L = params.L_geo / 1000   # m

    h = params.h / 1000       # m
    ds = params.ds / 1000     # m
    a = params.a / 1000       # m

    As = math.pi / 4 * ds ** 2 / a          # m2/m

    vol_s = 2 * As * L
    mass_s = vol_s * rho_s

    return mass_s


def out_rebar_name(params, **kwargs):
    try:
        YK_str = params.YK_str
        ds = params.ds  # mm
        a = params.a  # mm
        #mass_s = get_mass_s_opti(params)   # kg
        rebar_name = f" {YK_str}{str(int(ds))}/{int(a)} us-br"
        return rebar_name
    except:
        return "N/A"