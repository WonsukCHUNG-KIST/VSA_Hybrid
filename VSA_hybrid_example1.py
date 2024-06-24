import numpy as np
import pandas as pd
import VSA_hybrid as VSA


# Sample #1
zeolite13X_param = [
    5.4363,     # qmax_CO2
    8.5697,     # b_CO2
    19570.32,   # Hads_CO2
    2.1133,     # qmax_N2
    0.1527,     # b_N2
    14456.38,   # Hads_N2
]
yfeed = 0.13    # x_feed_CO2

if __name__ == '__main__':
    Pad, Pbl, Pev, W_cycle, W_tot, C_cap, purity, capture_rate, W_cap, Sel, SSE, Feasibility, time, VSA_model = VSA.VSA_hybrid(
        zeolite13X_param,
        Pfeed=1.,
        yfeed=0.13,
        Pad_value=5.,
        Pbl_value=1.,
        Pev_value=0.1,
        objtype='SSE',
        solver_type='IPOPT',
        tee=True,
        solverstate_print=True,
        print_summary=True
    )

