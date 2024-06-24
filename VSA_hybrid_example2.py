import numpy as np
import pandas as pd
import VSA_hybrid as VSA


# Optimization of VSA process
# Constraints: purity >= 0.8, capture rate >= 0.5
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
        purity_constr=0.8,
        purity_min=True,
        caprate_constr=0.5,
        caprate_min=True,
        objtype='cost',
        solver_type='IPOPT',
        tee=True,
        solverstate_print=True,
        print_summary=True
    )

