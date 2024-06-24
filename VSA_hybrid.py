import numpy as np
import pyomo.environ as pym


beta = np.array([
    [-1.24168921043209, 1.57588700131619, 0.813849438352230],
    [1.05263769861203, -0.463532692903032, 0.382397230795551],
    [-0.817681287065412, 0.0354324787302467, 0.726442852565983],
    [1.25829330439883, .317164188328447, -2.19495436117440],
    [-1.34135783064578, 0.175461559921245, 1.16333796232150],
    [-0.479971050113724, 0.356607934203232, 1.10846378066590],
    [0.844493543071416, -0.402644403394955, -1.17989310813660],
    [2.25862476162250, -2.16282442575187, -3.97403867395309],
    [-1.09836016215794, 0.135013082843976, 1.80137617841760],
    [0.968206919599024, -0.307160584209149, -2.88455756698917],
    [2.35208533783818, -0.0472375064594466, -0.304399095808333]
])

VSA_hybrid_init_str =\
"******************************************************\n\
  Hybrid model for vacuum Sswing adsorption process\n\
  Chung W., Kim J., Jung H., Lee J. H., 2024\n\
*******************************************************\n"

def VSA_hybrid(
        adsorbent,
        yfeed=0.1,
        Pfeed=1,
        beta=beta,
        Pad_value=None,
        Pbl_value=None,
        Pev_value=None,
        Tfeed=300,
        SSE_tol=0.01,
        elec_price=21.9,
        purity_constr=None,
        purity_min=False,
        caprate_constr=None,
        caprate_min=False,
        objtype='cost',
        solver_type='IPOPT',
        print_summary=True,
        tee=False,
        solverstate_print=False):

    '''
    :param adsorbent: [qCO2, bCO2, HadsCO2, qN2, bN2, HadsN2, yfeed]
    :param yfeed: Feed CO2 mole fraction, default value = 0.1
    :param beta: hyperparameter defined in Eq. (54)
    :param Pfeed: Feed pressure (bar)
    :param Pad_value: Fixed adsorption pressure (bar) if specified
    :param Pbl_value: Fixed blowdown pressure (bar) if specified
    :param Pev_value: Fixed evacuation pressure (bar) if specified
    :param Tfeed: Feed temperature (K)
    :param SSE_tol: Tolerance for sum of squared error. See Eq. (56)
    :param elec_price: Electricity price ($/GJ)
    :param purity_constr: Constraint for purity
    :param purity_min: If True, purity > purity_constraint
                       Otherwise purity_constraint - 0.005 < purity < purity_constraint + 0.005
    :param caprate_min:
    :param caprate_min: Constraint for capture rate
                        Otherwise caprate_constraint - 0.005 < caprate < caprate_constraint + 0.005
    :param objtype: If SSE, objective function is SSE (see Eq. (56)
                    If cost, objective function is capture cost per tonCO2 (see Eq. (57)
    :param solver_type: Solver for NLP.
                        IPOPT as default, BARON in GAMS environment,
                        user_specified solver as  pyomo.environ.SolverFactory
    :param print_summary: If True, print the detailed results
    :param tee: If True, print the solver optimization log
    :param solverstate_print: If true, print the solver state
    :return: [
        Pad,
        Pbl,
        Pev,
        W_cycle, total compression work per one cycle (J)
        W_tot, total capture energy (kJ/molCO2)
        C_cap, total capture cost ($/tonCO2)
        purity,
        capture_rate,
        W_cap, working capacity, see Eq. (44)
        Sel, selectivity, see Eq. (45)
        SSE, sum of squared error, see Eqs. (30) - (32), (56)
        Feasibility, False if the solver cannot find a solution
        time, Total computational time (sec)
        VSA_model, object for hybrid mode as pyomo.environ.ConcreteModel
        ]
    '''

    print(VSA_hybrid_init_str)

    # qCO2, bCO2, HadsCO2
    # qN2, bN2, HadsN2
    # xCO2, Pad, Pbl, Pev
    qm1 = float(adsorbent[0])
    b1 = float(adsorbent[1])
    Hads1 = float(adsorbent[2])
    qm2 = float(adsorbent[3])
    b2 = float(adsorbent[4])
    Hads2 =float( adsorbent[5])
    yfeed = yfeed

    R = 8.31447
    Tref = 298.15

    L = 1
    R_bed = 1 / np.sqrt(3.141592)
    V = L * 3.141592 * R_bed**2

    epsi = 0.322
    rho_s = 650  # density of solid adsorbent, kg/cum
    Cp_s = 1070  # solid heat capacity, J/kg K

    t_ann = 8000
    t_cycle = 0.2
    N_cycle_ann = t_ann / t_cycle
    C_ads = 0
    C_comp = 1.2 / (3600 * 8000)  # [$/W]
    C_elec = elec_price * 10**(-9)
    e_OPEX = 1.29
    e_TCI = 0.1693125
    e_YD = 0.1

    Patm = 1
    Ptr = 120

    # Hybrid model construction
    model = pym.ConcreteModel()

    qmax = 30
    model.q1ad0 = pym.Var(bounds=(0, qmax), initialize=4.140976569348023)
    model.q1ad1 = pym.Var(bounds=(0, qmax), initialize=3.650776206258698)
    model.q2ad0 = pym.Var(bounds=(0, qmax), initialize=0.20517900658412672)
    model.q2ad1 = pym.Var(bounds=(0, qmax), initialize=0.209434929792392)
    model.q1bl0 = pym.Var(bounds=(0, qmax), initialize=4.196263922785893)
    model.q1bl1 = pym.Var(bounds=(0, qmax), initialize=3.5670706425244676)
    model.q2bl0 = pym.Var(bounds=(0, qmax), initialize=0.1485046609820035)
    model.q2bl1 = pym.Var(bounds=(0, qmax), initialize=0.14829712585020652)
    model.q1ev0 = pym.Var(bounds=(0, qmax), initialize=3.2043921263458865)
    model.q1ev1 = pym.Var(bounds=(0, qmax), initialize=2.4552006036778193)
    model.q2ev0 = pym.Var(bounds=(0, qmax), initialize=0)
    model.q2ev1 = pym.Var(bounds=(0, qmax), initialize=0)

    model.errorq1ad0 = pym.Var(bounds=(-1, 1), initialize=0)
    model.errorq1ad1 = pym.Var(bounds=(-1, 1), initialize=0)
    model.errorq2ad0 = pym.Var(bounds=(-1, 1), initialize=0)
    model.errorq2ad1 = pym.Var(bounds=(-1, 1), initialize=0)
    model.errorq1bl0 = pym.Var(bounds=(-1, 1), initialize=0)
    model.errorq1bl1 = pym.Var(bounds=(-1, 1), initialize=0)
    model.errorq2bl0 = pym.Var(bounds=(-1, 1), initialize=0)
    model.errorq2bl1 = pym.Var(bounds=(-1, 1), initialize=0)
    model.errorq1ev0 = pym.Var(bounds=(-1, 1), initialize=0)
    model.errorq1ev1 = pym.Var(bounds=(-1, 1), initialize=0)
    model.errorq2ev0 = pym.Var(bounds=(-1, 1), initialize=0)
    model.errorq2ev1 = pym.Var(bounds=(-1, 1), initialize=0)

    model.n1ad = pym.Var(bounds=(0, 1e4), initialize=1716.91272410466)
    model.n2ad = pym.Var(bounds=(0, 1e4), initialize=91.3601808988655)
    model.n1bl = pym.Var(bounds=(0, 1e4), initialize=1710.65077145958)
    model.n2bl = pym.Var(bounds=(0, 1e4), initialize=65.4002737200622)
    model.n1ev = pym.Var(bounds=(0, 1e4), initialize=1247.09125806314)
    model.n2ev = pym.Var(bounds=(0, 1e4), initialize=0)

    model.dTev = pym.Var(bounds=(0, 100), initialize=19.902250363784486)
    model.dTad = pym.Var(bounds=(0, 100), initialize=24.696704863047337)
    model.dTbl = pym.Var(bounds=(0, 100), initialize=1.0557435408357752)

    model.Fin = pym.Var(bounds=(1e-3, 1e5), initialize=5886.01680457296)
    model.Fout_ad_ev = pym.Var(bounds=(1e-3, 1e5), initialize=5324.83515763824)
    model.Fout_bl_ad = pym.Var(bounds=(1e-3, 1e5), initialize=32.2218598238832)
    model.Fout_ev_bl = pym.Var(bounds=(1e-3, 1e5), initialize=528.959787110836)

    model.yev = pym.Var(bounds=(0, 1), initialize=1)
    model.yout_bl_ad = pym.Var(bounds=(0, 1), initialize=0.1943386470998627)
    model.ybl = pym.Var(bounds=(0, 1), initialize=0.43661184361619476)
    model.yout_ev_bl = pym.Var(bounds=(0, 1), initialize=0.876360594311875)
    model.yad_avg = pym.Var(bounds=(0, 1), initialize=0.05546851857542331)

    model.Pad = pym.Var(bounds=(1,10), initialize=5)
    model.Pbl = pym.Var(bounds=(0.1,10), initialize=1)
    model.Pev = pym.Var(bounds=(0.001,10), initialize=0.1)
    Pad = model.Pad
    Pbl = model.Pbl
    Pev = model.Pev

    if Pad_value is not None:
        model.Pad_constr_value = pym.Constraint(expr=Pad == Pad_value)
    else:
        model.Pad_constr = pym.Constraint(expr=Pad >= Pfeed)

    if Pbl_value is not None:
        model.Pbl_constr_value = pym.Constraint(expr=Pbl == Pbl_value)
    else:
        model.Pbl_constr1 = pym.Constraint(expr=pym.log(Pad / Pbl) / np.log(10) >= 0.1)
        model.Pbl_constr2 = pym.Constraint(expr=pym.log(Pad / Pbl) / np.log(10) <= 1)

    if Pev_value is not None:
        model.Pev_constr_value = pym.Constraint(expr=Pev == Pev_value)
    else:
        model.Pev_constr1 = pym.Constraint(expr=pym.log(Pbl/Pev)/np.log(10) >= 0.5)
        model.Pev_constr2 = pym.Constraint(expr=pym.log(Pbl/Pev)/np.log(10) <= 2)

    model.W_cycle = pym.Var(within=pym.NonNegativeReals)
    model.C_AVC = pym.Var(within=pym.NonNegativeReals)

    # Langmuir isotherm
    model.q1ad0_constr = pym.Constraint(
        expr=model.q1ad0 + model.errorq1ad0 ==
             qm1*b1*Pad*yfeed * pym.exp(Hads1/R * (1/(Tfeed)-1/Tref)) \
             / (1 + b1*Pad*yfeed * pym.exp(Hads1/R * (1/(Tfeed)-1/Tref))
                + b2*Pad*(1-yfeed) * pym.exp(Hads2/R * (1/(Tfeed) - 1/Tref))))
    model.q1ad1_constr = pym.Constraint(
        expr=model.q1ad1 + model.errorq1ad1 ==
             qm1*b1*Pad*yfeed * pym.exp(Hads1/R * (1/(Tfeed+model.dTad)-1/Tref)) \
            / (1 + b1*Pad*yfeed * pym.exp(Hads1/R * (1/(Tfeed+model.dTad)-1/Tref))
               + b2*Pad*(1-yfeed) * pym.exp(Hads2/R * (1/(Tfeed+model.dTad)-1/Tref))))
    model.q2ad0_constr = pym.Constraint(
        expr=model.q2ad0 + model.errorq2ad0 ==
             qm2*b2*Pad*(1-yfeed) * pym.exp(Hads2/R * (1/(Tfeed)-1/Tref)) \
             / (1 + b1*Pad*yfeed * pym.exp(Hads1/R * (1/(Tfeed)-1/Tref))
                + b2*Pad*(1-yfeed) * pym.exp(Hads2/R * (1/(Tfeed)-1/Tref))))
    model.q2ad1_constr = pym.Constraint(
        expr=model.q2ad1 + model.errorq2ad1 ==
             qm2*b2*Pad*(1-yfeed) * pym.exp(Hads2/R * (1/(Tfeed+model.dTad)-1/Tref)) \
             / (1 + b1*Pad*yfeed * pym.exp(Hads1/R * (1/(Tfeed+model.dTad)-1/Tref))
                + b2*Pad*(1-yfeed) * pym.exp(Hads2/R * (1/(Tfeed+model.dTad)-1/Tref))))

    model.q1bl0_constr = pym.Constraint(
        expr=model.q1bl0 + model.errorq1bl0 ==
             qm1*b1*Pbl*model.ybl * pym.exp(Hads1/R * (1/(Tfeed-model.dTbl)-1/Tref)) \
             / (1 + b1*Pbl*model.ybl * pym.exp(Hads1/R * (1/(Tfeed-model.dTbl)-1/Tref))
                + b2*Pbl*(1-model.ybl) * pym.exp(Hads2/R * (1/(Tfeed-model.dTbl)-1/Tref))))
    model.q1bl1_constr = pym.Constraint(
        expr=model.q1bl1 + model.errorq1bl1 ==
             qm1*b1*Pbl*model.ybl * pym.exp(Hads1/R * (1/(Tfeed+model.dTad-model.dTbl)-1/Tref)) \
             / (1 + b1*Pbl*model.ybl * pym.exp(Hads1/R * (1/(Tfeed+model.dTad-model.dTbl)-1/Tref))
                + b2*Pbl*(1-model.ybl) * pym.exp(Hads2/R * (1/(Tfeed+model.dTad-model.dTbl)-1/Tref))))
    model.q2bl0_constr = pym.Constraint(
        expr=model.q2bl0 + model.errorq2bl0 ==
             qm2*b2*Pbl*(1-model.ybl) * pym.exp(Hads2/R * (1/(Tfeed-model.dTbl)-1/Tref)) \
             / (1 + b1*Pbl*model.ybl * pym.exp(Hads1/R * (1/(Tfeed-model.dTbl)-1/Tref))
                + b2*Pbl*(1-model.ybl) * pym.exp(Hads2/R * (1/(Tfeed-model.dTbl)-1/Tref))))
    model.q2bl1_constr = pym.Constraint(
        expr=model.q2bl1 + model.errorq2bl1 ==
             qm2*b2*Pbl*(1-model.ybl) * pym.exp(Hads2/R * (1/(Tfeed+model.dTad-model.dTbl)-1/Tref)) \
             / (1 + b1*Pbl*model.ybl * pym.exp(Hads1/R * (1/(Tfeed+model.dTad-model.dTbl)-1/Tref))
                + b2*Pbl*(1-model.ybl) * pym.exp(Hads2/R * (1/(Tfeed+model.dTad-model.dTbl)-1/Tref))))
    #
    model.q1ev0_constr = pym.Constraint(
        expr=model.q1ev0 + model.errorq1ev0 ==
             qm1*b1*Pev*model.yev * pym.exp(Hads1/R * (1/(Tfeed-model.dTev)-1/Tref)) \
             / (1 + b1*Pev*model.yev * pym.exp(Hads1/R * (1/(Tfeed-model.dTev)-1/Tref))
                + b2*Pev*(1-model.yev) * pym.exp(Hads2/R * (1/(Tfeed-model.dTev)-1/Tref))))
    model.q1ev1_constr = pym.Constraint(
        expr=model.q1ev1 + model.errorq1ev1 ==
             qm1*b1*Pev*model.yev * pym.exp(Hads1/R * (1/(Tfeed)-1/Tref)) \
             / (1 + b1*Pev*model.yev * pym.exp(Hads1/R * (1/(Tfeed)-1/Tref))
                + b2*Pev*(1-model.yev) * pym.exp(Hads2/R * (1/(Tfeed)-1/Tref))))
    model.q2ev0_constr = pym.Constraint(
        expr=model.q2ev0 + model.errorq2ev0 ==
             qm2*b2*Pev*(1-model.yev) * pym.exp(Hads2/R * (1/(Tfeed-model.dTev)-1/Tref)) \
             / (1 + b1*Pev*model.yev * pym.exp(Hads1/R * (1/(Tfeed-model.dTev)-1/Tref))
                + b2*Pev*(1-model.yev) * pym.exp(Hads2/R * (1/(Tfeed-model.dTev)-1/Tref))))
    model.q2ev1_constr = pym.Constraint(
        expr=model.q2ev1 + model.errorq2ev1 ==
             qm2*b2*Pev*(1-model.yev) * pym.exp(Hads2/R * (1/(Tfeed)-1/Tref)) \
             / (1 + b1*Pev*model.yev * pym.exp(Hads1/R * (1/(Tfeed)-1/Tref))
                + b2*Pev*(1-model.yev) * pym.exp(Hads2/R * (1/(Tfeed)-1/Tref))))

    # gas uptake profile
    model.n1ad_constr = pym.Constraint(expr=model.n1ad == (1-epsi)*rho_s * 0.5*(model.q1ad0 + model.q1ad1))
    model.n2ad_constr = pym.Constraint(expr=model.n2ad == (1-epsi)*rho_s * 0.5*(model.q2ad0 + model.q2ad1))
    model.n1bl_constr = pym.Constraint(expr=model.n1bl == (1-epsi)*rho_s * 0.5*(model.q1bl0 + model.q1bl1))
    model.n2bl_constr = pym.Constraint(expr=model.n2bl == (1-epsi)*rho_s * 0.5*(model.q2bl0 + model.q2bl1))
    model.n1ev_constr = pym.Constraint(expr=model.n1ev == (1-epsi)*rho_s * 0.5*(model.q1ev0 + model.q1ev1))
    model.n2ev_constr = pym.Constraint(expr=model.n2ev == (1-epsi)*rho_s * 0.5*(model.q2ev0 + model.q2ev1))

    # temperature assumption
    model.dTev_constr = pym.Constraint(expr=model.dTev == (Hads1*(model.q1ad0-model.q1ev0) + Hads2*(model.q2ad0-model.q2ev0)) / Cp_s)
    model.dTad_constr = pym.Constraint(expr=model.dTad == (Hads1*(model.q1ad1-model.q1ev1) + Hads2*(model.q2ad1-model.q2ev1)) / Cp_s)
    model.dTbl_constr = pym.Constraint(
        expr=model.dTbl == (Hads1*(model.q1ad0-model.q1bl0) + Hads2*(model.q2ad0-model.q2bl0)) / 2/Cp_s + \
             (Hads1*(model.q1ad1-model.q1bl1) + Hads2*(model.q2ad1-model.q2bl1)) / 2/Cp_s)

    # mass balance
    model.Fad1_constr = pym.Constraint(expr=model.Fin*yfeed - model.Fout_ad_ev*model.yad_avg == -model.n1ev + model.n1ad)
    model.Fad2_constr = pym.Constraint(expr=model.Fin*(1-yfeed) - model.Fout_ad_ev*(1-model.yad_avg) == -model.n2ev + model.n2ad)
    model.Fbl1_constr = pym.Constraint(expr=-model.Fout_bl_ad*model.yout_bl_ad == -model.n1ad + model.n1bl)
    model.Fbl2_constr = pym.Constraint(expr=-model.Fout_bl_ad*(1-model.yout_bl_ad) == -model.n2ad + model.n2bl)
    model.Fev1_constr = pym.Constraint(expr=-model.Fout_ev_bl*model.yout_ev_bl == -model.n1bl + model.n1ev)
    model.Fev2_constr = pym.Constraint(expr=-model.Fout_ev_bl*(1-model.yout_ev_bl) == -model.n2bl + model.n2ev)

    # Surrogate model
    beta_input = [
        np.log(qm1),
        np.log(qm1/qm2),
        np.log(qm1*b1),
        np.log(qm2*b2),
        np.log(Hads1),
        np.log(Hads1/Hads2),
        np.log(yfeed),
        pym.log(Pad)/np.log(10),
        pym.log(Pad/Pbl)/np.log(10),
        pym.log(Pbl/Pev)/np.log(10)
    ]
    eMax = [3, 2, 6, 0, 11, 1, -0.22314, 1, 1, 2]
    eMin = [-1, -0.5, -1, -3, 9, 0, -3.50656, 0, 0.1, 0.5]
    beta_inm = []
    for i in range(10):
        beta_inm.append((beta_input[i]-eMin[i])/(eMax[i]-eMin[i]))
    phi1 = beta[0,0]
    phi2 = beta[0,1]
    phi3 = beta[0,2]
    for i in range(10):
        phi1 += beta_inm[i]*beta[i+1,0]
        phi2 += beta_inm[i]*beta[i+1,1]
        phi3 += beta_inm[i]*beta[i+1,2]
    a = pym.exp(phi1) / (1+pym.exp(phi1))
    b = pym.exp(phi2) / (1+pym.exp(phi2))
    c = pym.exp(phi3) / (1+pym.exp(phi3))
    model.yad_avg_constr = pym.Constraint(expr=model.yad_avg == a*(Pev/Pad*model.yev) + (1-a)*(yfeed))
    model.ybl1_constr = pym.Constraint(expr=model.ybl >= model.yout_bl_ad)
    model.ybl2_constr = pym.Constraint(expr=model.yad_avg <= model.yout_bl_ad)
    model.ybl3_constr = pym.Constraint(expr=model.yout_bl_ad == b*model.yad_avg + (1-b)*model.ybl)
    model.yev1_constr = pym.Constraint(expr=model.yev >= model.yout_ev_bl)
    model.yev2_constr = pym.Constraint(expr=model.ybl <= model.yout_ev_bl)
    model.yev3_constr = pym.Constraint(expr=model.yout_ev_bl == c*model.yout_bl_ad + (1-c)*model.yev)

    E_beta = np.array([
        0.518568556975217,
        - 0.0466952608810258,
        - 0.0349187312453107,
        0.213323408596075,
        - 0.0157832267614640,
        - 0.0348459473554172,
        0.0282098765270605,
        - 0.0358930244142880,
        0.0579823791606745,
        - 0.0804482390075651,
        - 0.164168082963339
    ])

    E_surrogate = E_beta[0]
    for i in range(10):
        E_surrogate += E_beta[i+1] * beta_inm[i]

    # Energy consumption and process evaluation
    W_feed = 1/0.8*R*Tfeed * model.Fin * 10/3 * ((Pad/Pfeed)**0.230679 - 1)
    W_cycle = 1/0.6*R*Tfeed * model.Fout_ev_bl*10/3 * ((Pbl/Pev)**0.230679 - 1) * E_surrogate\
              + 1/0.6*R*Tfeed * 0.5*model.Fout_bl_ad*10/3*((Patm/Pbl)**0.230679-1) * (pym.exp(-10*(Pbl-Patm)) / (1 + pym.exp(-10*(Pbl-Patm))))
    W_comp = 1/0.8*R*Tfeed * model.Fout_ev_bl * 10/3 * ((Ptr/Pbl)**0.230679 - 1)
    W_ann = (W_feed + W_cycle + W_comp) * V * N_cycle_ann
    TCI = 5.68 * 1.1 * (C_ads * V + C_comp * W_ann)
    OPEX = e_OPEX * C_elec * W_ann + 5.68 * 1.1 * (e_TCI) * (C_ads * V + C_comp * W_ann)
    TPC = e_OPEX * C_elec * W_ann + 5.68 * 1.1 * (e_TCI + e_YD) * (C_ads * V + C_comp * W_ann)
    F_ann = float(V) * model.Fout_ev_bl * model.yout_ev_bl * N_cycle_ann
    model.C_AVC_constr = pym.Constraint(expr=model.C_AVC*F_ann == TPC)
    model.W_cycle_constr = pym.Constraint(expr=model.W_cycle*F_ann == W_ann * 1e-3)

    # Objective function and constraint
    SSE = model.errorq1ad0**2 + model.errorq1ad1**2 + model.errorq2ad0**2 + model.errorq2ad1**2 + \
          model.errorq1bl0**2 + model.errorq1bl1**2 + model.errorq2bl0**2 + model.errorq2bl1**2 + \
          model.errorq1ev0**2 + model.errorq1ev1**2 + model.errorq2ev0**2 + model.errorq2ev1**2

    model.F_min = pym.Constraint(expr=model.Fin >= 0.1)
    if purity_constr is not None:
        if purity_min:
            model.purity_constr = pym.Constraint(expr=model.yout_ev_bl >= purity_constr)
        else:
            model.purity_constr = pym.Constraint(expr=model.yout_ev_bl >= purity_constr-0.005)
            model.purity_constr2 = pym.Constraint(expr=model.yout_ev_bl <= purity_constr+0.005)
    if caprate_constr is not None:
        if caprate_min:
            model.caprate_constr = pym.Constraint(expr=model.Fout_ev_bl * model.yout_ev_bl >= caprate_constr * model.Fin * yfeed)
        else:
            model.caprate_constr = pym.Constraint(expr=model.Fout_ev_bl*model.yout_ev_bl >= (caprate_constr-0.005) * model.Fin*yfeed)
            model.caprate_constr2 = pym.Constraint(expr=model.Fout_ev_bl*model.yout_ev_bl <= (caprate_constr+0.005) * model.Fin*yfeed)
    if objtype == 'cost':
        model.obj = pym.Objective(expr=model.C_AVC, sense=pym.minimize)
        model.SSE_constr = pym.Constraint(expr=SSE <= SSE_tol)
    else:
        model.obj = pym.Objective(expr=SSE, sense=pym.minimize)

    # Solve the problem
    Feasible = True

    if solver_type=='IPOPT' or solver_type=='ipopt':
        solver = pym.SolverFactory('ipopt', executable='Ipopt-3.14.16-win64-msvs2019-md/bin/ipopt.exe')
        solverstate = solver.solve(model, tee=tee)  # IPOPT
    elif solver_type=='BARON' or solver_type=='baron':
        solver = pym.SolverFactory('gams', solver='baron')
        option_BARON = ['GAMS_MODEL.optfile = 1;','$onecho > baron.opt', 'MaxTime 5', '$offecho']
        solverstate = solver.solve(model, tee=tee, add_options=option_BARON) # BARON
    else:
        solver = solver_type
        solverstate = solver.solve(model, tee=tee)
    solver_massage = solverstate['Solver'][0]['Termination condition']

    if solverstate_print:
        print(solverstate)

    if solver_type == 'IPOPT' or solver_type == 'ipopt':
        solvtime = solverstate.solver[0]['Time']
    else:
        solvtime = solverstate.solver[0]['User time'] # BARON

    if solver_massage == 'infeasible' or  solver_massage == 'maxTimeLimit' or solver_massage == 'error':
        Feasible = False
    if pym.value(model.Fout_ev_bl) == 0:
        Feasible = False

    if Feasible:
        if print_summary:
            ''' print start '''
            print('\t*** Variable summary ***')
            print(f"Pad:\t\t\t\t{model.Pad.value:.2f}\t\t\tbar")
            print(f"Plb:\t\t\t\t{model.Pbl.value:.2f}\t\t\tbar")
            print(f"Pev:\t\t\t\t{model.Pev.value:.2f}\t\t\tbar")
            print(f"Fin:\t\t\t\t{model.Fin.value:.2f}\t\t\tmol")
            print(f"Fad:\t\t\t\t{model.Fout_ad_ev.value:.2f}\t\t\tmol")
            print(f"Fbl:\t\t\t\t{model.Fout_bl_ad.value:.2f}\t\t\tmol")
            print(f"Fev:\t\t\t\t{model.Fout_ev_bl.value:.2f}\t\t\tmol")
            print(f"yad_avg:\t\t\t{pym.value(model.yad_avg):.4f}")
            print(f"ybl_avg:\t\t\t{pym.value(model.yout_bl_ad):.4f}")
            print(f"yev_avg:\t\t\t{pym.value(model.yout_ev_bl):.4f}")
            print(f"ybl:\t\t\t\t{pym.value(model.ybl):.4f}")
            print(f"yev:\t\t\t\t{pym.value(model.yev):.4f}")
            print(f"qCO2_ad (z=0, 1):\t[{model.q1ad0.value:.3f}\t{model.q1ad1.value:.3f}]\tmol/kg")
            print(f"qN2_ad  (z=0, 1):\t[{model.q2ad0.value:.3f}\t{model.q2ad1.value:.3f}]\tmol/kg")
            print(f"qCO2_bl (z=0, 1):\t[{model.q1bl0.value:.3f}\t{model.q1bl1.value:.3f}]\tmol/kg")
            print(f"qN2_bl  (z=0, 1):\t[{model.q2bl0.value:.3f}\t{model.q2bl1.value:.3f}]\tmol/kg")
            print(f"qCO2_ev (z=0, 1):\t[{model.q1ev0.value:.3f}\t{model.q1ev1.value:.3f}]\tmol/kg")
            print(f"qN2_ev  (z=0, 1):\t[{model.q2ev0.value:.3f}\t{model.q2ev1.value:.3f}]\tmol/kg")
            print(f"nCO2_ad:\t\t\t{model.n1ad.value:.2f}\t\t\tmol")
            print(f"nN2_ad :\t\t\t{model.n2ad.value:.2f}\t\t\tmol")
            print(f"nCO2_bl:\t\t\t{model.n1bl.value:.2f}\t\t\tmol")
            print(f"nN2_bl :\t\t\t{model.n2bl.value:.2f}\t\t\tmol")
            print(f"nCO2_ev:\t\t\t{model.n1ev.value:.2f}\t\t\tmol")
            print(f"nN2_ev :\t\t\t{model.n2ev.value:.2f}\t\t\tmol")
            print(f"Tad (z=0, 1):\t\t[{Tfeed:.1f}\t{Tfeed + model.dTad.value:.1f}]\tK")
            print(f"Tbl (z=0, 1):\t\t[{Tfeed - model.dTbl.value:.1f}\t{Tfeed + model.dTad.value - model.dTbl.value:.1f}]\tK")
            print(f"Tev (z=0, 1):\t\t[{Tfeed - model.dTev.value:.1f}\t{Tfeed:.1f}]\tK")
            print(f"model_error (\u03A3\u0394q):\t{pym.value(SSE):.3f}\t\t\tmol/kg")

            print('')
            print('\t*** Process evaluation summary ***')
            print(f"Electricity consumption: (cycle)\t{pym.value(W_cycle):.0f}\tJ/cycle")
            print(f"                         (specific)\t{pym.value(W_ann / F_ann * 1e-3):.2f}\tkJ/mol CO2")
            print(f"Capture energy (feed compression):\t{pym.value((W_feed) / 1000 / (model.Fout_ev_bl * model.yout_ev_bl)):.2f}\tkJ/molCO2")
            print(f"               (cycle):\t\t\t\t{pym.value((W_cycle) / 1000 / (model.Fout_ev_bl * model.yout_ev_bl)):.2f}\tkJ/molCO2")
            print(f"               (CO2 compression):\t{pym.value((W_comp) / 1000 / (model.Fout_ev_bl * model.yout_ev_bl)):.2f}\tkJ/molCO2")
            print(f"Annual CO2 capture amount:\t\t\t{pym.value(F_ann * 0.044 / 1000):.2f}\ttonCO2/yr bed")
            print(f"Capital expenditure:\t\t\t\t{pym.value(TCI):.0f}\t$/bed")
            print(f"Total production cost:\t\t\t\t{pym.value(TPC):.0f}\t$/bed")
            print(f"CO2 capture cost:\t\t\t\t\t{pym.value(pym.value(model.C_AVC) / 0.044 * 1000):.2f}\t$/tonCO2")
            print(f"Electricity cost:\t\t\t\t\t{pym.value(e_OPEX * C_elec * W_ann / F_ann / 0.044 * 1000):.2f}\t$/tonCO2")
            print(f"OPEX:\t\t\t\t\t\t\t\t{pym.value(OPEX / F_ann / 0.044 * 1000):.2f}\t$/tonCO2")
            print(f"CO2 capture rate:\t\t\t\t\t{pym.value((model.Fout_ev_bl * model.yout_ev_bl) / (model.Fin * yfeed)):.3f}")
            print(f"CO2 purity:\t\t\t\t\t\t\t{model.yout_ev_bl.value:.3f}")
            ''' print end '''

        return (Pad.value, Pbl.value, Pev.value,
            pym.value(W_cycle/1000/(model.Fout_ev_bl*model.yout_ev_bl)),  # Work_cycle [J]
            pym.value((W_feed+W_cycle+W_comp)/1000/(model.Fout_ev_bl*model.yout_ev_bl)),  # Work_total [kJ/molCO2]
            pym.value(model.C_AVC/0.044*1000),  # CO2 capture cost [$/tonCO2]
            pym.value(model.yout_ev_bl),  # CO2 purity
            pym.value((model.Fout_ev_bl * model.yout_ev_bl) / (model.Fin * yfeed)),  # CO2 capture rate
            pym.value(model.q1ad0+model.q1ad1)/2 - pym.value(model.q1ev0+model.q1ev1)/2,  # Working capacity
            pym.value(model.q1ad0+model.q1ad1)/pym.value(model.q2ad0+model.q2ad1),  # Selectivity
            pym.value(SSE),  # Sum of squared error
            Feasible,  # Feasibility result
            solvtime,  # Computational time
            model
        )
    else:
        return (0,0,0,
                0,0,0,
                0,0,0,0,0, Feasible, -1, model)
