import numpy as np
import pandas as pd

def viterbi_potts(delta: np.ndarray, beta: float):
    """
    2-state Potts Viterbi using unary costs:
      u_i(0) = 0
      u_i(1) = delta[i]
    beta: switch penalty (>0)
    Returns: states (0/1, shape T), total_cost (float)
    """
    T = len(delta)
    # DP tables
    D0 = np.empty(T)  # best cost up to i ending in state 0
    D1 = np.empty(T)  # best cost up to i ending in state 1
    bp0 = np.zeros(T, dtype=np.int8)  # backpointer if end in 0: previous state
    bp1 = np.zeros(T, dtype=np.int8)  # backpointer if end in 1: previous state

    # init (i=0)
    D0[0] = -delta[0]
    D1[0] =  delta[0]

    # recurrence
    for i in range(1, T):
        # end in 0
        stay0 = D0[i-1]
        switch10 = D1[i-1] + beta
        if stay0 <= switch10:  # tie-break: prefer staying
            D0[i] = -delta[i] + stay0
            bp0[i] = 0
        else:
            D0[i] = -delta[i] + switch10
            bp0[i] = 1

        # end in 1
        stay1 = D1[i-1]
        switch01 = D0[i-1] + beta
        if stay1 <= switch01:
            D1[i] = delta[i] + stay1
            bp1[i] = 1
        else:
            D1[i] = delta[i] + switch01
            bp1[i] = 0

    # termination
    if D0[-1] <= D1[-1]:
        last_state = 0
        total_cost = D0[-1]
    else:
        last_state = 1
        total_cost = D1[-1]

    # backtrack
    states = np.empty(T, dtype=np.int8)
    states[-1] = last_state
    for i in range(T-1, 0, -1):
        states[i-1] = bp0[i] if states[i] == 0 else bp1[i]

    return states, float(total_cost)

def segments_from_states(states: np.ndarray):
    """Return list of (start, end_inclusive, state)."""
    segs = []
    start = 0
    cur = states[0]
    for i in range(1, len(states)):
        if states[i] != cur:
            segs.append((start, i-1, int(cur)))
            start = i
            cur = states[i]
    segs.append((start, len(states)-1, int(cur)))
    return segs

def grid_search_beta(delta: np.ndarray, betas):
    """
    Try multiple betas; pick the one with minimal total objective.
    betas: iterable of positive floats
    Returns: best_beta, best_states, best_cost, results (list of dicts)
    """
    results = []
    best = None
    for b in betas:
        states, cost = viterbi_potts(delta, b)
        k = len(segments_from_states(states))
        results.append({"beta": b, "cost": cost, "n_segments": k})
        if best is None or cost < best[2]:
            best = (b, states, cost)
    best_beta, best_states, best_cost = best
    return best_beta, best_states, best_cost, results

def potts_segmentation(bin_ids: np.ndarray, snp_info: pd.DataFrame, costs: np.ndarray):
    """
    for each region, perform Potts segmentation
    return a 1D binary state array
    """
    region_ids = snp_info["region_id"].unique()
    snp_info_grps_reg = snp_info.groupby(by="region_id", sort=False)
    potts_states = np.zeros((len(bin_ids),), dtype=np.int8)
    for region_id in region_ids:
        snp_reg = snp_info_grps_reg.get_group(region_id)
        snp_bin_ids = snp_reg["bin_id"].unique()
        costs_reg = costs[snp_bin_ids]
        betas = np.log(len(costs_reg)) * np.array([0.1, 0.5, 1, 2, 3, 4])
        best_beta, states, cost, summary = grid_search_beta(costs_reg, betas)
        potts_states[snp_bin_ids] = states
        # print(best_beta, states, cost, summary)    
    return potts_states

def potts_baf_refinement(potts_states: np.ndarray, mixture_bafs: np.ndarray, nomix_bafs: np.ndarray):
    return np.choose(potts_states, [nomix_bafs, mixture_bafs])
