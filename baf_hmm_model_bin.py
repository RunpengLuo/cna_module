import numpy as np
from scipy.special import logsumexp
from scipy.stats import binom
from numba import njit
# from scipy.special import expit

# def build_sitewise_transmat(
#     hairs: np.ndarray, 
#     prior_phase: np.ndarray,
#     prior_phaseset: np.ndarray,
#     prior_unphased: np.ndarray, 
#     prior_conf=0.9, # prior confidence for phase
#     lam=0.3, # prior weight, 0 = Trust hairs only
#     alpha=1.0, # pseudocount for hairs
#     log=True,
#     tol=10e-6
# ):
#     # first row is invalid
#     sitewise_transmat = np.zeros_like(hairs, dtype=np.float64)

#     for i in range(1, len(sitewise_transmat)):
#         c00, c01, c10, c11 = hairs[i]
#         cis = c00 + c11
#         trans = c01 + c10

#         hair_cis = (cis + alpha) / (cis + trans + 2 * alpha)
#         # prior-based bias
#         prior_ignored = False
#         prior_cis = 0.0
#         if prior_unphased[i - 1] or prior_unphased[i]:
#             prior_ignored = True
#         else:
#             if prior_phaseset[i - 1] != prior_phaseset[i]:
#                 prior_ignored = True
#         if not prior_ignored:
#             if prior_phase[i - 1] == prior_phase[i]:
#                 prior_cis = prior_conf
#             else:
#                 prior_cis = 1 - prior_conf
#         p_cis = ((1 - lam) * hair_cis + lam * prior_cis) / 2
#         p_trans = (1 - p_cis) / 2
#         sitewise_transmat[i, 0] = p_cis   # 00
#         sitewise_transmat[i, 1] = p_trans # 01
#         sitewise_transmat[i, 2] = p_trans # 10
#         sitewise_transmat[i, 3] = p_cis   # 11
#     if log:
#         sitewise_transmat = np.log(np.clip(sitewise_transmat, tol, 1 - tol))
#     return sitewise_transmat

# new one, fixed
def build_sitewise_transmat(
    hairs: np.ndarray, 
    # prior_phase: np.ndarray,
    # prior_phaseset: np.ndarray,
    # prior_unphased: np.ndarray, 
    # prior_conf=0.9, # prior confidence for phase
    alpha=1.0, # pseudocount for hairs
    beta=5.0, # pseudocount for prior-phasing
    log=True,
    tol=10e-6,
):
    # first row is invalid
    sitewise_transmat = np.zeros_like(hairs, dtype=np.float64)

    for i in range(1, len(sitewise_transmat)):
        c00, c01, c10, c11 = hairs[i]
        cis = c00 + c11
        trans = c01 + c10

        # prior-based bias
        prior_ignored = True
        prior_cis = 0
        prior_trans = 0
        # if not prior_unphased[i - 1] and not prior_unphased[i]:
        #     if prior_phaseset[i - 1] == prior_phaseset[i]:
        #         prior_ignored = False
        # if not prior_ignored:
        #     if prior_phase[i - 1] == prior_phase[i]:
        #         prior_cis = prior_conf
        #     else:
        #         prior_cis = 1 - prior_conf
        #     prior_trans = 1 - prior_cis

        total_cis = (cis + alpha + prior_cis * beta)
        total_trans = (trans + alpha + prior_trans * beta)
        p_cis = total_cis / (total_cis + total_trans)
        p_trans = total_trans / (total_cis + total_trans)
        sitewise_transmat[i, 0] = p_cis   # 00
        sitewise_transmat[i, 1] = p_trans # 01
        sitewise_transmat[i, 2] = p_trans # 10
        sitewise_transmat[i, 3] = p_cis   # 11
    if log:
        sitewise_transmat = np.log(np.clip(sitewise_transmat, tol, 1 - tol))
    return sitewise_transmat

@njit(cache=True, fastmath=True)
def forward_backward_numba(nobs, log_emissions, log_transmat, log_startprob, log_alpha, log_beta):
    # Forward pass
    log_alpha[0, 0] = log_startprob[0] + log_emissions[0, 0]
    log_alpha[0, 1] = log_startprob[1] + log_emissions[0, 1]
    for obs in range(1, nobs):
        # log_alpha[obs, 0]
        tmp0 = log_alpha[obs - 1, 0] + log_transmat[obs, 0]
        tmp1 = log_alpha[obs - 1, 1] + log_transmat[obs, 2]
        m = tmp0 if tmp0 > tmp1 else tmp1
        log_alpha[obs, 0] = log_emissions[obs, 0] + (m + np.log(np.exp(tmp0 - m) + np.exp(tmp1 - m)))

        # log_alpha[obs, 1]
        tmp0 = log_alpha[obs - 1, 0] + log_transmat[obs, 1]
        tmp1 = log_alpha[obs - 1, 1] + log_transmat[obs, 3]
        m = tmp0 if tmp0 > tmp1 else tmp1
        log_alpha[obs, 1] = log_emissions[obs, 1] + (m + np.log(np.exp(tmp0 - m) + np.exp(tmp1 - m)))

    # Backward pass
    log_beta[nobs - 1, 0] = 0.0
    log_beta[nobs - 1, 1] = 0.0
    for obs in range(nobs - 2, -1, -1):
        # beta for state 0
        tmp0 = log_beta[obs + 1, 0] + log_transmat[obs, 0] + log_emissions[obs + 1, 0]
        tmp1 = log_beta[obs + 1, 1] + log_transmat[obs, 1] + log_emissions[obs + 1, 1]
        m = tmp0 if tmp0 > tmp1 else tmp1
        log_beta[obs, 0] = m + np.log(np.exp(tmp0 - m) + np.exp(tmp1 - m))

        # beta for state 1
        tmp0 = log_beta[obs + 1, 0] + log_transmat[obs, 2] + log_emissions[obs + 1, 0]
        tmp1 = log_beta[obs + 1, 1] + log_transmat[obs, 3] + log_emissions[obs + 1, 1]
        m = tmp0 if tmp0 > tmp1 else tmp1
        log_beta[obs, 1] = m + np.log(np.exp(tmp0 - m) + np.exp(tmp1 - m))

    return log_alpha, log_beta

def compute_log_alpha_beta(
    nobs: int,
    log_emissions: np.ndarray,
    log_transmat: np.ndarray,
    log_startprob: np.ndarray,
    log_alpha: np.ndarray,
    log_beta: np.ndarray
):
    log_alpha[0] = log_emissions[0] + log_startprob
    for obs in range(1, nobs):
        prev = log_alpha[obs - 1]
        log_alpha[obs] = log_emissions[obs] + np.array([
            logsumexp(prev + log_transmat[obs, [0, 2]]), # 00, 10
            logsumexp(prev + log_transmat[obs, [1, 3]]), # 01, 11
        ])
    log_beta[-1] = 0 # log(1)
    for obs in range(nobs - 2, -1, -1):
        next = log_beta[obs + 1] + log_emissions[obs + 1]
        log_beta[obs] = np.array([
            logsumexp(next + log_transmat[obs, [0, 1]]), # 00, 01
            logsumexp(next + log_transmat[obs, [2, 3]]), # 10, 11
        ])
    return log_alpha, log_beta

def compute_log_gamma(log_alpha: np.ndarray, log_beta: np.ndarray):
    """
    compute gamma(z_n) = alpha(z_n)beta(z_n) / p(data)
    """
    log_gamma = log_alpha + log_beta
    log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
    return log_gamma

def compute_log_xi(
    nobs: int,
    log_emissions: np.ndarray,
    log_transmat: np.ndarray,
    log_alpha: np.ndarray,
    log_beta: np.ndarray,
):
    """
    compute xi(z_n-1, z_n) = alpha(z_n-1) emission(x_n,z_n) transition(z_n-1, z_n) beta(z_n) / p(data)
    """
    log_xi = np.zeros((nobs, 4), dtype=np.float128)
    log_xi += log_transmat
    # 00, 01, 10, 11
    for obs in range(1, nobs):
        log_xi[obs, 0] = (
            log_alpha[obs - 1, 0] + log_beta[obs, 0] + log_emissions[obs, 0]
        )
        log_xi[obs, 1] = (
            log_alpha[obs - 1, 0] + log_beta[obs, 1] + log_emissions[obs, 1]
        )
        log_xi[obs, 2] = (
            log_alpha[obs - 1, 1] + log_beta[obs, 0] + log_emissions[obs, 0]
        )
        log_xi[obs, 3] = (
            log_alpha[obs - 1, 1] + log_beta[obs, 1] + log_emissions[obs, 1]
        )
    log_xi -= logsumexp(log_xi, axis=1, keepdims=True)
    return log_xi

def compute_log_emission_bin(
    X: np.ndarray, 
    D: np.ndarray, 
    p: np.ndarray, 
    log_emissions: np.ndarray
):
    log_emissions[:, 0] = np.sum(binom.logpmf(X, D, (1 - p)[:, None]), axis=0)
    log_emissions[:, 1] = np.sum(binom.logpmf(X, D, p[:, None]), axis=0)
    return

def multisample_hmm(
    X: np.ndarray,
    Y: np.ndarray,
    D: np.ndarray,
    init_p: float,
    log_transmat: np.ndarray,
    fix_p=False,
    fix_transmat=True,
    max_iter=20,
    tol=10e-6,
    mirror_mhBAF=False,
    map_BAF=True # select between p and 1-p based on MAP estimation
):
    """
    Multisample Non-homogeous Hidden Markov Model (HMM) for BAF estimation
    1. Latent phasing variable z_n = 0,1, site prior given by phasing software
    2. Observed ref counts (X), alt counts (Y)
    3. Given input total counts D
    4. Transition is given by Hairs/supporting reads
    5. Emission is conditional binomial emission.

    X, Y, D: (nsample, nobs) multisample allele counts information, X=ref-counts
    log_transmat: (nsnp, 4) transition matrix in log space
    """

    nsamples, nobs = X.shape
    totals_sum = np.sum(D, axis=1) # (nsample, )
    assert np.all(totals_sum > 0), f"All samples should have per-bin total counts > 0 {totals_sum}"
    
    p = np.repeat(init_p, nsamples).astype(np.float64).clip(tol, 1 - tol)  # avoid log(0) in emission
    log_startprob = np.log([0.5, 0.5])
    log_emissions = np.zeros((nobs, 2), dtype=np.float64)
    gamma = np.zeros((nobs, 2), dtype=np.float64)
    log_alpha = np.zeros((nobs, 2), dtype=np.float64)  # alpha(z_n)
    log_beta = np.zeros((nobs, 2), dtype=np.float64)  # beta(z_n)

    for iteration in range(max_iter):
        prev_p = p
        p = np.clip(p, tol, 1 - tol)

        compute_log_emission_bin(X, D, p, log_emissions)

        # E-step
        forward_backward_numba(nobs, log_emissions, log_transmat, log_startprob, log_alpha, log_beta)
        log_gamma = compute_log_gamma(log_alpha, log_beta)
        np.exp(log_gamma, out=gamma)  # phasing information, in-place

        # log_xi = compute_log_xi(nobs, log_emissions, log_transmat, log_alpha, log_beta)
        # xi = np.exp(log_xi)

        # M-step: optimize p
        if not fix_p:
            p = (X @ gamma[:, 1] + Y @ gamma[:, 0]) / totals_sum
        # print(prev_p, p, log_emissions[:5], gamma[:5])
        if np.all(np.abs(p - prev_p) < tol):
            break
    
    phases = gamma[:, 1]  # posterior prob of USEREF
    # compute likelihoods
    observed_loglik = logsumexp(log_alpha[-1])

    phases_map = np.round(phases).astype(np.int8) # (snp, )
    complete_loglik = log_startprob[phases_map[0]]
    for i in range(1, nobs):
        complete_loglik += log_transmat[i, phases_map[i - 1] * 2 + phases_map[i]]
    complete_loglik += np.choose(phases_map, [log_emissions[:, 0], log_emissions[:, 1]]).sum()

    if map_BAF and np.sum(phases_map == 1) < np.sum(phases_map == 0):
        p = np.clip(1 - p, tol, 1 - tol)

    if mirror_mhBAF and np.mean(p) > 0.5:
        p = np.clip(1 - p, tol, 1 - tol)
        phases = gamma[:, 0]
    return p, phases, complete_loglik, observed_loglik
