import os
import sys
import numpy as np
import pandas as pd
from scipy.special import softmax, betaln, digamma
from scipy.stats import binomtest, combine_pvalues, binom, beta, norm
from scipy.optimize import minimize, minimize_scalar


# def get_phasing_entropy(phases: np.ndarray, tol=10e-6):
#     """
#     Given phasing posterior, compute average phasing entropy
#     higher entropy indicates more allele-balanced bin
#     """
#     phases = np.clip(phases, tol, 1 - tol)
#     entropy = -phases * np.log(phases) - (1 - phases) * np.log(1 - phases)
#     return entropy

# def random_phasing(alts, refs, totals):
#     _, n_snps = refs.shape
#     phases = np.random.binomial(n=1, p=0.5, size=n_snps).astype(np.int8)
#     theta = (refs @ phases + alts @ (1 - phases)) / np.sum(totals)
#     log_likelihood = 0
#     # t1 = np.sum(np.log(theta) * (phases * refs).T, axis=0)
#     # t2 = np.sum(np.log(1 - theta) * (phases * alts).T, axis=0)
#     # t3 = np.sum(np.log(theta) * ((1 - phases) * alts).T, axis=0)
#     # t4 = np.sum(np.log(1 - theta) * ((1 - phases) * refs).T, axis=0)
#     # log_likelihood = np.sum(t1 + t2 + t3 + t4)
#     return theta, phases, log_likelihood

# def binom_test(refs, totals, p=0.5):
#     n_samples, n_snps = refs.shape
#     snp_pvals = []
#     ssnp_pvals = np.zeros(n_samples, dtype=np.float64)
#     for i in range(n_snps):
#         if np.any(totals[:, i] <= 0):
#             continue
#         for s in range(n_samples):
#             ssnp_pvals[s] = binomtest(refs[s, i], totals[s, i], p=p, alternative="two-sided").pvalue
#         pval = combine_pvalues(ssnp_pvals, method = "fisher")[1]
#         snp_pvals.append(pval)

#     if len(snp_pvals) == 0:
#         return np.nan

#     _, combined_pval = combine_pvalues(snp_pvals, method = "fisher")
#     return combined_pval

# def binom_test_approx(refs, totals, p=0.5):
#     mask = np.all(totals > 0, axis=1)
#     if np.sum(mask) <= 0:
#         return np.nan
#     expected = totals[mask] * p
#     std = np.sqrt(totals[mask] * p * (1 - p))

#     Z = (refs[mask] - expected) / std
#     pvals = 2 * norm.sf(np.abs(Z)) # (nsample, nsnp)
#     if pvals.shape[0] == 1:
#         pvals = pvals[0]
#     else:
#         pvals = [combine_pvalues(pvals[:, i], method = "fisher")[1] for i in range(pvals.shape[1])]
#     _, combined_pval = combine_pvalues(pvals, method = "fisher")
#     return combined_pval

# Binom-EM model
def binomial_em(refs, alts, start, tol=10e-6):
    totals = alts + refs

    n_samples, n_snps = alts.shape
    totals_sum = np.sum(totals, axis=1)
    assert np.all(totals_sum > 0), (
        "Every bin must have >0 SNP-covering reads in each sample!"
    )

    e_arr = np.zeros((2, n_snps))
    if np.all(np.logical_or(refs == 0, alts == 0)):
        return np.array([0.0] * n_samples), np.ones(n_snps) * 0.5, 0.0
    else:
        theta = np.repeat(start, n_samples).astype(np.float64)
        prev = None
        while prev is None or np.all(np.abs(prev - theta) >= tol):
            prev = theta

            # Ensure that theta is not exactly 0 or 1 to avoid log(0)
            theta = np.clip(theta, tol, 1 - tol)

            # E-step in log space
            e_arr[0] = np.log(theta) @ refs + np.log(1 - theta) @ alts
            e_arr[1] = np.log(1 - theta) @ refs + np.log(theta) @ alts
            phases = softmax(e_arr, axis=0)[0, :]
            assert not np.any(np.isnan(phases)), (phases, e_arr)

            # M-step
            t1 = refs @ phases + alts @ (1 - phases)
            theta = t1 / totals_sum
            assert not np.any(np.isnan(theta)), (theta, t1)

    totals = refs + alts
    logA = binom.logpmf(refs, totals, theta[:, None]).sum(axis=0)
    logB = binom.logpmf(refs, totals, (1 - theta)[:, None]).sum(axis=0)
    ll = np.sum(np.logaddexp(logA, logB) - np.log(2.0))

    phases_map = np.round(phases).astype(np.int8) # (snp, )
    if np.sum(phases_map == 1) < np.sum(phases_map == 0):
        theta = np.clip(1 - theta, tol, 1 - tol)
    return theta, phases, ll

# Gaussian-EM model
def baf_var0_bin(n, p):
    return p * (1 - p) / n

def baf_var0_bbin(n, p, omega):
    return (p * (1 - p) / n) * ((n + omega) / (omega + 1))

def gaussian_em(props: np.ndarray, p0: float, var0: float, tol=10e-6, maxiter=100):
    n_samples, n_snps = props.shape
    e_arr = np.zeros((2, n_snps))

    ps = np.repeat(p0, n_samples).astype(np.float64)
    std2s = np.repeat(var0, n_samples).astype(np.float64)

    prev_ps = None
    prev_vars = None
    for _ in range(maxiter):
        prev_ps = ps
        prev_vars = std2s
        std2s = np.clip(std2s, a_min=tol, a_max=None)[:, None]
        p = np.clip(ps, tol, 1 - tol)[:, None]
        p_ = 1 - p

        e_arr[0] = -0.5 * np.sum((props - p_) ** 2 / std2s, axis=0)
        e_arr[1] = -0.5 * np.sum((props - p) ** 2 / std2s, axis=0)
        resp = softmax(e_arr, axis=0)
        gamma0, gamma1 = resp[0], resp[1]
        phases = gamma1

        ps = (np.sum(gamma0) - props @ (gamma0 - gamma1)) / n_snps
        std2s = (((props - p_) ** 2) @ gamma0 + ((props - p) ** 2) @ gamma1) / n_snps
        if np.all(np.abs(prev_ps - ps) < tol) and np.all(np.abs(prev_vars - std2s) < tol):
            break
    
    std = np.clip(np.sqrt(std2s), a_min=tol, a_max=None)
    logA = norm.logpdf(props, loc=ps[:, None], scale=std[:, None]).sum(axis=0)
    logB = norm.logpdf(props, loc=1 - ps[:, None], scale=std[:, None]).sum(axis=0)
    ll = np.sum(np.logaddexp(logA, logB) - np.log(2.0))

    phases_map = np.round(phases).astype(np.int8) # (snp, )
    if np.sum(phases_map == 1) < np.sum(phases_map == 0):
        ps = np.clip(1 - ps, tol, 1 - tol)
    return (ps, std2s), phases, ll

# BetaBinom-EM model
def omega_mle(refs, alts, p, omega0s=None, max_omega=np.Inf):
    def neg_loglik_logw(logw):
        w = np.exp(logw[0])
        a0 = w*p
        b0 = w*(1-p)
        a1 = refs + a0
        b1 = alts + b0
        ll = np.sum(betaln(a1, b1) - betaln(a0, b0))
        return -ll
    
    if omega0s is None:
        omega0s = np.arange(1, 2000, 100)

    best = (None, np.inf)
    for omega0 in omega0s:
        res = minimize(
            neg_loglik_logw,
            x0=np.log([omega0]),
            method="L-BFGS-B",
            bounds=[(np.log(1), np.log(max_omega))],
            # options={"ftol":1e-6}
        )
        if res.fun < best[1]:
            best = (np.exp(res.x[0]), res.fun)
        # if not res.success:
        #     print("warning:", omega0, res.message)
    return np.round(best[0]).astype(int), best[1]

def betabinomial_em(
    refs: np.ndarray,
    alts: np.ndarray, 
    p0: float, 
    w0: float, 
    tol=10e-6,
    maxiter=100, 
    fix_omega=True,
    max_omega=500):
    totals = alts + refs

    def neg_loglik(x):
        if fix_omega:
            opt_p = x[0]
            opt_w = w0
        else:
            [opt_p, opt_w] = x
        opt_p_ = 1 - opt_p
        a0 = opt_w * opt_p
        b0 = opt_w * opt_p_
        logprob0 = betaln(refs[si] + b0, alts[si] + a0) - betaln(b0, a0)
        logprob1 = betaln(refs[si] + a0, alts[si] + b0) - betaln(a0, b0)

        logprob0 = logprob0 @ gamma0
        logprob1 = logprob1 @ gamma1
        ll = np.sum(logprob0 + logprob1)
        return -ll


    n_samples, n_snps = alts.shape
    totals_sum = np.sum(totals, axis=1)
    assert np.all(totals_sum > 0), (
        "Every bin must have >0 SNP-covering reads in each sample!"
    )

    e_arr = np.zeros((2, n_snps))
    ps = np.repeat(p0, n_samples).astype(np.float64)
    ws = np.repeat(w0, n_samples).astype(np.float64)

    prev_ps = None
    prev_ws = None
    for _ in range(maxiter):
        prev_ps = ps
        prev_ws = ws
        p = np.clip(ps, tol, 1 - tol)[:, None] # (nsamples, 1)
        p_ = 1 - p

        ws_ = np.clip(ws, a_min=1, a_max=max_omega)[:, None]

        # E-step in log space
        e_arr[0] = betaln(refs + ws_ * p_, alts + ws_ * p) - betaln(ws_ * p_, ws_ * p)
        e_arr[1] = betaln(refs + ws_ * p, alts + ws_ * p_) - betaln(ws_ * p, ws_ * p_)
        resp = softmax(e_arr, axis=0)
        gamma0, gamma1 = resp[0], resp[1]
        phases = gamma1

        # M-step
        # optimize for each sample
        for si in range(n_samples):
            if fix_omega:
                res = minimize_scalar(lambda p: neg_loglik([p]), bounds=(tol, 1 - tol), method="bounded")
                ps[si] = np.clip(res.x, tol, 1 - tol)
            else:
                x0 = [p[si], np.log(np.clip(ws_[si], a_min=1, a_max=max_omega))]
                bnds = [(tol, 1 - tol), (np.log(1), np.log(max_omega))]
                res = minimize(
                    neg_loglik,
                    x0=x0,
                    method="L-BFGS-B",
                    bounds=bnds,
                    # options={"ftol":1e-6}
                )
                ps[si] = np.clip(res.x[0], tol, 1 - tol)
                ws[si] = np.exp(res.x[1])

        if np.all(np.abs(prev_ps - ps) < tol) and (not fix_omega or np.all(np.abs(prev_ws - ws) < tol)):
            break
    a0 = ws[:, None] * ps[:, None]
    b0 = ws[:, None] * (1 - ps[:, None])
    logA = (betaln(refs + b0, alts + a0) - betaln(b0, a0)).sum(axis=0)
    logB = (betaln(refs + a0, alts + b0) - betaln(a0, b0)).sum(axis=0)

    ll = np.sum(np.logaddexp(logA, logB) - np.log(2.0))
    phases_map = np.round(phases).astype(np.int8) # (snp, )
    if np.sum(phases_map == 1) < np.sum(phases_map == 0):
        ps = np.clip(1 - ps, tol, 1 - tol)
    return (ps, ws), phases, ll
