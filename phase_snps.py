import os
import sys
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import binomtest, combine_pvalues, norm

def get_phasing_entropy(phases: np.ndarray, tol=10e-6):
    """
    Given phasing posterior, compute average phasing entropy
    higher entropy indicates more allele-balanced bin
    """
    phases = np.clip(phases, tol, 1 - tol)
    entropy = -phases * np.log(phases) - (1 - phases) * np.log(1 - phases)
    return entropy

def random_phasing(alts, refs, totals):
    _, n_snps = refs.shape
    phases = np.random.binomial(n=1, p=0.5, size=n_snps).astype(np.int8)
    theta = (refs @ phases + alts @ (1 - phases)) / np.sum(totals)
    log_likelihood = 0
    # t1 = np.sum(np.log(theta) * (phases * refs).T, axis=0)
    # t2 = np.sum(np.log(1 - theta) * (phases * alts).T, axis=0)
    # t3 = np.sum(np.log(theta) * ((1 - phases) * alts).T, axis=0)
    # t4 = np.sum(np.log(1 - theta) * ((1 - phases) * refs).T, axis=0)
    # log_likelihood = np.sum(t1 + t2 + t3 + t4)
    return theta, phases, log_likelihood

def binom_test(refs, totals, p=0.5):
    n_samples, n_snps = refs.shape
    snp_pvals = []
    ssnp_pvals = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_snps):
        if np.any(totals[:, i] <= 0):
            continue
        for s in range(n_samples):
            ssnp_pvals[s] = binomtest(refs[s, i], totals[s, i], p=p, alternative="two-sided").pvalue
        pval = combine_pvalues(ssnp_pvals, method = "fisher")[1]
        snp_pvals.append(pval)

    if len(snp_pvals) == 0:
        return np.nan

    _, combined_pval = combine_pvalues(snp_pvals, method = "fisher")
    return combined_pval

def binom_test_approx(refs, totals, p=0.5):
    mask = np.all(totals > 0, axis=1)
    if np.sum(mask) <= 0:
        return np.nan
    expected = totals[mask] * p
    std = np.sqrt(totals[mask] * p * (1 - p))

    Z = (refs[mask] - expected) / std
    pvals = 2 * norm.sf(np.abs(Z)) # (nsample, nsnp)
    if pvals.shape[0] == 1:
        pvals = pvals[0]
    else:
        pvals = [combine_pvalues(pvals[:, i], method = "fisher")[1] for i in range(pvals.shape[1])]
    _, combined_pval = combine_pvalues(pvals, method = "fisher")
    return combined_pval


def multisample_em(alts, refs, start, mirror=True, tol=10e-6):
    assert refs.shape == alts.shape, (
        "Alternate and reference count arrays must have the same shape"
    )
    # assert 0 < start <= 0.5, "Initial estimate must be in (0, 0.5]"

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
            # phases = softmax(e_arr, axis=0)[0, :]

            max_logl = np.maximum(e_arr[0], e_arr[1])
            logsumexp = max_logl + np.log(np.exp(e_arr[0] - max_logl) + np.exp(e_arr[1] - max_logl))
            phases = np.exp(e_arr[0] - logsumexp)
            assert not np.any(np.isnan(phases)), (phases, e_arr)

            # M-step
            t1 = refs @ phases + alts @ (1 - phases)
            theta = t1 / totals_sum
            assert not np.any(np.isnan(theta)), (theta, t1)

    # If mean(BAF) > 0.5, flip phases accordingly
    if mirror and np.mean(theta) > 0.5:
        theta = np.clip(theta, tol, 1 - tol)
        theta = 1 - theta

        e_arr[0] = np.log(theta) @ refs + np.log(1 - theta) @ alts
        e_arr[1] = np.log(1 - theta) @ refs + np.log(theta) @ alts
        # phases = softmax(e_arr, axis=0)[0, :]

        max_logl = np.maximum(e_arr[0], e_arr[1])
        logsumexp = max_logl + np.log(np.exp(e_arr[0] - max_logl) + np.exp(e_arr[1] - max_logl))
        phases = np.exp(e_arr[0] - logsumexp)

    t1 = np.sum(np.log(theta) * (phases * refs).T, axis=0)
    t2 = np.sum(np.log(1 - theta) * (phases * alts).T, axis=0)
    t3 = np.sum(np.log(theta) * ((1 - phases) * alts).T, axis=0)
    t4 = np.sum(np.log(1 - theta) * ((1 - phases) * refs).T, axis=0)
    log_likelihood = np.sum(t1 + t2 + t3 + t4)

    return theta, phases, log_likelihood
