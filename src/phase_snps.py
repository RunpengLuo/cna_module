import os
import sys
import numpy as np
import pandas as pd
from scipy.special import softmax


def get_phasing_entropy(phases: np.ndarray, tol=10e-6):
    """
    Given phasing posterior, compute average phasing entropy
    higher entropy indicates more allele-balanced bin
    """
    phases = np.clip(phases, tol, 1 - tol)
    entropy = -phases * np.log(phases) - (1 - phases) * np.log(1 - phases)
    return entropy

# def random_baf(alts: np.ndarray, refs: np.ndarray):
#     totals = refs + alts
#     totals_sum = np.sum(totals, axis=1)
#     n_samples, n_snps = totals.shape
#     phases = np.random.binomial(n=1, p=0.5, size=n_snps).astype(np.int8) # random phasing
#     betas = (refs @ phases[:, np.newaxis] + alts @ (1 - phases)[:, np.newaxis]).reshape(-1)
#     bafs = betas / totals_sum
#     if np.mean(bafs) > 0.5:
#         phases = 1 - phases
#         betas = (refs @ phases[:, np.newaxis] + alts @ (1 - phases)[:, np.newaxis]).reshape(-1)
#         bafs = betas / totals_sum
#     return bafs, phases

# def check_allelic_balanced(nalts: np.ndarray, nrefs: np.ndarray, tumor_phases: np.ndarray):
#     tumor_entropy = get_phasing_entropy(tumor_phases)
#     nalts = nalts.reshape(1, len(nalts))
#     nrefs = nrefs.reshape(1, len(nrefs))
#     runs = {
#         b: multisample_em(nalts, nrefs, b, mirror=True) for b in np.arange(0.45, 0.55, 0.01)
#     }
#     bafs, phases, ll = max(runs.values(), key=lambda x: x[-1])
#     normal_entropy = get_phasing_entropy(phases)
#     return tumor_entropy, normal_entropy, bafs[0]

def multisample_em(alts, refs, start, mirror=True, tol=10e-6):
    assert (
        refs.shape == alts.shape
    ), "Alternate and reference count arrays must have the same shape"
    # assert 0 < start <= 0.5, "Initial estimate must be in (0, 0.5]"

    totals = alts + refs

    n_samples, n_snps = alts.shape
    totals_sum = np.sum(totals, axis=1)
    assert np.all(
        totals_sum > 0
    ), "Every bin must have >0 SNP-covering reads in each sample!"

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

    # If mean(BAF) > 0.5, flip phases accordingly
    if mirror and np.mean(theta) > 0.5:
        theta = np.clip(theta, tol, 1 - tol)

        theta = 1 - theta
        t1 = np.sum(np.log(theta) @ refs + np.log(1 - theta) @ alts, axis=0)
        t2 = np.sum(np.log(1 - theta) @ refs + np.log(theta) @ alts, axis=0)
        phases = softmax(np.vstack([t1, t2]), axis=0)[0, :]

    t1 = np.sum(np.log(theta) * (phases * refs).T, axis=0)
    t2 = np.sum(np.log(1 - theta) * (phases * alts).T, axis=0)
    t3 = np.sum(np.log(theta) * ((1 - phases) * alts).T, axis=0)
    t4 = np.sum(np.log(1 - theta) * ((1 - phases) * refs).T, axis=0)
    log_likelihood = np.sum(t1 + t2 + t3 + t4)

    return theta, phases, log_likelihood
