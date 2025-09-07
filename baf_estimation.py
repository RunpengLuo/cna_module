import time

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import binom
from numba import njit
# from scipy.special import expit

from potts_segmentation_utils import *
from baf_hmm_model_bin import *


def single_binom_mle(
    X: np.ndarray, Y: np.ndarray, D: np.ndarray, mirror=True, tol=10e-6
):
    p = np.sum(X, axis=1) / np.sum(D, axis=1)
    p = np.clip(p, tol, 1 - tol)

    counts = X
    if mirror and np.mean(p) > 0.5:
        p = np.clip(1 - p, tol, 1 - tol)
        counts = Y
    bal_p = np.repeat(0.5, len(p))
    ll = np.sum(binom.logpmf(counts, D, bal_p[:, None]))
    return p, ll


def compute_ICL(ll: float, k: int, n: int):
    """
    return ICL if ll is complete loglik
    return BIC if ll is observed loglik
    """
    return -2 * ll + k * np.log(n)


def compute_BAF(
    bin_ids: np.ndarray,
    snp_info: pd.DataFrame,
    ref_mat: np.ndarray,
    alt_mat: np.ndarray,
    tot_mat: np.ndarray,
    hairs: np.ndarray,
    nbins: int,
    nsamples: int,
    mirror_mhBAF=False,
    map_BAF=True,
    v=1,
):
    print("compute mhBAF via multisample binomial HMM")
    init_starts = np.arange(0, 0.55, 0.05)
    np.random.seed(42)
    # for single-binomial MLE to avoid ref-biases
    rand_phases = np.random.randint(2, size=len(snp_info))
    rand_ref_mat = np.where(rand_phases[:, None] == 0, alt_mat, ref_mat)
    rand_alt_mat = tot_mat - rand_ref_mat

    # snp_gts = snp_info["GT"].to_numpy()
    # snp_pss = snp_info["PS"].to_numpy()
    # snp_unphased = pd.isna(snp_gts)
    # snp_gts[pd.isna(snp_gts)] = 1
    # snp_gts = snp_gts.astype(np.int8)

    # sitewise_transmat = build_sitewise_transmat(
    #     hairs, snp_gts, snp_pss, snp_unphased, log=True,
    # )
    sitewise_transmat = build_sitewise_transmat(hairs, log=True)

    # outputs
    cov_mat = np.zeros((nbins, nsamples), dtype=np.float64)
    mix_baf_mat = np.zeros((nbins, nsamples), dtype=np.float64)
    nomix_baf_mat = np.zeros((nbins, nsamples), dtype=np.float64)
    # normal and tumor runs
    mix_icl_mat = np.zeros((nbins, 2), dtype=np.float64)
    nomix_icl_mat = np.zeros((nbins, 2), dtype=np.float64)

    snp_bins = snp_info.groupby(by="bin_id", sort=False)
    ts = time.time()
    for bi in bin_ids:
        # if bi >= 10:
        #     sys.exit(0)
        if v > 0 and bi % 500 == 0:
            print(f"elapsed={time.time() - ts:.3f}s {bi}/{nbins}")

        bin_snps = snp_bins.get_group(bi)
        bin_nsnp = len(bin_snps)
        snp_idx = bin_snps.index.to_numpy()

        bin_refs = ref_mat[snp_idx, :].T
        bin_alts = alt_mat[snp_idx, :].T
        bin_tots = tot_mat[snp_idx, :].T

        cov_mat[bi, :] = np.sum(bin_tots, axis=1) / bin_snps["#SNPS"].sum()

        # tumor run
        runs = {
            b: multisample_hmm(
                bin_refs[1:],
                bin_alts[1:],
                bin_tots[1:],
                b,
                sitewise_transmat[snp_idx, :],
                mirror_mhBAF=mirror_mhBAF,
                map_BAF=map_BAF,
            )
            for b in init_starts
        }
        best_b = max(runs.keys(), key=lambda b: runs[b][-1])
        bafs, phases, cll, ll = runs[best_b]
        mix_baf_mat[bi, 1:] = bafs
        mix_icl_mat[bi, 1] = compute_ICL(cll, 1, bin_nsnp)
        snp_info.loc[snp_idx, "PHASE"] = phases

        # normal run
        runs = {
            b: multisample_hmm(
                bin_refs[0][None, :],
                bin_alts[0][None, :],
                bin_tots[0][None, :],
                b,
                sitewise_transmat[snp_idx, :],
                mirror_mhBAF=mirror_mhBAF,
                map_BAF=map_BAF,
            )
            for b in init_starts
        }
        best_b = max(runs.keys(), key=lambda b: runs[b][-1])
        bafs, phases, cll, _ = runs[best_b]
        mix_baf_mat[bi, 0] = bafs[0]
        mix_icl_mat[bi, 0] = compute_ICL(cll, 1, bin_nsnp)

        bin_rrefs = rand_ref_mat[snp_idx, :].T
        bin_ralts = rand_alt_mat[snp_idx, :].T
        baf, ll = single_binom_mle(
            bin_rrefs[1:], bin_ralts[1:], bin_tots[1:], mirror=mirror_mhBAF
        )
        nomix_baf_mat[bi, 1:] = baf
        nomix_icl_mat[bi, 1] = compute_ICL(ll, 1, bin_nsnp)

        baf, ll = single_binom_mle(
            bin_rrefs[0][None, :],
            bin_ralts[0][None, :],
            bin_tots[0][None, :],
            mirror=mirror_mhBAF,
        )
        nomix_baf_mat[bi, 0] = baf
        nomix_icl_mat[bi, 0] = compute_ICL(ll, 1, bin_nsnp)
    return cov_mat, mix_baf_mat, mix_icl_mat, nomix_baf_mat, nomix_icl_mat


def compute_BAF_prior(
    bin_ids: np.ndarray,
    snp_info: pd.DataFrame,
    ref_mat: np.ndarray,
    alt_mat: np.ndarray,
    tot_mat: np.ndarray,
    nbins: int,
    nsamples: int,
    mirror_mhBAF=True,
    v=1,
):
    print("compute mhBAF via prior phasing labels")

    # outputs
    baf_mat = np.zeros((nbins, nsamples), dtype=np.float64)
    alpha_mat = np.zeros((nbins, nsamples), dtype=np.int32)
    beta_mat = np.zeros((nbins, nsamples), dtype=np.int32)
    cov_mat = np.zeros((nbins, nsamples), dtype=np.int32)

    icl_mat = np.zeros((nbins, 2), dtype=np.float64)

    snp_bins = snp_info.groupby(by="bin_id", sort=False)
    for bi in bin_ids:
        bin_snps = snp_bins.get_group(bi)
        bin_nsnp = len(bin_snps)
        snp_idx = bin_snps.index.to_numpy()

        bin_refs = ref_mat[snp_idx, :].T
        bin_alts = alt_mat[snp_idx, :].T
        bin_tots = tot_mat[snp_idx, :].T

        beta_mat[bi] = np.sum(bin_refs, axis=1)
        alpha_mat[bi] = np.sum(bin_alts, axis=1)
        bin_tot = np.sum(bin_tots, axis=1)
        baf_mat[bi] = beta_mat[bi] / bin_tot
        cov_mat[bi] = bin_tot / bin_snps["#SNPS"].sum()
        # ll_normal = np.sum(binom.logpmf(bin_refs[0][None, :], bin_tots[0][None, :], baf_mat[bi, 0][:, None]))
        # ll_tumor = np.sum(binom.logpmf(bin_refs[1:], bin_tots[1:], baf_mat[bi, 1:][:, None]))
        # icl_mat[bi, 0] = compute_ICL(ll_normal, k=1, n=bin_nsnp)
        # icl_mat[bi, 1] = compute_ICL(ll_tumor, k=1, n=bin_nsnp)
        if mirror_mhBAF and np.mean(baf_mat[bi, 1:]) > 0.5:
            baf_mat[bi, 1:] = 1 - baf_mat[bi, 1:]
            beta_mat[bi, 1:] = alpha_mat[bi, 1:]
            alpha_mat[bi, 1:] = bin_tot[1:] - beta_mat[bi, 1:]
    return cov_mat, baf_mat, alpha_mat, beta_mat


def baf_model_select(
    bin_ids: np.ndarray,
    snp_info: pd.DataFrame,
    mix_baf_mat: np.ndarray,
    mix_icl_mat: np.ndarray,
    nomix_baf_mat: np.ndarray,
    nomix_icl_mat: np.ndarray,
    nbins: int,
):
    nsnp_per_bin = np.array(
        [len(snp_info.loc[snp_info["bin_id"] == b, :]) for b in bin_ids]
    )
    scaled_icl_delta = (mix_icl_mat - nomix_icl_mat) / nsnp_per_bin[:, None]

    potts_states = np.zeros((nbins, 2), dtype=np.int8)
    potts_states[:, 0] = potts_segmentation(bin_ids, snp_info, scaled_icl_delta[:, 0])
    potts_states[:, 1] = potts_segmentation(bin_ids, snp_info, scaled_icl_delta[:, 1])

    potts_bafs = np.zeros(mix_baf_mat.shape, dtype=np.float64)
    potts_bafs[:, 0] = potts_baf_refinement(
        potts_states[:, 0], mix_baf_mat[:, 0], nomix_baf_mat[:, 0]
    )
    for i in range(1, mix_baf_mat.shape[1]):
        potts_bafs[:, i] = potts_baf_refinement(
            potts_states[:, 1], mix_baf_mat[:, i], nomix_baf_mat[:, i]
        )
    return potts_states, potts_bafs, scaled_icl_delta
