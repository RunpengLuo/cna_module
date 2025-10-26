import os
import sys
import time
from collections import Counter
import kneed
import numpy as np
import pandas as pd
from scipy.special import betaln, gammaln, logsumexp, xlogy
from scipy.optimize import minimize, minimize_scalar
from sklearn import cluster, mixture
from utils import *

import ruptures

# from numba import njit
import matplotlib.pyplot as plt
import seaborn as sns
from plot_utils_new import plot_1d2d

def impose_mhbaf_constraints(baf_means: np.ndarray):
    is_minor = np.mean(baf_means, axis=1) <= 0.5
    baf_means[~is_minor, :] = 1 - baf_means[~is_minor, :]
    return baf_means

##################################################
def init_hmm(
    X_rdrs: np.ndarray,
    X_alphas: np.ndarray,
    X_betas: np.ndarray,
    X_totals: np.ndarray,
    K: int,
    random_state=42,
    min_covar=1e-3,
    tol=1e-6,
    n_init=10,
    init_minor=True,
    init_method="k-means++",
    verbose=True,
):
    assert init_method in ["k-means++", "gmm", "dir_gmm"]
    M = X_rdrs.shape[1]

    offset = 0
    rdr_scale = np.max(X_rdrs, axis=0, keepdims=True)
    X_rdrs_scaled = (X_rdrs - offset) / rdr_scale
    X_bafs = X_betas / X_totals
    X_bafs = impose_mhbaf_constraints(X_bafs)
    # X_inits1 = np.concatenate([X_rdrs_scaled, X_bafs], axis=1)
    # X_inits2 = np.concatenate([X_rdrs_scaled, 1 - X_bafs], axis=1)
    # init_mask = np.random.randint(0, 2, size=X_bafs.shape[0], dtype=bool)
    # X_inits = np.where(init_mask[:, None], X_inits1, X_inits2)
    X_inits = np.concatenate([X_rdrs_scaled, X_bafs], axis=1)

    if init_method == "k-means++":
        kmeans = cluster.KMeans(
            n_clusters=K,
            random_state=random_state,
            init="k-means++",
            n_init="auto",
            max_iter=1
        )
        cluster_labels = kmeans.fit_predict(X=X_inits)
        means = kmeans.cluster_centers_
        
        baf_means = means[:, M:]
        if init_minor:
            baf_means = impose_mhbaf_constraints(baf_means)
        baf_means = np.clip(baf_means, a_min=tol, a_max=1 - tol)  # avoid log(0)
        
        rdr_means = means[:, :M] * rdr_scale + offset
        rdrs = X_inits[:, :M]
        if rdrs.ndim == 1:
            rdrs = rdrs[:, np.newaxis]
        rdr_vars = np.full((K, M), fill_value=min_covar, dtype=np.float32)
        for k in range(K):
            mask = cluster_labels == k
            num_points = np.sum(mask)
            if num_points < 2:
                continue
            cluster_rdrs = rdrs[mask, :]
            rdr_vars[k, :] = np.sum((cluster_rdrs - rdr_means[k, :]) ** 2, axis=0) / (
                num_points - 1
            )
        rdr_vars = np.maximum(rdr_vars, min_covar)
    # elif init_method == "gmm":
    #     gmm = mixture.GaussianMixture(
    #         n_components=K,
    #         random_state=random_state,
    #         covariance_type="diag",
    #         init_params="k-means++",
    #         n_init=1,
    #         reg_covar=min_covar,
    #     )
    #     cluster_labels = gmm.fit_predict(X=X_inits)

    #     means = gmm.means_
    #     baf_means = means[:, M:]
    #     if init_minor:
    #         baf_means = impose_mhbaf_constraints(baf_means)
    #     baf_means = np.clip(baf_means, a_min=tol, a_max=1 - tol)  # avoid log(0)
    #     rdr_means = means[:, :M] * rdr_scale + offset
    #     rdr_vars = gmm.covariances_[:, :M]
    #     rdr_vars = np.maximum(rdr_vars * (rdr_scale**2), min_covar)
    # elif init_method == "dir_gmm":
    #     gmm = mixture.BayesianGaussianMixture(
    #         n_components=K,
    #         random_state=random_state,
    #         covariance_type="diag",
    #         init_params="k-means++",
    #         n_init=n_init,
    #         max_iter=200,
    #         reg_covar=min_covar,
    #         weight_concentration_prior_type="dirichlet_distribution",
    #         weight_concentration_prior=1.1
    #     )
    #     cluster_labels = gmm.fit_predict(X=X_inits)

    #     means = gmm.means_
    #     vars = gmm.covariances_
    else:
        raise ValueError()
    if verbose:
        print(f"Init")
        print(baf_means.flatten())
        print(rdr_means.flatten())
        print(rdr_vars.flatten())

    return rdr_means, rdr_vars, baf_means

##################################################
def bkps_to_labels(bkps, N=None):
    """Convert ruptures breakpoints to integer segment labels."""
    if N is None:
        N = bkps[-1]
    labels = np.zeros(N, dtype=int)
    start = 0
    for k, end in enumerate(bkps):
        labels[start:end] = k
        start = end
    return labels

def fused_lasso_segmentations(
    blocks: pd.DataFrame,
    X_rdrs: np.ndarray,
    X_alphas: np.ndarray,
    X_betas: np.ndarray,
    X_totals: np.ndarray,
    X_lengths: np.ndarray,
    plot_dir: str,
    penalty=3,
    genome_file=None,
):
    M = X_alphas.shape[1]
    X_bafs = X_betas / X_totals
    X_mhbafs = impose_mhbaf_constraints(X_bafs)
    X_inits = np.concatenate([X_rdrs, X_mhbafs], axis=1)
    binary_labels = np.empty(np.sum(X_lengths), dtype=np.int32)

    start = 0
    abs_k = 0
    bin_k = 0

    seg_rdrs = []
    seg_mhbafs = []
    for s, nobs in enumerate(X_lengths):
        print(s, nobs)
        end = start + nobs
        X_inits_sub = X_inits[start:end, :]
        algo = ruptures.Pelt(
            model="l2",
            min_size=5,
            jump=2
        ).fit(X_inits_sub)
        bkps = algo.predict(pen=penalty)
        rel_start = 0
        for rel_end in bkps:
            abs_start = start + rel_start
            abs_end = start + rel_end
            binary_labels[abs_start:abs_end] = bin_k
            mean_rdrs = np.mean(X_rdrs[abs_start:abs_end, :], axis=0)
            mean_mhbafs = np.mean(X_mhbafs[abs_start:abs_end, :], axis=0)
            seg_rdrs.append(mean_rdrs)
            seg_mhbafs.append(mean_mhbafs)

            bin_k = (bin_k + 1) % 2
            rel_start = rel_end
        abs_k += len(bkps)
        start = end
    
    print(f"#fused-lasso segments={abs_k}")
    plot_1d2d(
        blocks,
        X_bafs,
        X_rdrs,
        binary_labels,
        None,
        None,
        genome_file,
        plot_dir,
        out_prefix=f"seg_",
        plot_mirror_baf=False,
    )

    seg_rdrs = np.vstack(seg_rdrs)
    seg_mhbafs = np.vstack(seg_mhbafs)

    return seg_rdrs, seg_mhbafs

def init_hmm_segs(
    seg_rdrs: np.ndarray,
    seg_mhbafs: np.ndarray,
    K: int,
    random_state=42,
    min_covar=1e-3,
    tol=1e-6,
    init_method="k-means++",
    verbose=True,
    plot_dir=None,
):
    assert init_method in ["k-means++", "ward"]
    M = seg_rdrs.shape[1]

    offset = 0
    rdr_scale = np.max(seg_rdrs, axis=0, keepdims=True)
    seg_rdrs_scaled = (seg_rdrs - offset) / rdr_scale # (0, 1)

    seg_inits = np.concatenate([seg_rdrs_scaled, seg_mhbafs], axis=1)
    if init_method == "k-means++":
        kmeans = cluster.KMeans(
            n_clusters=K,
            random_state=random_state,
            init="k-means++",
            max_iter=1
        )
        cluster_labels = kmeans.fit_predict(X=seg_inits)
        means = kmeans.cluster_centers_
        baf_means = means[:, M:]
        rdr_means = means[:, :M] * rdr_scale + offset
        rdr_vars = np.full((K, M), fill_value=min_covar, dtype=np.float32)
    elif init_method == "ward":
        hier = cluster.AgglomerativeClustering(
            n_clusters=K,
            linkage="ward",
        )
        cluster_labels = hier.fit_predict(X=seg_inits)
        means = np.array([seg_inits[cluster_labels==k].mean(axis=0) for 
                          k in np.unique(cluster_labels)])
        baf_means = means[:, M:]
        rdr_means = means[:, :M] * rdr_scale + offset
        rdr_vars = np.full((K, M), fill_value=min_covar, dtype=np.float32)
    else:
        raise ValueError()

    if not plot_dir is None:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(
            seg_mhbafs, seg_rdrs,
            c=cluster_labels,
            s=40,
            alpha=0.6,
            cmap="tab10",
            label="Segments"
        )

        # Plot centroids
        ax.scatter(
            baf_means, rdr_means,
            c="black", s=40, marker="X", edgecolor="white", linewidth=1.5,
            label="Centroids"
        )

        ax.set_xlabel("Minor BAF deviation")
        ax.set_ylabel("RDR")
        ax.set_title(f"Segment-level clusters and {init_method} centroids")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"init_seg_K{K}.png"), dpi=300)
        plt.close(fig)
    if verbose:
        print(f"Init")
        print(baf_means.flatten())
        print(rdr_means.flatten())
        print(rdr_vars.flatten())
    return rdr_means, rdr_vars, baf_means
