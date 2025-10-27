import os
import sys
import time
from collections import Counter
import kneed
import numpy as np
import pandas as pd
from sklearn import cluster, mixture
from utils import *
from sklearn.preprocessing import StandardScaler


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
        print(baf_means.flatten().round(3))
        print(rdr_means.flatten().round(3))
        print(rdr_vars.flatten().round(3))

    return rdr_means, rdr_vars, baf_means

##################################################
def split_segment(rel_start, rel_end, cap):
    """Split [rel_start, rel_end) into balanced sub-blocks capped by cap."""
    L = rel_end - rel_start
    if L <= cap:
        return [(rel_start, rel_end)]
    n_blocks = int(np.ceil(L / cap))
    cut_points = np.linspace(rel_start, rel_end, n_blocks + 1, dtype=int)
    return list(zip(cut_points[:-1], cut_points[1:]))

def fused_lasso_segmentations(
    blocks: pd.DataFrame,
    X_rdrs: np.ndarray,
    X_mhbafs: np.ndarray,
    X_lengths: np.ndarray,
    plot_dir: str,
    penalty=3,
    min_block_size=100,
    genome_file=None,
):
    X_inits = np.concatenate([X_rdrs, X_mhbafs], axis=1)
    flasso_labels = np.empty(np.sum(X_lengths), dtype=np.int32)
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
        assert bkps[-1] == nobs
        rel_start = 0
        for rel_end in bkps:
            prev_k = abs_k
            # iterate sub-blocks with cap
            for sub_start in range(rel_start, rel_end, min_block_size):
                abs_start = start + sub_start
                abs_end = min(start + rel_end, abs_start + min_block_size)

                binary_labels[abs_start:abs_end]  = bin_k
                flasso_labels[abs_start:abs_end] = abs_k

                seg_rdrs.append(np.mean(X_rdrs[abs_start:abs_end, :], axis=0))
                seg_mhbafs.append(np.mean(X_mhbafs[abs_start:abs_end, :], axis=0))

                bin_k = (bin_k + 1) % 2
                abs_k += 1

            abs_k = max(abs_k, prev_k + 1)
            rel_start = rel_end
        start = end
    
    print(f"#fused-lasso segments={abs_k}")
    plot_1d2d(
        blocks,
        X_mhbafs,
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

    return seg_rdrs, seg_mhbafs, flasso_labels

def init_hmm_segs(
    X_rdrs: np.ndarray,
    X_mhbafs: np.ndarray,
    seg_rdrs: np.ndarray,
    seg_mhbafs: np.ndarray,
    flasso_labels: np.ndarray,
    K: int,
    random_state=42,
    min_covar=1e-3,
    tol=1e-6,
    init_method="ward",
    plot_dir=None,
    verbose=True,
):
    assert init_method in ["k-means++", "ward"]
    M = seg_rdrs.shape[1]

    seg_inits = np.concatenate([seg_rdrs, seg_mhbafs], axis=1)
    seg_inits_std = StandardScaler().fit_transform(seg_inits)
    if init_method == "k-means++":
        kmeans = cluster.KMeans(
            n_clusters=K,
            random_state=random_state,
            init="k-means++",
            n_init=1,
        )
        cluster_labels = kmeans.fit_predict(X=seg_inits_std)
    elif init_method == "ward":
        hier = cluster.AgglomerativeClustering(
            n_clusters=K,
            linkage="ward",
        )
        cluster_labels = hier.fit_predict(X=seg_inits_std)
    else:
        raise ValueError()
    
    baf_means = np.full((K, M), fill_value=0, dtype=np.float32)
    rdr_means = np.full((K, M), fill_value=0, dtype=np.float32)
    rdr_vars = np.full((K, M), fill_value=min_covar, dtype=np.float32)
    seg_labels = np.arange(seg_mhbafs.shape[0])
    for k in range(K):
        raw_mask = np.isin(flasso_labels, seg_labels[cluster_labels == k])
        num_points = np.sum(raw_mask)
        if num_points < 2:
            continue
        cluster_rdrs = X_rdrs[raw_mask, :]
        cluster_bafs = X_mhbafs[raw_mask, :]
        rdr_means[k] = np.mean(cluster_rdrs, axis=0)
        baf_means[k] = np.mean(cluster_bafs, axis=0)
        rdr_vars[k, :] = np.sum((cluster_rdrs - rdr_means[k]) ** 2, axis=0) / (
            num_points - 1
        )

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
