import os
import sys
from collections import Counter
import kneed
import numpy as np
import pandas as pd
from utils import *

from hatchet_parser import parse_arguments_cluster_blocks
from cluster_utils import *
from plot_utils_new import plot_1d2d

"""
Input:
RDR block by sample
a-allele, b-allele matrix, block by (sample + 1)

Run 2-mixture binomial or beta-binomial and Gaussian HMM model to fit states
"""

if __name__ == "__main__":
    args = parse_arguments_cluster_blocks()
    # model parameters
    work_dir = args["work_dir"]
    out_dir = os.path.join(work_dir, args["out_dir"])

    diag_t = args["t"]
    minK = args["minK"]
    maxK = args["maxK"]
    restarts = args["restarts"]
    n_iter = args["niters"]
    verbose = True

    # input files
    block_dir = os.path.join(work_dir, args["block_dir"])
    block_file = os.path.join(block_dir, "block_info.tsv.gz")
    rdr_mfile = os.path.join(block_dir, "block_matrix.rdr.npz")
    a_mfile = os.path.join(block_dir, "block_matrix.alpha.npz")
    b_mfile = os.path.join(block_dir, "block_matrix.beta.npz")
    t_mfile = os.path.join(block_dir, "block_matrix.total.npz")

    # output files
    os.makedirs(out_dir, exist_ok=True)
    plot_dir = os.path.join(out_dir, "plots")
    bbc_dir = os.path.join(out_dir, "labels")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(bbc_dir, exist_ok=True)

    ##################################################
    print("load arguments")
    blocks = pd.read_table(block_file, sep="\t")
    rdrs = np.load(rdr_mfile)["mat"].astype(np.float32)
    alphas = np.load(a_mfile)["mat"].astype(np.int32)
    betas = np.load(b_mfile)["mat"].astype(np.int32)
    totals = np.load(t_mfile)["mat"].astype(np.int32)
    (nsegments, nsamples) = rdrs.shape
    assert len(blocks) == nsegments, f"unmatched {len(blocks)} and {nsegments}"

    ##################################################
    print("prepare HMM inputs")

    # estimate over-dispersion parameter from normal sample
    init_taus = np.arange(10, 100, 10)
    tau = estimate_overdispersion(alphas[:, 0], betas[:, 0])
    use_binom = tau is None
    print(f"estimated tau={tau}, use binom={use_binom}")
    
    # divide into chromosome segments
    X_lengths = blocks.groupby(by="region_id", sort=False).agg("size").to_numpy()
    _nsegments = np.sum(X_lengths)
    assert _nsegments == nsegments, f"unmatched {_nsegments} and {nsegments}"
    print(f"#blocks={nsegments}")
    print(f"#segments={len(X_lengths)}")

    X_rdrs = rdrs[:, :]
    X_alphas = alphas[:, 1:]
    X_betas = betas[:, 1:]
    X_totals = totals[:, 1:]
    if X_rdrs.ndim == 1: # only one tumor sample
        X_rdrs = X_rdrs[:, np.newaxis]
        X_alphas = X_alphas[:, np.newaxis]
        X_betas = X_betas[:, np.newaxis]
        X_totals = X_totals[:, np.newaxis]

    # used for init p
    X_bafs = X_betas / X_totals
    baf_means = np.mean(X_bafs, axis=1)
    X_mhbafs = np.where(baf_means[:, None] > 0.5, 1 - X_bafs, X_bafs)
    X_inits = np.concatenate([X_rdrs, X_mhbafs], axis=1)

    # transition matrix - phasing, (N, )
    switchprobs = blocks["switchprobs"].to_numpy()
    log_switchprobs = np.log(switchprobs)
    log_stayprobs = np.log(1 - switchprobs)

    for K in range(minK, maxK + 1):
        print("==================================================")
        print(f"running HMM on K={K}")
        log_transmat = np.log(make_transmat(1 - diag_t, K))
        best_model = {"model_ll": -np.inf}
        for s in range(restarts):
            curr_model = run_hmm(
                X_rdrs, X_alphas, X_betas,
                X_totals, X_inits, X_lengths,
                log_switchprobs, log_stayprobs,
                log_transmat, K, tau,
                n_iter, random_state=s,
            )
            model_ll = curr_model["model_ll"]
            print(f"restart={s}, model loglik={model_ll: .6f}")
            if model_ll > best_model["model_ll"]:
                best_model = curr_model
        
        raw_cluster_labels = best_model["cluster_labels"]
        # map to rank labels to avoid cluster label gaps
        _, inv = np.unique(raw_cluster_labels, return_inverse=True)
        cluster_labels = inv + 1

        # cluster_labels = np.argsort(np.argsort(raw_cluster_labels)) + 1

        # phase_labels = best_model["phase_labels"]
        elbo_trace = best_model["elbo_trace"]
        num_clusters = len(np.unique(cluster_labels))

        # (#cluster, #samples)
        expected_rdr_mean = best_model["RDR_means"]
        expected_rdr_var = best_model["RDR_vars"]
        expected_baf_mean = best_model["BAF_means"]

        # expected_rdr_mean = np.zeros((len(num_clusters), nsamples), dtype=np.float64)
        # expected_rdr_var = np.zeros((len(num_clusters), nsamples), dtype=np.float64)
        # expected_baf_mean = np.zeros((len(num_clusters), nsamples), dtype=np.float64)

        # for cluster_id in range(1, num_clusters + 1):
        #     cluster_mask = cluster_labels == cluster_id
        #     expected_rdr_mean[cluster_id, :] = np.mean(rdrs[cluster_mask, :], axis=0)
        #     expected_rdr_var[cluster_id, :] = np.var(rdrs[cluster_mask, :], axis=0)

        genome_file = "./reference/T2T-CHM13v2.0.sizes"
        plot_1d2d(
            blocks,
            X_bafs,
            X_rdrs,
            cluster_labels,
            expected_rdr_mean,
            expected_baf_mean,
            genome_file,
            plot_dir,
            out_prefix=f"K{K}_",
            plot_mirror_baf=True
        )
