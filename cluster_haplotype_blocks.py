import os
import sys
import shutil
from pathlib import Path

from collections import Counter
import kneed
import numpy as np
import pandas as pd
from utils import *

from hatchet_parser import parse_arguments_cluster_blocks
from cluster_utils import *
from plot_utils_new import plot_1d2d
from initialize_hmm import *

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

    lasso_penalty = 0.5
    init_method = "ward" # "k-means++" "gmm"
    decode_method = "map"
    score_method = "bic"

    # input files
    block_dir = os.path.join(work_dir, args["block_dir"])
    bb_file = os.path.join(block_dir, "bulk.bb")
    block_file = os.path.join(block_dir, "block_info.tsv.gz")
    rdr_mfile = os.path.join(block_dir, "block_matrix.rdr.npz")
    a_mfile = os.path.join(block_dir, "block_matrix.alpha.npz")
    b_mfile = os.path.join(block_dir, "block_matrix.beta.npz")
    t_mfile = os.path.join(block_dir, "block_matrix.total.npz")

    genome_file = args["genome_file"]

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
    tau = estimate_overdispersion(alphas[:, 0], betas[:, 0], max_tau=1e5)
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
    if X_rdrs.ndim == 1:  # only one tumor sample
        X_rdrs = X_rdrs[:, np.newaxis]
        X_alphas = X_alphas[:, np.newaxis]
        X_betas = X_betas[:, np.newaxis]
        X_totals = X_totals[:, np.newaxis]

    # transition matrix - phasing, (N, )
    switchprobs = blocks["switchprobs"].to_numpy()
    log_switchprobs = np.log(switchprobs)
    log_stayprobs = np.log(1 - switchprobs)

    ##################################################
    print("fused lasso segmentation")
    seg_rdrs, seg_mhbafs = fused_lasso_segmentations(
        blocks,
        X_rdrs,
        X_alphas,
        X_betas,
        X_totals,
        X_lengths,
        plot_dir,
        penalty=lasso_penalty,
        genome_file=genome_file,
    )

    ##################################################
    bb = pd.read_csv(bb_file, sep="\t")
    samples = bb["SAMPLE"].unique()
    model_scores = []
    # blocks["CLUSTER"] = 0
    bb["CLUSTER"] = 0
    for K in range(minK, maxK + 1):
        print("==================================================")
        print(f"running HMM on K={K}")
        log_transmat = np.log(make_transmat(1 - diag_t, K))
        best_model = {"model_ll": -np.inf}
        for s in range(restarts):
            rdr_means, rdr_vars, baf_means = init_hmm_segs(
                seg_rdrs, 
                seg_mhbafs, 
                K, 
                s, 
                init_method=init_method, 
                plot_dir=plot_dir
            )
            # rdr_means, rdr_vars, baf_means = init_hmm(
            #     X_rdrs,
            #     X_alphas,
            #     X_betas,
            #     X_totals,
            #     K,
            #     random_state=s,
            #     init_minor=True,
            #     init_method=init_method,
            #     verbose=verbose,
            # )
            curr_model = run_hmm(
                X_rdrs,
                X_alphas,
                X_betas,
                X_totals,
                X_lengths,
                log_switchprobs,
                log_stayprobs,
                log_transmat,
                K,
                rdr_means,
                rdr_vars,
                baf_means,
                tau,
                n_iter,
                decode_method=decode_method,
                score_method=score_method,
                plot_dir=plot_dir,
            )
            model_ll = curr_model["model_ll"]
            print(f"restart={s}, model loglik={model_ll: .6f}")
            if model_ll > best_model["model_ll"]:
                best_model = curr_model

        ##################################################
        # map to rank labels to avoid cluster label gaps
        raw_cluster_labels = best_model["cluster_labels"]
        _, inv = np.unique(raw_cluster_labels, return_inverse=True)
        cluster_labels = inv + 1
        best_model["cluster_labels"] = cluster_labels
        unique_labels = np.unique(cluster_labels)

        phase_labels = best_model["phase_labels"]
        elbo_trace = best_model["elbo_trace"]

        # (#cluster, #samples)
        expected_rdr_mean = best_model["RDR_means"]
        expected_rdr_var = best_model["RDR_vars"]
        expected_baf_mean = best_model["BAF_means"]

        X_betas_phased = (
            X_alphas * (1 - phase_labels[:, None]) + X_betas * phase_labels[:, None]
        )
        phased_bafs = X_betas_phased / X_totals
        # manually merge clusters? TODO

        model_scores.append(best_model["model_score"])

        ##################################################
        plot_1d2d(
            blocks,
            phased_bafs,
            X_rdrs,
            cluster_labels,
            expected_rdr_mean,
            expected_baf_mean,
            genome_file,
            plot_dir,
            out_prefix=f"K{K}_",
            plot_mirror_baf=False,
        )

        ##################################################
        # save to bbc and block TODO remove this later!
        # blocks["CLUSTER"] = cluster_labels
        bb["CLUSTER"] = np.repeat(cluster_labels, nsamples)
        bb_grps = bb.groupby(by="CLUSTER", sort=False)
        seg_rows = []
        for l, label in enumerate(unique_labels):
            bb_grp = bb_grps.get_group(label)
            for s, sample in enumerate(samples):
                bb_sample = bb_grp.loc[bb_grp["SAMPLE"] == sample, :]
                seg_nsnps = bb_sample["#SNPS"].sum()
                ave_cov = (
                    np.sum(bb_sample["COV"].to_numpy() * bb_sample["#SNPS"].to_numpy())
                    / seg_nsnps
                )
                seg_rows.append(
                    [
                        label,
                        sample,
                        len(bb_sample),
                        bb_sample["#SNPS"].sum(),
                        ave_cov,
                        expected_baf_mean[l, s],
                        expected_rdr_mean[l, s],
                        expected_rdr_var[l, s],
                    ]
                )
        seg = pd.DataFrame(
            data=seg_rows,
            columns=[
                "#ID",
                "SAMPLE",
                "#BINS",
                "#SNPS",
                "COV",
                "BAF",
                "RD",
                "RD-var",
            ],
        )

        bbc_file = os.path.join(bbc_dir, f"bulk{K}.bbc")
        seg_file = os.path.join(bbc_dir, f"bulk{K}.seg")
        bb.to_csv(bbc_file, sep="\t", header=True, index=False)
        seg.to_csv(seg_file, sep="\t", header=True, index=False)

    if score_method == "bic":
        opt_K = minK + np.argmin(model_scores)
    else:
        raise ValueError()

    scores_df = pd.DataFrame(
        data={"K": list(range(minK, maxK + 1)), score_method: model_scores}
    )
    scores_df.to_csv(
        os.path.join(out_dir, "model_scores.tsv"), sep="\t", header=True, index=False
    )

    print("Optimal K:", opt_K)

    shutil.copy2(
        Path(os.path.join(bbc_dir, f"bulk{opt_K}.bbc")),
        Path(os.path.join(bbc_dir, f"bulk.bbc")),
    )
    shutil.copy2(
        Path(os.path.join(bbc_dir, f"bulk{opt_K}.seg")),
        Path(os.path.join(bbc_dir, f"bulk.seg")),
    )
