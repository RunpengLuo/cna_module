import os
import sys
from collections import Counter
import kneed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
from plot_utils import plot_1d2d
from constrained_gmmhmm import *

if __name__ == "__main__":
    ##################################################
    args = sys.argv
    print(args)
    _, bb_dir, out_dir = args[:3]
    
    # hyper-parameter
    baf_tol = 0.02 # collapse to 0.5 if estimated BAF around 0.5-baf_tol and 0.5+baf_tol

    # model-parameters
    tau = 1e-12
    minK = 2
    maxK = 10
    restarts = 10
    n_iter = 10

    os.makedirs(out_dir, exist_ok=True)
    plot_dir = os.path.join(out_dir, "plots")
    bbc_dir = os.path.join(out_dir, "labels")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(bbc_dir, exist_ok=True)

    bb_file = os.path.join(bb_dir, "bulk.bb")
    bin_pfile = os.path.join(bb_dir, "bin_info.tsv.gz")
    baf_mfile = os.path.join(bb_dir, "bin_matrix.baf.npz")
    rdr_mfile = os.path.join(bb_dir, "bin_matrix.rdr.npz")

    ##################################################
    print("load arguments")
    bins = pd.read_table(bin_pfile, sep="\t")
    bafs = np.load(baf_mfile)["mat"].astype(np.float64)
    rdrs = np.load(rdr_mfile)["mat"].astype(np.float64)
    (nsegments, nsamples) = rdrs.shape
    assert len(bins) == nsegments, f"unmatched {len(bins)} and {nsegments}"

    ##################################################
    print("prepare HMM inputs")
    X_lengths = bins.groupby(by="region_id", sort=False).agg("size").to_numpy()
    X_bafs = bafs[:, 1:]  # ignore normal
    X_rdrs = rdrs[:, :]
    X = np.concatenate([X_bafs, X_rdrs], axis=1)

    _nsegments = np.sum(X_lengths)
    assert _nsegments == nsegments, f"unmatched {_nsegments} and {nsegments}"
    print(f"#segments={nsegments}")

    mhbafs = split_X(mirror_baf(X, nsamples), nsamples)[0]

    X_in = X
    X_in_init = mirror_baf(X_in, nsamples)
    # estimate global-BAF-variance from normal
    est_baf_var = np.var(bafs[:, :nsamples], ddof=1, axis=0)
    print(f"estimated normal BAF variance {est_baf_var.round(3)}")

    # TODO RDR is not gaussian shape, and also has variance propto mean
    # proper standardization may needed to balance between baf and rdr
    # in likelihood contribution
    # print(f"RDR mixture variance={np.var(X_rdrs, axis=0)}")
    # rdr_scale = np.sqrt(est_baf_var / np.var(X_rdrs, axis=0))
    # print(f"rdr scale={rdr_scale}")

    # X_rdrs_rescaled = X_rdrs * rdr_scale
    # X_in = np.concatenate([X_bafs, X_rdrs_rescaled], axis=1)
    # X_in_init = mirror_baf(X_in, nsamples)

    # X_scaler, X_in = standardize_data(X, nsamples)
    # X_in_init = X_in[np.mean(X[:, :nsamples], axis=1) <= 0.5]

    # X_in[:, nsamples:] = np.log2(X_in[:, nsamples:])
    # log-transform RDR to stablize variance

    # priors
    baf_weight = 1.0
    rdr_weight = 1.0
    means_prior = [0.5, 1.0]
    means_weight = [1e-2, 1e-2]

    rdr_alpha = 1e-3
    rdr_beta = 1e-3
    prior_baf_mode = np.mean(est_baf_var)
    baf_alpha = 5
    baf_beta = prior_baf_mode * (baf_alpha + 1)

    covars_alpha = [baf_alpha, rdr_alpha]
    covars_beta = [baf_beta, rdr_beta]
    print(f"inverse-gamma covars_prior: alpha={covars_alpha}, beta={covars_beta}")


    # hmm output
    all_labels = np.zeros((maxK + 1, nsegments), dtype=np.int32)
    all_bics = np.full(maxK + 1, fill_value=np.inf)
    all_lls = np.full((maxK + 1, restarts), fill_value=-np.inf)

    all_models = []
    uniq_labels = {}
    expected_rdrs = {}  # each has K values
    expected_bafs = {}
    centroid_bafs = {}

    for K in range(minK, maxK + 1):
        print(f"running HMM on K={K}")
        best_ll = -1 * np.inf
        best_model = {"model": None, "labels": None, "means": None, "init_means": None, "init_covs": None}
        for s in range(restarts):
            A = make_transmat(1 - tau, K)
            assert np.all(A > 0), "unstable tau"
            model = CONSTRAINED_GMMHMM(
                bafdim=nsamples,
                baf_weight=baf_weight,
                rdr_weight=rdr_weight,
                n_components=K,
                init_params="mcw",
                params="smct",
                # params="smctw",
                random_state=s,
                means_weight=means_weight,
                means_prior=means_prior,
                covars_alpha=covars_alpha,
                covars_beta=covars_beta,
                n_iter=n_iter,
            )
            model.startprob_ = np.ones(K) / K  # s
            model.transmat_ = A # t
            model._init(X_in_init, None)
            model.init_params = ""
            init_means = np.copy(model.means_)
            init_covs = np.copy(model.covars_)
            model.fit(X_in, X_lengths)
            if not model.monitor_.converged:
                print(f"warning, model is not coverged K={K}, restart={s}")
            # Viterbi decoding
            ll, labels = model.decode(X_in, X_lengths, algorithm="map")
            all_lls[K, s] = ll
            if ll > best_ll:
                best_ll = ll
                best_model["model"] = model
                best_model["labels"] = labels
                best_model["init_means"] = init_means
                best_model["init_covs"] = init_covs

        print(f"ll={best_ll}")
        model = best_model["model"]
        transmat = model.transmat_
        raw_labels = best_model["labels"]
        raw_init_means = best_model["init_means"]
        raw_init_covs = best_model["init_covs"]
        raw_means = model.means_
        raw_covars = model.covars_
        weights = model.weights_

        # segment_lengths, num_bps = count_breakpoints(raw_labels, X_lengths)
        # print(f"uniq_labels={len(np.unique(raw_labels))}")
        # print(f"#breakpoints={num_bps}")
        # print(f"mean-segment-length={np.mean(segment_lengths):.3f}")
        # print(f"median-segment-length={np.median(segment_lengths):.3f}")

        # print(X_scaler.mean_)
        # print(raw_means)
        # means = raw_means * X_scaler.scale_ + X_scaler.mean_
        # means[:, :nsamples] += 0.5
        # covars = raw_covars * X_scaler.scale_**2
        # means[:, nsamples:] = np.exp2(means[:, nsamples:])

        means = raw_means
        covars = raw_covars

        # remap labels
        j = 1
        k2label = {}
        label2k = {}
        for i, _ in Counter(raw_labels).most_common():
            k2label[i] = j
            label2k[j] = i
            j += 1
        labels = list(k2label.values())
        uniq_labels[K] = labels
        all_labels[K, :] = np.array([k2label[v] for v in raw_labels])
        all_models.append(model)
        all_bics[K] = model.bic(X_in)

        for k in range(len(labels)):
            label = labels[k]
            print(labels[k], np.sum(raw_labels == label2k[label]), weights[label2k[label], :])
            # print(raw_means[label2k[label], :], raw_covars[label2k[label], :])
            print(raw_init_means[label2k[label], :], raw_init_covs[label2k[label], :])
            print(means[label2k[label], :], covars[label2k[label], :])
            for k2 in range(len(labels)):
                print("\t->", labels[k2], transmat[label2k[label], label2k[labels[k2]]].round(5))
        # compute expected RDR and BAF for each K
        expected_rdr_mean = np.zeros((len(labels), nsamples), dtype=np.float64)
        expected_rdr_std = np.zeros((len(labels), nsamples), dtype=np.float64)
        expected_baf_mean = np.zeros((len(labels), nsamples + 1), dtype=np.float64)
        expected_baf_std = np.zeros((len(labels), nsamples + 1), dtype=np.float64)
        for k in range(len(labels)):
            label = labels[k]
            label_mask = all_labels[K, :] == label
            expected_rdr_mean[k, :] = np.mean(rdrs[label_mask, :], axis=0)
            expected_rdr_std[k, :] = np.std(rdrs[label_mask, :], axis=0)

            baf_means = means[label2k[label], :nsamples]
            if np.mean(np.abs(baf_means - 0.5)) <= baf_tol:
                # allelic balanced cluster, set BAF=0.5
                expected_baf_mean[k, 1:] = 0.5
            else:
                # MLE of BAF mean
                expected_baf_mean[k, 1:] = np.mean(mhbafs[label_mask, :], axis=0)
            baf_vars = np.mean((mhbafs[label_mask, :] - expected_baf_mean[k, 1:]) ** 2, axis=0)
            expected_baf_std[k, 1:] = np.sqrt(baf_vars)

        expected_rdrs[K] = [expected_rdr_mean, expected_rdr_std]
        expected_bafs[K] = [expected_baf_mean[:, 1:], expected_baf_std[:, 1:]]
    
        plot_1d2d(
            bb_dir=bb_dir, 
            out_dir=plot_dir, 
            out_prefix=f"K{K}_", 
            plot_normal=False,
            clusters=all_labels[K], 
            expected_rdrs=expected_rdr_mean, 
            expected_bafs=expected_baf_mean,
            fitted_means=means,
            fitted_covs=covars
        )

    opt_K = np.argmin(all_bics)
    opt_model = all_models[opt_K - minK]
    opt_labels = all_labels[opt_K]
    print("Optimal K:", opt_K)
    print("Optimal BIC:", all_bics[opt_K])
    print("BICs:", all_bics[minK : maxK + 1])

    bb = pd.read_csv(bb_file, sep="\t")
    samples = bb["SAMPLE"].unique()

    # save mhBAF
    for i, sample in enumerate(samples):
        bb.loc[bb["SAMPLE"] == sample, "BAF"] = mhbafs[:, i]

    bb["CLUSTER"] = 0
    # save to bulk.bbc
    for K in range(minK, maxK + 1):
        if opt_K == K:
            bbc_file = os.path.join(out_dir, f"bulk.bbc")
            seg_file = os.path.join(out_dir, f"bulk.seg")
        else:
            bbc_file = os.path.join(bbc_dir, f"bulk{K}.bbc")
            seg_file = os.path.join(bbc_dir, f"bulk{K}.seg")        
        bb["CLUSTER"] = np.repeat(all_labels[K], nsamples)
        labels = uniq_labels[K]
        seg_rows = []
        bb_grps = bb.groupby(by="CLUSTER", sort=False)
        [bb_bafs_means, bb_bafs_stds] = expected_bafs[K]    
        [bb_rdrs_means, bb_rdrs_stds] = expected_rdrs[K]
        for l, label in enumerate(labels):
            bb_grp = bb_grps.get_group(label)
            for s, sample in enumerate(samples):
                bb_sample = bb_grp.loc[bb_grp["SAMPLE"] == sample, :]

                # seg_rows.append([label, sample, len(bb_sample), bb_rdrs[l, s], 
                #                  bb_sample["#SNPS"].sum(), bb_sample["COV"].mean(), 
                #                  bb_sample["ALPHA"].sum(), bb_sample["BETA"].sum(),
                #                  bb_bafs[l, s]])
                seg_nsnps = bb_sample["#SNPS"].sum()
                ave_cov = np.sum(bb_sample["COV"].to_numpy() * bb_sample["#SNPS"].to_numpy()) / seg_nsnps
                seg_rows.append([label, sample, len(bb_sample), bb_sample["#SNPS"].sum(), ave_cov,
                                 bb_bafs_means[l, s], bb_bafs_stds[l, s], bb_rdrs_means[l, s], bb_rdrs_stds[l, s]
                                 ])
        seg = pd.DataFrame(
            data=seg_rows,
            # columns=["#ID", "SAMPLE", "#BINS", "RD", "#SNPS", "COV", "ALPHA", "BETA", "BAF"]
            columns=["#ID", "SAMPLE", "#BINS", "#SNPS", "COV", "BAF", "BAF-std", "RD", "RD-std"]
        )
        bb.to_csv(bbc_file, sep="\t", header=True, index=False)
        seg.to_csv(seg_file, sep="\t", header=True, index=False)
