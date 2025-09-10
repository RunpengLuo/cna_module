import os
import sys
from collections import Counter
import kneed
import numpy as np
import pandas as pd
from utils import *
from plot_utils import plot_1d2d
from constrained_gmmhmm import *

if __name__ == "__main__":
    ##################################################
    args = sys.argv
    print(args)
    _, bb_dir, out_dir = args[:3]

    # hyper-parameter
    baf_tol = 0.01  # collapse to 0.5 if estimated BAF deviation within 0.5-baf_tol and 0.5+baf_tol

    # model-parameters
    tau = 1e-12
    minK = 2
    maxK = 10
    restarts = 10
    n_iter = 10
    decode_algo = "viterbi"
    verbose = 1

    # "smctw" "smct"
    hmm_params = "smc"
    diag_transmat = True

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
    normal_baf_variance = np.var(bafs[:, 0], ddof=1)
    print(f"estimated normal BAF variance {normal_baf_variance.round(3)}")

    # priors
    means_prior = [0.5, 1.0]
    means_weight = [1e-2, 1e-2]

    rdr_alpha = 1e-3
    rdr_beta = 1e-3
    baf_alpha = 5
    prior_baf_mode = normal_baf_variance
    baf_beta = prior_baf_mode * (baf_alpha + 1)

    covars_alpha = [baf_alpha, rdr_alpha]
    covars_beta = [baf_beta, rdr_beta]
    print(f"inverse-gamma covars_prior: alpha={covars_alpha}, beta={covars_beta}")

    # hmm output
    all_labels = np.zeros((maxK + 1, nsegments), dtype=np.int32)
    all_bics = np.full(maxK + 1, fill_value=np.inf)
    all_model_lls = np.full((maxK + 1, restarts), fill_value=-np.inf)
    all_decode_lls = np.full((maxK + 1, restarts), fill_value=-np.inf)

    all_models = []
    uniq_labels = {}
    expected_rdrs = {}  # each has K values
    expected_bafs = {}
    centroid_bafs = {}

    for K in range(minK, maxK + 1):
        print("=========================")
        print(f"running HMM on K={K}")
        best_model = {
            "model": None,
            "labels": None,
            "means": None,
            "model_ll": -1 * np.inf,
            "decode_ll": -1 * np.inf,
        }
        for s in range(restarts):
            A = make_transmat(1 - tau, K)
            assert np.all(A > 0), "unstable tau"
            model = CONSTRAINED_GMMHMM(
                bafdim=nsamples,
                n_components=K,
                init_params="mcw",
                params=hmm_params,
                random_state=s,
                means_weight=means_weight,
                means_prior=means_prior,
                covars_alpha=covars_alpha,
                covars_beta=covars_beta,
                n_iter=n_iter,
                diag_transmat=diag_transmat,
            )
            model.startprob_ = np.ones(K) / K  # s
            model.transmat_ = A  # t
            model._init(X_in_init, None)
            model.init_params = ""
            model.fit(X_in, X_lengths)
            if not model.monitor_.converged:
                print(f"warning, model is not coverged K={K}, restart={s}")

            model_ll = model.score(X_in, X_lengths)
            decode_ll, labels = model.decode_labels(
                X_in, X_lengths, A, decode_algo=decode_algo
            )
            all_model_lls[K, s] = model_ll
            all_decode_lls[K, s] = decode_ll

            if verbose:
                segment_lengths, num_bps = count_breakpoints(labels, X_lengths)
                print(f"\ts={s} {model_ll:.3f} {decode_ll:.3f} #switches={num_bps}")
            if decode_ll > best_model["decode_ll"]:
                best_model["model_ll"] = model_ll
                best_model["decode_ll"] = decode_ll
                best_model["model"] = model
                best_model["labels"] = labels

        model = best_model["model"]
        transmat = model.transmat_
        raw_labels = best_model["labels"]
        means = model.means_
        covars = model.covars_
        weights = model.weights_
        model_ll = best_model["model_ll"]
        decode_ll = best_model["decode_ll"]
        print(f"best model ll={model_ll} decode ll={decode_ll}")

        # remap labels to 1,...,K
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

        if verbose:
            for k in range(len(labels)):
                label = labels[k]
                print(
                    labels[k],
                    np.sum(raw_labels == label2k[label]),
                    weights[label2k[label], :],
                )
                print(means[label2k[label], :], covars[label2k[label], :])
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
                exp_baf_mean = 0.5
                exp_baf_std = np.std(X_bafs[label_mask, :], axis=0)
            else:
                # MLE of BAF mean
                exp_baf_mean = np.mean(mhbafs[label_mask, :], axis=0)
                exp_baf_std = np.std(mhbafs[label_mask, :], axis=0)
            expected_baf_mean[k, 1:] = exp_baf_mean
            expected_baf_std[k, 1:] = exp_baf_std

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
            fitted_covs=covars,
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
                        bb_bafs_means[l, s],
                        bb_bafs_stds[l, s],
                        bb_rdrs_means[l, s],
                        bb_rdrs_stds[l, s],
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
                "BAF-std",
                "RD",
                "RD-std",
            ],
        )
        bb.to_csv(bbc_file, sep="\t", header=True, index=False)
        seg.to_csv(seg_file, sep="\t", header=True, index=False)
