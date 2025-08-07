import os
import sys
from collections import Counter
import kneed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
from hmm_models import *
from plot_utils import plot_1d2d
from hmm_models2 import MIXBAFHMM, mirror_baf, split_X

if __name__ == "__main__":
    ##################################################
    args = sys.argv
    print(args)
    _, bb_dir, out_dir = args[:3]

    tau = 1e-12
    minK = 2
    maxK = 8
    restarts = 10

    # model_type = "unmirrored"
    model_type = "mirrored"
    # model_type = "gaussianHMM"
    print(f"model={model_type}")

    os.makedirs(out_dir, exist_ok=True)
    plot_dir = os.path.join(out_dir, "plots")
    bbc_dir = os.path.join(out_dir, "labels")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(bbc_dir, exist_ok=True)

    bb_file = os.path.join(bb_dir, "bulk.bb")
    bin_pfile = os.path.join(bb_dir, "bin_position.tsv.gz")
    baf_mfile = os.path.join(bb_dir, "bin_matrix.baf.npz")
    rdr_mfile = os.path.join(bb_dir, "bin_matrix.rdr_corr.npz")

    ##################################################
    print("load arguments")
    bins = pd.read_table(bin_pfile, sep="\t")
    bafs = np.load(baf_mfile)["mat"].astype(np.float64)
    rdrs = np.load(rdr_mfile)["mat"].astype(np.float64)
    (nsegments, nsamples) = rdrs.shape
    assert len(bins) == nsegments, f"unmatched {len(bins)} and {nsegments}"
    # assign region id for consecutive segments
    same_chr = bins["#CHR"] == bins["#CHR"].shift()
    adj_pos = bins["START"] == bins["END"].shift()
    new_cluster = ~(same_chr & adj_pos)
    bins["region_id"] = new_cluster.cumsum()

    ##################################################
    print("prepare HMM inputs")
    X_lengths = bins.groupby(by="region_id", sort=False).agg("size").to_numpy()
    X_bafs = bafs[:, 1:]  # ignore normal
    X_rdrs = rdrs[:, :]
    X = np.concatenate([X_bafs, X_rdrs], axis=1)
    mhbafs = split_X(mirror_baf(X, nsamples), nsamples)[0]

    _nsegments = np.sum(X_lengths)
    assert _nsegments == nsegments, f"unmatched {_nsegments} and {nsegments}"
    print(f"#segments={nsegments}")

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
        my_best_ll = -1 * np.inf
        my_best_labels = None
        my_best_model = None
        my_best_means = None
        for s in range(restarts):
            A = make_transmat(1 - tau, K)
            assert np.all(A > 0), "unstable tau"
            if model_type == "mirrored":
                model = DiagGHMM(
                    n_components=K,
                    init_params="mc",
                    params="smct",
                    covariance_type="diag",
                    # covariance_type="full",
                    random_state=s,
                )
            elif model_type == "unmirrored":
                model = MIXBAFHMM(
                        bafdim=nsamples,
                        n_components=K,
                        init_params="mc",
                        params="smct",
                        random_state=s
                )
            elif model_type == "gaussianHMM":
                model = GaussianHMM(
                    n_components=K,
                    init_params="mc",
                    params="smct",
                    covariance_type="diag",
                    random_state=s
                )
            else:
                raise NotImplementedError(f"{model_type}")
            model.startprob_ = np.ones(K) / K  # s
            # model.transmat_ = A # t

            transmat = np.full((K, K), 1e-4)
            np.fill_diagonal(transmat, 1 - (K - 1) * 1e-4)
            model.transmat_ = transmat

            model.fit(X, X_lengths)
            if not model.monitor_.converged:
                print(f"warning, model is not coverged K={K}, restart={s}")

            # Viterbi decoding
            prob, labels = model.decode(X, X_lengths, algorithm="map")
            # print(prob)
            all_lls[K, s] = prob
            if prob > my_best_ll:
                my_best_labels = labels
                my_best_ll = prob
                my_best_model = model
                my_best_means = model.means_

        # remap labels
        j = 1
        k2label = {}
        label2k = {}
        for i, _ in Counter(my_best_labels).most_common():
            k2label[i] = j
            label2k[j] = i
            j += 1
        labels = list(k2label.values())
        uniq_labels[K] = labels
        all_labels[K, :] = np.array([k2label[v] for v in my_best_labels])
        all_models.append(my_best_model)
        all_bics[K] = my_best_model.bic(X)

        print(my_best_means)

        # compute expected RDR and BAF for each K
        expected_rdr = np.zeros((len(labels), nsamples), dtype=np.float64)
        expected_baf = np.zeros((len(labels), nsamples + 1), dtype=np.float64)
        centroid_baf = np.zeros((len(labels), nsamples + 1), dtype=np.float64)
        for k in range(len(labels)):
            label = labels[k]
            expected_rdr[k, :] = np.mean(rdrs[all_labels[K, :] == label, :], axis=0)
            # enforce mhBAF constraints here
            my_mhbafs = mhbafs[all_labels[K, :] == label, :]
            expected_baf[k, 1:] = np.mean(my_mhbafs, axis=0)
            # print(label2k[label], label2k)
            centroid_baf[k, 1:] = my_best_means[label2k[label], :nsamples]

        expected_rdrs[K] = expected_rdr
        expected_bafs[K] = expected_baf
        centroid_bafs[K] = centroid_baf
    
        plot_1d2d(
            bb_dir, plot_dir, f"K{K}_", False, all_labels[K], expected_rdr, expected_baf
        )

    opt_K = np.argmin(all_bics)
    opt_model = all_models[opt_K - minK]
    opt_labels = all_labels[opt_K]
    print("Optimal K:", opt_K)
    print("Optimal BIC:", all_bics[opt_K])
    print("BICs:", all_bics[minK : maxK + 1])

    bb = pd.read_csv(bb_file, sep="\t")
    samples = bb["SAMPLE"].unique()
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
        bb_bafs = expected_bafs[K][:, 1:]
        bb_rdrs = expected_rdrs[K]
        for l, label in enumerate(labels):
            bb_grp = bb_grps.get_group(label)
            for s, sample in enumerate(samples):
                bb_sample = bb_grp.loc[bb_grp["SAMPLE"] == sample, :]
                seg_rows.append([label, sample, len(bb_sample), bb_rdrs[l, s], 
                                 bb_sample["#SNPS"].sum(), bb_sample["COV"].mean(), 
                                 bb_sample["ALPHA"].sum(), bb_sample["BETA"].sum(),
                                 bb_bafs[l, s]])
        seg = pd.DataFrame(
            data=seg_rows,
            columns=["#ID", "SAMPLE", "#BINS", "RD", "#SNPS", "COV", "ALPHA", "BETA", "BAF"]
        )
        bb.to_csv(bbc_file, sep="\t", header=True, index=False)
        seg.to_csv(seg_file, sep="\t", header=True, index=False)
    # we can compute cluster variance here,

    # elbow_bic(minK, maxK, all_bics, "bic", out_dir)
