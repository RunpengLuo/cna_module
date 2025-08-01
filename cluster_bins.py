import os
import sys
import kneed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
from hmm_models import *
from plot_1d2d import plot_1d2d

# def elbow_bic(minK: int, maxK: int, all_bics: np.ndarray, state_selection: str, outdir: str):
#     if state_selection != "bic":
#         return None, False

#     scores = all_bics[minK : maxK + 1]
#     Ks = [k for k in range(minK, maxK + 1)]
#     kl = kneed.KneeLocator(x=Ks, y=scores, curve="convex", direction="decreasing")

#     elbow_x, elbow_y = kl.elbow, kl.elbow_y
#     kl.plot_knee(
#         title=f"{state_selection} Curve",
#         xlabel="K",
#         ylabel=f"{state_selection}",
#     )
#     plt.savefig(os.path.join(outdir, f"{state_selection}-curve.png"), dpi=300)
#     return None, False


if __name__ == "__main__":
    _, run_dir, out_dir = sys.argv
    os.makedirs(out_dir, exist_ok=True)
    print("Working directory:", run_dir)
    print("Output directory:", out_dir)
    rdr_dir = os.path.join(run_dir, "rdr")
    bb_dir = os.path.join(run_dir, "bb")

    # sample_file = os.path.join(rdr_dir, "sample_ids.tsv")
    bin_pfile = os.path.join(bb_dir, "bin_position.tsv")
    baf_mfile = os.path.join(bb_dir, "bin_matrix.baf.npz")
    rdr_mfile = os.path.join(bb_dir, "bin_matrix.rdr_corr.npz")

    bins = pd.read_table(bin_pfile, sep="\t")
    bafs = np.load(baf_mfile)["mat"].astype(np.float64)
    rdrs = np.load(rdr_mfile)["mat"].astype(np.float64)

    # assign region id for consecutive SNPs
    same_chr = bins["#CHR"] == bins["#CHR"].shift()
    adj_pos = bins["START"] == bins["END"].shift()
    new_cluster = ~(same_chr & adj_pos)
    bins["region_id"] = new_cluster.cumsum()
    region_ids = bins["region_id"].unique()

    # hmm input
    X_lengths = bins.groupby(by="region_id", sort=False).agg("size").to_numpy()
    X_bafs = bafs[:, 1:]
    X_rdrs = rdrs[:, :]
    X = np.concatenate([X_bafs, X_rdrs], axis=1)
    nsamples = X_rdrs.shape[1]
    print(X.shape, X_lengths.shape, X[:5])

    tau = 1e-12
    minK = 2
    maxK = 8
    restarts = 10

    # hmm output
    all_labels = np.zeros((maxK + 1, X.shape[0]), dtype=np.int32)
    all_bics = np.full(maxK + 1, fill_value=np.inf)
    all_models = []
    expected_rdrs = []
    expected_bafs = []

    for K in range(minK, maxK + 1):
        my_best_ll = -1 * np.inf
        my_best_labels = None
        my_best_model = None
        for s in range(restarts):
            A = make_transmat(1 - tau, K)
            assert np.all(A > 0), "unstable tau"
            model = DiagGMMHMM(
                nsamples=nsamples,
                n_components=K,
                n_mix=2,
                init_params="mcw",
                params="smctw",
                covariance_type="diag",
                random_state=s,
            )
            # model = DiagGHMM(
            #     n_components=K,
            #     init_params="mc",
            #     params="smct",
            #     covariance_type="diag",
            #     random_state=s,
            # )
            model.startprob_ = np.ones(K) / K  # uniform start probabilities
            model.weights_ = np.ones((K, 2)) * 0.5  # uniform mixture weights
            model.transmat_ = A

            model.fit(X, X_lengths)
            prob, labels = model.decode(X, X_lengths, algorithm="map")
            if prob > my_best_ll:
                my_best_labels = labels
                my_best_ll = prob
                my_best_model = model
        
        all_labels[K, :] = my_best_labels.astype(np.int32)
        all_models.append(my_best_model)
        all_bics[K] = my_best_model.bic(X)
        
        # compute expected RDR and BAF for each K
        expected_rdr = np.zeros((K, nsamples), dtype=np.float64)
        expected_baf = np.zeros((K, nsamples + 1), dtype=np.float64)
        for k in range(K):
            expected_rdr[k, :] = np.mean(rdrs[all_labels[K, :] == k, :], axis=0)
            expected_baf[k, :] = np.mean(bafs[all_labels[K, :] == k, :], axis=0)
        expected_rdrs.append(expected_rdr)
        expected_bafs.append(expected_baf)
        plot_1d2d(run_dir, out_dir, f"K{K}_", False, all_labels[K], expected_rdrs[K - minK], expected_bafs[K - minK])

    opt_K = np.argmin(all_bics)
    opt_model = all_models[opt_K - minK]
    opt_labels = all_labels[opt_K]
    print("Optimal K:", opt_K)
    print("Optimal BIC:", all_bics[opt_K])
    print("BICs:", all_bics[minK:maxK + 1])

    # elbow_bic(minK, maxK, all_bics, "bic", out_dir)
    # for K in range(minK, opt_K + 1):
    #     plot_1d2d(run_dir, out_dir, f"K{K}_", False, all_labels[K], expected_rdrs[K - minK], expected_bafs[K - minK])
