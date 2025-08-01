import os
import sys
import kneed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
from hmm_models import *
from plot_utils import plot_1d2d

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
    ##################################################
    args = sys.argv
    print(args)
    _, bb_dir, out_dir = args[:3]

    tau = 1e-12
    minK = 2
    maxK = 8
    restarts = 10
    model_type = ""

    os.makedirs(out_dir, exist_ok=True)
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
    _nsegments = np.sum(X_lengths)
    assert _nsegments == nsegments, f"unmatched {_nsegments} and {nsegments}"
    print(f"#segments={nsegments}")

    # hmm output
    all_labels = np.zeros((maxK + 1, nsegments), dtype=np.int32)
    all_bics = np.full(maxK + 1, fill_value=np.inf)
    all_lls = np.full((maxK + 1, restarts), fill_value=-np.inf)
    all_models = []
    expected_rdrs = []  # each has K values
    expected_bafs = []

    for K in range(minK, maxK + 1):
        print(f"running HMM on K={K}")
        my_best_ll = -1 * np.inf
        my_best_labels = None
        my_best_model = None
        ll_histories = []
        for s in range(restarts):
            A = make_transmat(1 - tau, K)
            assert np.all(A > 0), "unstable tau"

            # model = DiagGMMHMM(
            #     nsamples=nsamples,
            #     n_components=K,
            #     n_mix=2,
            #     init_params="mcw",
            #     params="smctw",
            #     covariance_type="diag",
            #     random_state=s,
            # )
            model = DiagGHMM(
                n_components=K,
                init_params="mc",
                params="smct",
                covariance_type="diag",
                random_state=s,
            )

            # init model
            model.startprob_ = np.ones(K) / K  # s
            # model.weights_ = np.ones((K, 2)) * 0.5  # uniform mixture weights
            model.transmat_ = A # t

            # Baum-welch learning
            model.fit(X, X_lengths)

            ll_histories.append(list(model.monitor_.history))
            if not model.monitor_.converged:
                print(f"warning, model is not coverged K={K}, restart={s}")

            # Viterbi decoding
            prob, labels = model.decode(X, X_lengths, algorithm="map")
            all_lls[K, s] = prob
            if prob > my_best_ll:
                my_best_labels = labels
                my_best_ll = prob
                my_best_model = model

        # plot likelihoods
        out_ll_file = os.path.join(out_dir, f"LL-{K}.png")
        plot_likelihoods(ll_histories, out_ll_file, f"K={K}")

        all_labels[K, :] = my_best_labels.astype(np.int32)
        all_models.append(my_best_model)
        all_bics[K] = my_best_model.bic(X)

        # if mhBAF is not imposed during binning, we can do it here TODO
        # compute expected RDR and BAF for each K
        expected_rdr = np.zeros((K, nsamples), dtype=np.float64)
        expected_baf = np.zeros((K, nsamples + 1), dtype=np.float64)
        for k in range(K):
            expected_rdr[k, :] = np.mean(rdrs[all_labels[K, :] == k, :], axis=0)
            expected_baf[k, :] = np.mean(bafs[all_labels[K, :] == k, :], axis=0)

        expected_rdrs.append(expected_rdr)
        expected_bafs.append(expected_baf)
        plot_1d2d(
            bb_dir, out_dir, f"K{K}_", True, all_labels[K], expected_rdr, expected_baf
        )

    opt_K = np.argmin(all_bics)
    opt_model = all_models[opt_K - minK]
    opt_labels = all_labels[opt_K]
    print("Optimal K:", opt_K)
    print("Optimal BIC:", all_bics[opt_K])
    print("BICs:", all_bics[minK : maxK + 1])

    # we can compute cluster variance here,

    # elbow_bic(minK, maxK, all_bics, "bic", out_dir)
