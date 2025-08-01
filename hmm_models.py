import numpy as np
from scipy.special import logsumexp
from hmmlearn.hmm import GMMHMM, GaussianHMM
import matplotlib.pyplot as plt

def make_transmat(diag, K):
    offdiag = (1 - diag) / (K - 1)
    transmat_ = np.diag([diag - offdiag] * K)
    transmat_ += offdiag
    return transmat_

def plot_likelihoods(ll_histories: list, out_file: str, info=""):
    fig = plt.figure(figsize=(8, 6))
    for i, history in enumerate(ll_histories):
        plt.plot(history, marker="o", label=f"Restart {i+1}", alpha=0.7)
    plt.xlabel("Iteration")
    plt.ylabel("Log Likelihood")
    plt.title(f"HMM Convergence Plot {info}")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)
    return


class DiagGMMHMM(GMMHMM):
    def __init__(self, ncol_baf=None, **kwargs):
        super().__init__(**kwargs)
        self.ncol_baf = ncol_baf

    def _accumulate_sufficient_statistics(
        self, stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice
    ):
        super()._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice
        )

        if "t" in self.params:
            # for each ij, recover sum_t xi_ij from the inferred transition matrix
            bothlattice = fwdlattice + bwdlattice
            loggamma = (bothlattice.T - logsumexp(bothlattice, axis=1)).T

            # denominator for each ij is the sum of gammas over i
            denoms = np.sum(np.exp(loggamma), axis=0)
            # transpose to perform row-wise multiplication
            stats["denoms"] = denoms

    def _do_mstep(self, stats):
        super()._do_mstep(stats)
        if "t" in self.params:
            denoms = stats["denoms"]
            x = (self.transmat_.T * denoms).T

            # numerator is the sum of ii elements
            num = np.sum(np.diag(x))
            # denominator is the sum of all elements
            denom = np.sum(x)

            # (this is the same as sum_i gamma_i)
            # assert np.isclose(denom, np.sum(denoms))

            stats["diag"] = num / denom
            # print(num.shape)
            # print(denom.shape)

            self.transmat_ = self.form_transition_matrix(stats["diag"])
        if "m" in self.params:
            for state in range(self.n_components):
                if self.n_mix != 2:
                    continue  # only implemented for n_mix=2

                # average and reflect the BAF dimension
                m1 = self.means_[state, 0, : self.ncol_baf]  # mean for mix1 BAF
                m2 = self.means_[state, 1, : self.ncol_baf]  # mean for mix2 BAF
                avg = (m1 + (1 - m2)) / 2  # solve: m2 = 1 - m1

                self.means_[state, 0, : self.ncol_baf] = avg
                self.means_[state, 1, : self.ncol_baf] = 1 - avg

    def form_transition_matrix(self, diag):
        tol = 1e-10
        diag = np.clip(diag, tol, 1 - tol)

        offdiag = (1 - diag) / (self.n_components - 1)
        transmat_ = np.diag([diag - offdiag] * self.n_components)
        transmat_ += offdiag
        # assert np.all(transmat_ > 0), (diag, offdiag, transmat_)
        return transmat_

# original HATCHet2 cluster_bins HMM clustering
class DiagGHMM(GaussianHMM):
    def _accumulate_sufficient_statistics(
        self, stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice
    ):
        super()._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice
        )

        if "t" in self.params:
            # for each ij, recover sum_t xi_ij from the inferred transition matrix
            bothlattice = fwdlattice + bwdlattice
            loggamma = (bothlattice.T - logsumexp(bothlattice, axis=1)).T

            # denominator for each ij is the sum of gammas over i
            denoms = np.sum(np.exp(loggamma), axis=0)
            # transpose to perform row-wise multiplication
            stats["denoms"] = denoms

    def _do_mstep(self, stats):
        super()._do_mstep(stats)
        if "t" in self.params:
            denoms = stats["denoms"]
            x = (self.transmat_.T * denoms).T

            # numerator is the sum of ii elements
            num = np.sum(np.diag(x))
            # denominator is the sum of all elements
            denom = np.sum(x)

            # (this is the same as sum_i gamma_i)
            # assert np.isclose(denom, np.sum(denoms))

            stats["diag"] = num / denom
            # print(num.shape)
            # print(denom.shape)

            self.transmat_ = self.form_transition_matrix(stats["diag"])

    def form_transition_matrix(self, diag):
        tol = 1e-10
        diag = np.clip(diag, tol, 1 - tol)

        offdiag = (1 - diag) / (self.n_components - 1)
        transmat_ = np.diag([diag - offdiag] * self.n_components)
        transmat_ += offdiag
        # assert np.all(transmat_ > 0), (diag, offdiag, transmat_)
        return transmat_
