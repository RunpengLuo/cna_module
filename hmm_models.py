import numpy as np
import pandas as pd
from scipy.special import logsumexp

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import cluster
from hmmlearn.hmm import GMMHMM, GaussianHMM
from hmmlearn.base import BaseHMM, _AbstractHMM
from hmmlearn.stats import log_multivariate_normal_density

def make_transmat(diag, K):
    offdiag = (1 - diag) / (K - 1)
    transmat_ = np.diag([diag - offdiag] * K)
    transmat_ += offdiag
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
