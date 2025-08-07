import sys
import numpy as np
import pandas as pd
from scipy.special import logsumexp

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import cluster
from hmmlearn.hmm import GMMHMM, GaussianHMM
from hmmlearn.base import BaseHMM, _AbstractHMM
from hmmlearn.stats import log_multivariate_normal_density
from hmmlearn.utils import log_normalize


def split_X(X: np.ndarray, bafdim=1):
    bafs = X[:, :bafdim]
    rdrs = X[:, bafdim:]
    return bafs, rdrs


def merge_X(bafs: np.ndarray, rdrs: np.ndarray):
    return np.concatenate([bafs, rdrs], axis=1)


def mirror_baf(X: np.ndarray, bafdim=1):
    bafs, rdrs = split_X(X, bafdim)
    baf_means = np.mean(bafs, axis=1)
    mhbafs = np.where(baf_means[:, None] > 0.5, 1 - bafs, bafs)
    return merge_X(mhbafs, rdrs)


class MIXBAFHMM(BaseHMM):
    """
    HMM model
    observation: RDR, BAF
    emission, gaussian RDR, 2-mixture gaussian BAF
    diagonal covariance, 2-mixture has complement BAF, same covariance, and fixed equal weights

    Attributes
    ----------
    startprob_: (n_components,)
    transmat_ : (n_components, n_components)
    means_ : (n_components, n_features)
    """

    def __init__(
        self,
        bafdim=1,
        n_components=1,
        min_covar=1e-3,
        startprob_prior=1.0,
        transmat_prior=1.0,
        means_prior=0.0,
        means_weight=0.0,
        covars_prior=1e-2, 
        covars_weight=1,
        algorithm="viterbi",
        random_state=None,
        n_iter=10,
        tol=1e-2,
        verbose=False,
        params="stmc",
        init_params="stmc",
        implementation="log",
    ):
        BaseHMM.__init__(
            self,
            n_components,
            startprob_prior=startprob_prior,
            transmat_prior=transmat_prior,
            algorithm=algorithm,
            random_state=random_state,
            n_iter=n_iter,
            tol=tol,
            verbose=verbose,
            params=params,
            init_params=init_params,
            implementation=implementation,
        )
        self.min_covar = min_covar
        self.means_prior = means_prior
        self.means_weight = means_weight
        self.means_prior_nm = np.broadcast_to(means_prior, (self.n_components, 2, bafdim)).copy()
        self.means_weight_nm = np.broadcast_to(means_weight, (self.n_components, 2)).copy()

        self.covars_prior = covars_prior
        self.covars_weight = covars_weight

        self.n_mix = 2
        self.bafdim = bafdim

    def _init(self, X, lengths=None):
        super()._init(X, lengths=None)  # startprobs, transmats
        nc = self.n_components
        nf = self.n_features

        # run k-means++ on the mirrored version
        X_mirrored = mirror_baf(X, self.bafdim)

        # init means (K, D)
        if self._needs_init("m", "means_"):  # means has shape (nc, nm, nf)
            kmeans = cluster.KMeans(n_clusters=self.n_components,
                                    random_state=self.random_state,
                                    n_init=10)  # sklearn <1.4 backcompat.
            kmeans.fit(X_mirrored)
            centroids = kmeans.cluster_centers_
            # balanced_s = np.argmax(np.mean(np.abs(centroids[:, :self.bafdim] - 0.5), axis=1))
            # centroids[balanced_s, :] = 0.5
            self.means_ = centroids
        
        # emission covariance, diagonal (K, D, 1)
        if self._needs_init("c", "covars_"):
            cv = np.cov(X_mirrored.T) + self.min_covar * np.eye(nf) # cov(X) + sig*I
            if not cv.shape:
                cv.shape = (1, 1)
            self.covars_ = np.tile(np.diag(cv), (nc, 1)) 
        
        # print(self.means_)
        # print(self.covars_)
        # sys.exit(0)
        return

    def _get_n_fit_scalars_per_param(self):
        nc = self.n_components
        nf = self.n_features
        return {
            "s": nc - 1,
            "t": nc * (nc - 1),
            "m": nc * nf,
            "c": nc * nf,
        }

    def _check(self): #TODO
        pass

    def _compute_log_weighted_gaussian_densities(self, X, i_comp):
        cur_means = self.means_[i_comp]
        cur_covs = self.covars_[i_comp]
        log_cur_weights = np.log(self.weights_[i_comp])

        return (
            log_multivariate_normal_density(X, cur_means, cur_covs, "diag")
            + log_cur_weights
        )
    
    def _compute_log_likelihood_rdr(self, X):
        """
        compute log-likelihood for RDR
        """
        X_rdrs = split_X(X, self.bafdim)[1]
        rdr_means = split_X(self.means_, self.bafdim)[1]
        rdr_covs = split_X(self.covars_, self.bafdim)[1]
        rdr_covs = np.maximum(rdr_covs, np.finfo(float).tiny)    
        return log_multivariate_normal_density(X_rdrs, rdr_means, rdr_covs, "diag")

    def _compute_log_likelihood_baf(self, X):
        """
        compute log-likelihood for BAF
        """
        X_bafs = split_X(X, self.bafdim)[0]
        baf_means = split_X(self.means_, self.bafdim)[0]
        baf_covs = split_X(self.covars_, self.bafdim)[0]
        baf_covs = np.maximum(baf_covs, np.finfo(float).tiny)

        baf_ll_a = log_multivariate_normal_density(X_bafs, baf_means, baf_covs, "diag") # P(baf|z,w=0)
        baf_ll_b = log_multivariate_normal_density(X_bafs, 1 - baf_means, baf_covs, "diag") # P(baf|z,w=1)
        return baf_ll_a, baf_ll_b

    def _compute_log_likelihood(self, X):
        """
        Compute log-likelihood for BAF mixture HMM
            1. with complement mean for 2-mixture BAF
            2. with standard mean for RDR
        Returns
        log_prob: array, shape (n_samples, n_components)
            Emission log probability of each sample in ``X`` for each of the
            model states, i.e., ``log(p(X|state))``.
        """
        rdr_lls = self._compute_log_likelihood_rdr(X)
        baf_lls_a, baf_lls_b = self._compute_log_likelihood_baf(X)
        baf_lls = np.log(np.exp(baf_lls_a) + np.exp(baf_lls_b)) + np.log(0.5) # P(baf|z) TODO use logsumexp
        # print(X[:5])
        # print(rdr_ll.shape, baf_ll_a.shape)
        # print("RDR", rdr_means, rdr_ll[:5])
        # print()
        # print("BAF", baf_means, baf_ll_a[:5], baf_ll_b[:5])
        # print(baf_ll[:5])
        # sys.exit(0)
        return rdr_lls + baf_lls

    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        stats["post"] = np.zeros(self.n_components) # P(z=k|X;\theta)
        stats['obs'] = np.zeros((self.n_components, self.n_features)) # gamma @ X        
        stats['obs**2'] = np.zeros((self.n_components, self.n_features))

        stats["post_mix"] = np.zeros((self.n_components, 2)) # P(z=k,w=i|X;\theta)
        stats["m_n"] = self.means_weight_nm[:, :, None] * self.means_prior_nm
        return stats

    def _accumulate_sufficient_statistics(
        self, stats, X, lattice, posteriors, fwdlattice, bwdlattice
    ):
        """
        X: (nsamples, nfeatures)
        lattice: (n_samples, n_components) P(X_i|z_i=k)
        posteriors: (n_samples, n_components) P(z=k|X)
        fwdlattice, bwdlattice: (n_samples, n_components), alpha and beta terms
        """
        super()._accumulate_sufficient_statistics(
            stats, X, lattice, posteriors, fwdlattice, bwdlattice
        )

        n_samples, _ = X.shape

        X_bafs = X[:, :self.bafdim]
        baf_lls_a, baf_lls_b = self._compute_log_likelihood_baf(X)
        baf_lls = np.stack((baf_lls_a, baf_lls_b), axis=-1)
        log_normalize(baf_lls, axis=-1)
        with np.errstate(under="ignore"):
            post_mix = np.exp(baf_lls)
            post_comp_mix = posteriors[:, :, None] * post_mix

        stats["post_mix"] += post_comp_mix.sum(axis = 0) # sum over samples
        stats["post"] += posteriors.sum(axis=0)

        if "m" in self.params:
            stats["obs"] += posteriors.T @ X
            stats['m_n'] += np.einsum('ijk,il->jkl', post_comp_mix, X_bafs)
        
        if "c" in self.params:
            stats['obs**2'] += posteriors.T @ X**2

        return

    def _do_mstep(self, stats):
        super()._do_mstep(stats)

        means_prior = self.means_prior
        means_weight = self.means_weight
        covars_prior = self.covars_prior
        covars_weight = self.covars_weight
        bafdim = self.bafdim

        # Maximizing means
        denom = stats["post"][:, None] # (n_components, 1)
        if "m" in self.params:
            rdr_obs = split_X(stats["obs"], bafdim)[1]
            # standard gaussian updates
            # n_components, n_features
            rdr_means = ((means_weight * means_prior + rdr_obs)
                         / (means_weight + denom))

            m_n = stats['m_n']
            m_d = stats['post_mix'] + self.means_weight_nm
            m_d[(m_n == 0).all(axis=-1)] = 1

            # n_components, n_mix, n_features
            baf_means = m_n / m_d[:, :, None]
            baf_means = baf_means[:, 0, :]

            self.means_ = np.concatenate([baf_means, rdr_means], axis=1)
            # print(self.means_)
            # sys.exit(0)

        # Maximizing covariances
        if "c" in self.params:
            meandiff = self.means_ - means_prior
            c_n = (means_weight * meandiff**2
                    + stats['obs**2']
                    - 2 * self.means_ * stats['obs']
                    + self.means_**2 * denom)
            c_d = max(covars_weight - 1, 0) + denom
            self._covars_ = (covars_prior + c_n) / np.maximum(c_d, 1e-5)
