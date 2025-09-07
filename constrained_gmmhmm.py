import sys
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import cluster
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GMMHMM, GaussianHMM
from hmmlearn.base import BaseHMM, _AbstractHMM
from hmmlearn.stats import log_multivariate_normal_density
from hmmlearn.utils import log_normalize


def make_transmat(diag, K):
    offdiag = (1 - diag) / (K - 1)
    transmat_ = np.diag([diag - offdiag] * K)
    transmat_ += offdiag
    return transmat_

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

def standardize_data(X: np.ndarray, bafdim=1):
    X_ = np.copy(X)
    X_[:, :bafdim] -= 0.5 # convert to [-0.5, 0.5]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_)
    return scaler, X_scaled


class CONSTRAINED_GMMHMM(BaseHMM):
    """
    Constrained GMM-HMM model
    """
    def __init__(
        self,
        bafdim=1,
        baf_weight=1.0,
        rdr_weight=1.0,
        n_components=1,
        min_covar=1e-3,
        startprob_prior=1.0,
        transmat_prior=1.0,
        weights_prior=2.0,
        means_prior=[0.5, 1.0],
        means_weight=[1e-3, 1e-3],
        covars_alpha=[1e-3, 1e-3],
        covars_beta=[1e-3, 1e-3],
        algorithm="viterbi",
        random_state=None,
        n_iter=10,
        tol=1e-2,
        verbose=False,
        params="stmc",
        init_params="stmc",
        implementation="log",
        diag_transmat=False,
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

        self.bafdim = bafdim # first <bafdim> features are BAF, other features are RDR
        self.baf_weight = baf_weight
        self.rdr_weight = rdr_weight
        self.min_covar = min_covar
        self.n_mix = 2

        self.diag_transmat = diag_transmat

        # Dir(2,2) prior
        self.weights_prior = weights_prior

        # baf and rdr means prior
        self.means_prior = means_prior
        self.means_weight = means_weight

        # inv-gamma prior
        self.covars_alpha = covars_alpha # alpha
        self.covars_beta = covars_beta # beta

    # TODO better means and covariance init?
    def _init(self, X, lengths=None):
        super()._init(X, lengths=None)  # startprobs, transmats
        nc = self.n_components
        nf = self.n_features
        nm = self.n_mix

        # init weights (K, 2)
        if self._needs_init("w", "weights_"):
            self.weights_ = np.full((nc, nm), 1 / nm)

        # init means (K, D)
        if self._needs_init("m", "means_"):
            kmeans = cluster.KMeans(
                n_clusters=self.n_components,
                random_state=self.random_state,
                init="k-means++",
                n_init="auto",
            )  # sklearn <1.4 backcompat.
            kmeans.fit(X)
            centroids = kmeans.cluster_centers_
            # (n_components, n_features)
            self.means_ = centroids
        
        # emission covariance, diagonal (K, D)
        if self._needs_init("c", "covars_"):
            cv = np.cov(X.T) + self.min_covar * np.eye(nf) # cov(X) + sig*I
            if not cv.shape:
                cv.shape = (1, 1)
            # (n_components, n_features)
            self.covars_ = np.tile(np.diag(cv), (nc, 1)) 
        return

    def _get_n_fit_scalars_per_param(self):
        nc = self.n_components
        nf = self.n_features
        return {
            "s": nc - 1,
            "t": nc * (nc - 1),
            "m": nc * nf,
            "c": nc * nf,
            "w": nc
        }
    
    def _compute_log_weighted_gaussian_densities(self, X, i_comp):
        """
        compute logP(rdr_n,baf_n|z_n=k,c_n=c) fixed k.
        return:
            ll: (nobs, nmix)
        """
        n_samples, _ = X.shape
        log_cur_weights = np.log(self.weights_[i_comp])
        X_bafs, X_rdrs = split_X(X, self.bafdim)
        baf_means, rdr_means = split_X(self.means_[i_comp][None, :], self.bafdim)
        baf_covs, rdr_covs = split_X(self.covars_[i_comp][None, :], self.bafdim)

        rdr_lls = log_multivariate_normal_density(
            X_rdrs, rdr_means, rdr_covs, "diag"
        ) * self.rdr_weight
        baf_lls_a = log_multivariate_normal_density(
            X_bafs, 1 - baf_means, baf_covs, "diag"
        ) * self.baf_weight
        baf_lls_b = log_multivariate_normal_density(
            X_bafs, baf_means, baf_covs, "diag"
        ) * self.baf_weight
        lls = np.zeros((n_samples, self.n_mix))
        # logP(baf,rdr,c_n=0|z_n=k)
        # = logP(baf|z_n=k,c_n=0) + logP(rdr|z_n=k,c_n=0) + P(c_n=0|z_n=k)
        lls[:, 0] = (rdr_lls + baf_lls_a + log_cur_weights[0])[:, 0]

        # logP(baf,rdr,c_n=1|z_n=k)
        # logP(baf|z_n=k,c_n=1) + logP(rdr|z_n=k,c_n=1) + P(c_n=1|z_n=k)
        lls[:, 1] = (rdr_lls + baf_lls_b + log_cur_weights[1])[:, 0]
        return lls

    def _compute_log_likelihood(self, X):
        """
        returns (n_samples, n_components)
        logP(baf,rdr|z_n=k)
        """
        logprobs = np.empty((len(X), self.n_components))  # P(x_i|z_i=k)
        for i in range(self.n_components):
            log_denses = self._compute_log_weighted_gaussian_densities(X, i)
            logprobs[:, i] = logsumexp(log_denses, axis=1)
        return logprobs

    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        
        # gamma_n(k), sum over all observations
        stats["post_sum"] = np.zeros(self.n_components)

        # gamma_n(k, 0) and gamma_n(k, 1), sum over all obversations
        stats["post_mix_sum"] = np.zeros((self.n_components, self.n_mix))

        # numerator for mean updates
        stats["m_n"] = np.zeros((self.n_components, self.n_features))

        # numerator for variance updates
        stats["c_n"] = np.zeros((self.n_components, self.n_features))

        stats["raw_data"] = []


        return stats

    def _accumulate_sufficient_statistics(
        self, stats, X, lattice, posteriors, fwdlattice, bwdlattice
    ):
        """
        posteriors: gamma_n(k), (n_samples, n_components)
        """
        super()._accumulate_sufficient_statistics(
            stats, X, lattice, posteriors, fwdlattice, bwdlattice
        )

        n_samples, _ = X.shape
        bafdim = self.bafdim
        
        # P(c_n=c|z_n=k, X, Y)
        post_mix = np.zeros((n_samples, self.n_components, self.n_mix))
        for k in range(self.n_components):
            log_denses = self._compute_log_weighted_gaussian_densities(X, k)
            log_normalize(log_denses, axis=-1)
            with np.errstate(under="ignore"):
                post_mix[:, k, :] = np.exp(log_denses)


        with np.errstate(under="ignore"):
            # gamma_n(k,0) and gamma_n(k,1)
            # (n_samples, n_components, 2)
            post_comp_mix = posteriors[:, :, None] * post_mix

        # denominators
        stats['post_mix_sum'] += post_comp_mix.sum(axis=0)
        stats['post_sum'] += posteriors.sum(axis=0)

        if "m" in self.params:
            for k in range(self.n_components):
                # BAF features
                for i in range(self.bafdim):
                    num0 = post_comp_mix[:, k, 0] * (1 - X[:, i])
                    num1 = post_comp_mix[:, k, 1] * (X[:, i])
                    stats["m_n"][k, i] += np.sum(num0 + num1)
                # RDR features
                for j in range(self.bafdim, self.n_features):
                    num = posteriors[:, k] * X[:, j]
                    stats["m_n"][k, j] += np.sum(num)
        
        if "c" in self.params:
            stats["raw_data"].append([X, post_comp_mix, posteriors])

        # diagonal transition constraint
        if "t" in self.params and self.diag_transmat:
            # for each ij, recover sum_t xi_ij from the inferred transition matrix
            bothlattice = fwdlattice + bwdlattice
            loggamma = (bothlattice.T - logsumexp(bothlattice, axis=1)).T

            # denominator for each ij is the sum of gammas over i
            denoms = np.sum(np.exp(loggamma), axis=0)
            # transpose to perform row-wise multiplication
            stats["denoms"] = denoms

    def _do_mstep(self, stats):
        super()._do_mstep(stats)
        nf = self.n_features
        nm = self.n_mix

        # print("updates")
        # # print("weights")
        # # print(self.weights_)
        # print("means")
        # print(self.means_)
        # print("covars")
        # print(self.covars_)
        # Maximizaing mixture weights per state, w_k and 1-w_k
        if "w" in self.params:
            alpha = self.weights_prior
            w_n = stats['post_mix_sum'] + (alpha - 1)
            w_d = (stats['post_sum'] + 2 * (alpha - 1))[:, None]
            self.weights_ = w_n / w_d

        if "m" in self.params:
            m_n = stats["m_n"]
            m_d = stats["post_sum"][:, None]
            [baf_prior, rdr_prior] = self.means_prior
            [baf_lambda, rdr_lambda] = self.means_weight # strength

            m_n[:, :self.bafdim] += baf_lambda * baf_prior
            m_n[:, self.bafdim:] += rdr_lambda * rdr_prior
            self.means_[:, :self.bafdim] = m_n[:, :self.bafdim] / (m_d + baf_lambda)
            self.means_[:, self.bafdim:] = m_n[:, self.bafdim:] / (m_d + rdr_lambda)
        
        if "c" in self.params:
            # c_n = stats["c_n"]
            c_n = np.zeros((self.n_components, self.n_features))
            for [X, post_comp_mix, posteriors] in stats["raw_data"]:
                for k in range(self.n_components):
                    # BAF features
                    for i in range(self.bafdim):
                        num0 = post_comp_mix[:, k, 0] * (X[:, i] - (1 - self.means_[k, i])) ** 2
                        num1 = post_comp_mix[:, k, 1] * (X[:, i] - self.means_[k, i]) ** 2
                        c_n[k, i] += np.sum(num0 + num1)
                    # RDR features
                    for j in range(self.bafdim, self.n_features):
                        num = posteriors[:, k] * (X[:, j] - self.means_[k, j]) ** 2
                        c_n[k, j] += np.sum(num)
            c_d = stats["post_sum"][:, None]

            # diag-covariance priors
            [baf_alpha, rdr_alpha] = self.covars_alpha
            [baf_beta, rdr_beta] = self.covars_beta
            c_n[:, :self.bafdim] += 2 * baf_beta
            c_n[:, self.bafdim:] += 2 * rdr_beta
            self.covars_[:, :self.bafdim] = c_n[:, :self.bafdim] / (c_d + 2 * (baf_alpha + 1))
            self.covars_[:, self.bafdim:] = c_n[:, self.bafdim:] / (c_d + 2 * (rdr_alpha + 1))

        if "t" in self.params and self.diag_transmat:
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
