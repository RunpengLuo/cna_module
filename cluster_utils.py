import os
import sys
from collections import Counter
import kneed
import numpy as np
import pandas as pd
from scipy.special import betaln, gammaln, logsumexp, xlogy
from scipy.optimize import minimize, minimize_scalar
from sklearn import cluster
from utils import *

import matplotlib.pyplot as plt


##################################################
def estimate_overdispersion(
    a_counts: np.ndarray, b_counts: np.ndarray, p=0.5
):
    """learn over-dispersion parameter from netrual bins"""

    def neg_loglik_logw(logw):
        w = np.exp(logw)
        a0 = w * p
        b0 = w * (1 - p)
        a1 = a_counts + a0
        b1 = b_counts + b0
        ll = np.sum(betaln(a1, b1) - betaln(a0, b0))
        return -ll
    max_tau = 200
    res = minimize_scalar(
        neg_loglik_logw,
        method="bounded",
        bounds=(np.log(1), np.log(max_tau)),
        # options={"ftol":1e-6}
    )
    tau = np.exp(res.x)
    if np.abs(tau - max_tau) <= 1:
        return None # fall-back to binomial instead
    return tau

##################################################
def make_transmat(diag, K):
    offdiag = (1 - diag) / (K - 1)
    transmat_ = np.diag([diag - offdiag] * K)
    transmat_ += offdiag
    return transmat_

def init_hmm_kmeans_plus_plus(
    X_inits: np.ndarray, K: int, M: int, random_state=42, n_init=10, min_covars=1e-6
):
    """
    X_inits: (N, M * 2), M=#iid tumor samples.
    """
    kmeans = cluster.KMeans(
        n_clusters=K,
        random_state=random_state,
        init="k-means++",
        n_init=n_init,
    )
    cluster_labels = kmeans.fit_predict(X=X_inits)
    means = kmeans.cluster_centers_

    baf_means = means[:, M:]
    # TODO set netural -> 0.5
    # baf_means =

    rdr_means = means[:, :M]
    rdrs = X_inits[:, :M]
    if rdrs.ndim == 1:
        rdrs = rdrs[:, np.newaxis]
    rdr_vars = np.full((K, M), fill_value=min_covars, dtype=np.float32)
    for k in range(K):
        mask = cluster_labels == k
        num_points = np.sum(mask)
        if num_points < 2:
            continue
        cluster_rdrs = rdrs[mask, :]
        rdr_vars[k, :] = np.sum((cluster_rdrs - rdr_means[k, :]) ** 2, axis=0) / (
            num_points - 1
        )

    rdr_vars = np.maximum(rdr_vars, min_covars)
    return rdr_means, baf_means, rdr_vars


def compute_loglik(
    X_rdrs: np.ndarray,
    X_alphas: np.ndarray,
    X_betas: np.ndarray,
    X_totals: np.ndarray,
    rdr_means: np.ndarray,
    rdr_vars: np.ndarray,
    baf_means: np.ndarray,
    tau=None,
):
    log_norm_const = 0.5 * np.sum(np.log(2 * np.pi * rdr_vars), axis=1)
    # (N, 1, M) - (1, K, M) → (N, K, M)
    sq_diff = (X_rdrs[:, None, :] - rdr_means[None, :, :]) ** 2
    quad = 0.5 * np.sum(sq_diff / rdr_vars[None, :, :], axis=2)
    ll_rdrs = -quad - log_norm_const[None, :]  # (N, K)

    if not tau is None:
        # beta-binomial
        # (N, M)
        log_binom_const = (
            gammaln(X_totals + 1) - gammaln(X_betas + 1) - gammaln(X_alphas + 1)
        )
        # (1, K, M)
        bb_alpha = tau * baf_means[None, :, :]
        bb_beta = tau * (1 - baf_means[None, :, :])
        bb_delta = betaln(bb_alpha, bb_beta)

        # h1, beta is b-allele
        lnB_num_h0 = betaln(
            X_alphas[:, None, :] + bb_alpha, X_betas[:, None, :] + bb_beta
        )
        lnB_num_h1 = betaln(
            X_betas[:, None, :] + bb_alpha, X_alphas[:, None, :] + bb_beta
        )

        # (N, K) sum across features
        ll_bafs_h0 = np.sum(log_binom_const[:, None, :] + lnB_num_h0 - bb_delta, axis=2)
        ll_bafs_h1 = np.sum(log_binom_const[:, None, :] + lnB_num_h1 - bb_delta, axis=2)
    else:
        # binomial
        # (N, M)
        log_binom_const = (
            gammaln(X_totals + 1) - gammaln(X_betas + 1) - gammaln(X_alphas + 1)
        )

        # (N, K, M)
        # h1, beta is b-allele
        lnB_h0 = xlogy(X_alphas[:, None, :], baf_means[None, :, :]) + xlogy(
            X_betas[:, None, :], 1 - baf_means[None, :, :]
        )
        lnB_h1 = xlogy(X_betas[:, None, :], baf_means[None, :, :]) + xlogy(
            X_alphas[:, None, :], 1 - baf_means[None, :, :]
        )

        # (N, K) sum across features
        ll_bafs_h0 = np.sum(log_binom_const[:, None, :] + lnB_h0, axis=2)
        ll_bafs_h1 = np.sum(log_binom_const[:, None, :] + lnB_h1, axis=2)

    # LLs for h0 and h1 (N, K) * 2
    lls0 = ll_rdrs + ll_bafs_h0
    lls1 = ll_rdrs + ll_bafs_h1
    return lls0, lls1


def forward_backward(
    lls0: np.ndarray,
    lls1: np.ndarray,
    X_lengths: np.ndarray,
    log_startprobs: np.ndarray,
    log_switchprobs: np.ndarray,
    log_stayprobs: np.ndarray,
    log_transmat: np.ndarray,
    K: int,  # states
    N: int,  # blocks
):
    """
    E-step, conditional posterior and per-segment data loglik
    """
    posts = np.zeros((N, K, 2), dtype=np.float64)
    logliks = np.zeros_like(X_lengths, dtype=np.float64)
    start = 0
    for n, nobs in enumerate(X_lengths):
        end = start + nobs
        lls0_seg = lls0[start:end]  # (nobs, K)
        lls1_seg = lls1[start:end]
        log_switchprobs_seg = log_switchprobs[start:end]  # (nobs, )
        log_stayprobs_seg = log_stayprobs[start:end]

        # forward lattice
        fwd_lattice = np.zeros((nobs, K, 2), dtype=np.float64)
        fwd_lattice[0, :, 0] = lls0_seg[0] + log_startprobs[:, 0]
        fwd_lattice[0, :, 1] = lls1_seg[0] + log_startprobs[:, 1]

        # normalization constant
        log_c = np.zeros(nobs, dtype=np.float64)
        log_c[0] = logsumexp(fwd_lattice[0])
        fwd_lattice[0] -= log_c[0]

        for obs in range(1, nobs):
            pswitch = log_switchprobs_seg[obs]
            pstay = log_stayprobs_seg[obs]
            for k in range(K):
                fwd_lattice[obs, k, 0] = lls0_seg[obs, k] + logsumexp(
                    np.concatenate(
                        [
                            fwd_lattice[obs - 1, :, 0] + log_transmat[:, k] + pstay,
                            fwd_lattice[obs - 1, :, 1] + log_transmat[:, k] + pswitch,
                        ]
                    )
                )

                fwd_lattice[obs, k, 1] = lls1_seg[obs, k] + logsumexp(
                    np.concatenate(
                        [
                            fwd_lattice[obs - 1, :, 0] + log_transmat[:, k] + pswitch,
                            fwd_lattice[obs - 1, :, 1] + log_transmat[:, k] + pstay,
                        ]
                    )
                )
            # normalize this step
            log_c[obs] = logsumexp(fwd_lattice[obs])
            fwd_lattice[obs] -= log_c[obs]

        # backward lattice
        bwd_lattice = np.zeros((nobs, K, 2), dtype=np.float64)
        bwd_lattice[-1] = 0.0
        for obs in reversed(range(nobs - 1)):
            pswitch = log_switchprobs_seg[obs + 1]
            pstay = log_stayprobs_seg[obs + 1]

            for k in range(K):
                bwd_lattice[obs, k, 0] = logsumexp(
                    np.concatenate(
                        [
                            log_transmat[k, :]
                            + pstay
                            + lls0_seg[obs + 1, :]
                            + bwd_lattice[obs + 1, :, 0],
                            log_transmat[k, :]
                            + pswitch
                            + lls1_seg[obs + 1, :]
                            + bwd_lattice[obs + 1, :, 1],
                        ]
                    )
                )

                bwd_lattice[obs, k, 1] = logsumexp(
                    np.concatenate(
                        [
                            log_transmat[k, :]
                            + pstay
                            + lls0_seg[obs + 1, :]
                            + bwd_lattice[obs + 1, :, 1],
                            log_transmat[k, :]
                            + pswitch
                            + lls1_seg[obs + 1, :]
                            + bwd_lattice[obs + 1, :, 0],
                        ]
                    )
                )
            bwd_lattice[obs] -= log_c[obs + 1]

        log_gamma = fwd_lattice + bwd_lattice
        log_gamma -= logsumexp(log_gamma, axis=(1, 2), keepdims=True)
        posts[start:end, :, :] = np.exp(log_gamma)
        # total log-likelihood = sum(log_c)
        logliks[n] = np.sum(log_c)
        start = end
    return posts, np.sum(logliks)


def _do_mstep(
    X_rdrs: np.ndarray,
    X_alphas: np.ndarray,
    X_betas: np.ndarray,
    posts: np.ndarray,  # (N, K, 2)
    K: int,
    M: int,
    N: int,
    tau=None,
    min_covar=1e-6,
    tol=1e-6,
):
    def neg_loglik(p): # convexity proof?
        a0 = tau * p
        b0 = tau * (1 - p)
        logprob0 = betaln(X_alphas[:, m] + a0, X_betas[:, m] + b0) - betaln(a0, b0)# beta is not b-allele
        logprob1 = betaln(X_betas[:, m] + a0, X_alphas[:, m] + b0) - betaln(a0, b0)# beta is b-allele
        ll = np.sum(logprob0 * posts[:, k, 0]) + np.sum(logprob1 * posts[:, k, 1])
        return -ll

    posts_marg = np.sum(posts, axis=-1)  # (N, K)
    Nk = np.sum(posts_marg, axis=0)  # (K, )

    # update startprobs TODO

    # update RDR means and covars
    rdr_means = np.zeros((K, M), dtype=np.float64)
    rdr_vars = np.zeros((K, M), dtype=np.float64)
    for k in range(K):
        w = posts_marg[:, k][:, None]  # (N, 1)
        rdr_means[k] = np.sum(w * X_rdrs, axis=0) / Nk[k]
        rdr_vars[k] = np.maximum(
            np.sum(w * (X_rdrs - rdr_means[k]) ** 2, axis=0) / Nk[k], min_covar
        )
    # update BAF
    baf_means = np.zeros((K, M), dtype=np.float64)
    for k in range(K):
        acounts0 = np.sum(X_alphas * posts[:, k, 0][:, None], axis=0)
        acounts1 = np.sum(X_alphas * posts[:, k, 1][:, None], axis=0)
        bcounts0 = np.sum(X_betas * posts[:, k, 0][:, None], axis=0)
        bcounts1 = np.sum(X_betas * posts[:, k, 1][:, None], axis=0)
        tcounts = acounts0 + acounts1 + bcounts0 + bcounts1
        baf_means[k] = (bcounts1 + acounts0) / tcounts
    if tau is None:
        for k in range(K):
            acounts0 = np.sum(X_alphas * posts[:, k, 0][:, None], axis=0)
            bcounts1 = np.sum(X_betas * posts[:, k, 1][:, None], axis=0)
            baf_means[k] = (bcounts1 + acounts0) / tcounts
    else:
        for k in range(K):
            for m in range(M):
                res = minimize_scalar(neg_loglik, bounds=(tol, 1 - tol))
                baf_means[k, m] = res.x
    return rdr_means, rdr_vars, baf_means


def run_hmm(
    X_rdrs: np.ndarray,
    X_alphas: np.ndarray,
    X_betas: np.ndarray,
    X_totals: np.ndarray,
    X_inits: np.ndarray,
    X_lengths: np.ndarray,
    log_switchprobs: np.ndarray,
    log_stayprobs: np.ndarray,
    log_transmat: np.ndarray,
    K: int,
    tau=None,
    n_iter=10,
    random_state=42,
    min_covar=1e-6,
    tol_ll=1e-4,
    tol=1e-6,
    init_method="k-means++",
    verbose=True,
):
    """co-cluster allele counts and RDRs
    latent variables z_n=1...K, h_n=0 or 1.
    h_n=1 if beta counts is b-allele.

    Args:
        X_rdrs, X_alphas, X_betas, X_totals (np.ndarray): (N, M)
        X_lengths (np.ndarray): (S, )
        switchprobs (np.ndarray): (N, )
        diag_transmat (np.ndarray): diagonal transition. (K, K)
        K (int): number of clusters
        tau (float, optional): over-dispersion. Defaults to None.
        min_covar (float, optional): min covariance for RDR. Defaults to 1e-3.
        n_iter (int, optional): number of iterations. Defaults to 10.
        random_state (int, optional): Defaults to 42.
        tol (_type_, optional): Defaults to 1e-6.
        init_method (str, optional): Defaults to "k-means++".
        verbose (bool, optional): Defaults to True.
    """
    (N, M) = X_rdrs.shape

    # init means and covars
    # TODO standardize means and vars
    assert init_method == "k-means++"
    if init_method == "k-means++":
        rdr_means, baf_means, rdr_vars = init_hmm_kmeans_plus_plus(
            X_inits, K, M, random_state=random_state, min_covars=min_covar
        )
        baf_means = np.clip(baf_means, a_min=tol, a_max=1 - tol)  # avoid log(0)
    
    if verbose:
        print(f"Init")
        print(baf_means.flatten())
        print(rdr_means.flatten())
        print(rdr_vars.flatten())

    # segment-specific startprobs?
    log_startprobs = np.log(np.full((K, 2), 1.0 / (2 * K)))

    elbo_trace = [-np.inf]
    for it in range(n_iter):
        # (N, K)
        lls0, lls1 = compute_loglik(
            X_rdrs, X_alphas, X_betas, X_totals, rdr_means, rdr_vars, baf_means, tau
        )
        posts, loglik = forward_backward(
            lls0,
            lls1,
            X_lengths,
            log_startprobs,
            log_switchprobs,
            log_stayprobs,
            log_transmat,
            K,
            N,
        )

        # update parameter
        rdr_means, rdr_vars, baf_means = _do_mstep(
            X_rdrs, X_alphas, X_betas, posts, K, M, N, tau, min_covar, tol
        )

        delta_ll = loglik - elbo_trace[-1]
        elbo_trace.append(loglik)
        if verbose:
            print(f"Iter {it:03d} | Q={loglik: .6f} | delta={delta_ll: .6f}")
            print(baf_means.flatten())
            print(rdr_means.flatten())
            print(rdr_vars.flatten())

        if np.abs(delta_ll) < tol_ll:
            if verbose:
                print(f"Converged at iteration {it}")
            break
    # sys.exit(0)
    # decode cluster states with MAP estimate
    posts_marg = np.sum(posts, axis=-1) # (N,K)
    cluster_labels = np.argmax(posts_marg, axis=1)
    # posts_phase = np.sum(posts[:, cluster_labels, :], axis=1) # (N, 2)
    # phase_labels = np.argmax(posts_phase, axis=1)
    phase_labels = None
    return {
        "seed": random_state,
        "RDR_means": rdr_means,
        "RDR_vars": rdr_vars,
        "BAF_means": baf_means,
        "tau": tau,
        "elbo_trace": elbo_trace,
        "model_ll": elbo_trace[-1],
        "cluster_labels": cluster_labels,
        "phase_labels": phase_labels
    }


# viterbi decode
