#!/usr/bin/env python2

from __future__ import absolute_import, division, print_function

from math import log

import random
import numpy as np

alpha = 10
beta = 25

n = [10, 50, 100]

n_samp = 200


def mle_alphas(x):
    """Takes a two dimensional ndarray with rows
    corresponding to a simulation and columns corresponding
    to individual random samples for a given simulation
    and calculates the MLE alpha estimate for all simulations"""
    assert len(np.shape(x)) == 2
    num_sims = np.shape(x)[0]
    sample_size = np.shape(x)[1]

    # Calculate element-wise MLE alphas
    max_by_sim = np.amax(x, axis=1)
    alphas = np.divide(sample_size,
                       np.subtract(np.multiply(sample_size,
                                               np.log(max_by_sim)),
                                   np.log(np.product(x, axis=1))))

    return alphas


def mle_betas(x):
    """Takes a two dimensional ndarray with rows
    corresponding to a simulation and columns corresponding
    to individual random samples for a given simulation
    and calculates the MLE beta estimate for all simulations"""
    assert len(np.shape(x)) == 2
    num_sims = np.shape(x)[0]
    sample_size = np.shape(x)[1]

    betas = np.amax(x, axis=1)

    return betas


def mom_alphas(x):
    first_samp_moment = np.mean(x, axis=1)
    second_samp_moment = np.mean(np.square(x),
                                 axis=1)

    q = np.divide(second_samp_moment,
                  np.square(first_samp_moment))
    q_1 = np.subtract(q, 1)
    alphas = np.subtract(np.sqrt(np.divide(q,
                                           q_1)),
                         1)
    return alphas

    
def mom_betas(x):
    first_samp_moment = np.mean(x, axis=1)
    alphas = mom_alphas(x)
    num = np.multiply(first_samp_moment,
                      np.add(alphas, 1))
    betas = np.divide(num, alphas)
    return betas

    
def inverse_cdf(y):
    return beta * y ** (1/alpha)

if __name__ == "__main__":

    experiments = []
        
    for i in range(len(n)):
        experiments.append(np.zeros((n_samp, n[i])))

        # Perform the MC simulation
        for j in range(n_samp):
            for k in range(n[i]):
                u = random.uniform(0,1)
                experiments[i][j][k] = inverse_cdf(u)

        # Compute MSE for MM and MLE
        ml_alphas = mle_alphas(experiments[i])
        mm_alphas = mom_alphas(experiments[i])
        ml_betas = mle_betas(experiments[i])
        mm_betas = mom_betas(experiments[i])

        ml_alpha_mse = np.mean(np.square(np.subtract(ml_alphas,
                                                     alpha)))
        mm_alpha_mse = np.mean(np.square(np.subtract(mm_alphas,
                                                     alpha)))
        ml_beta_mse = np.mean(np.square(np.subtract(ml_betas,
                                                    beta)))
        mm_beta_mse = np.mean(np.square(np.subtract(mm_betas,
                                                    beta)))

        # Print the results
        print("n =",n[i])
        print("MLE MSE for Alpha:", ml_alpha_mse)
        print("MM MSE for Alpha:", mm_alpha_mse)
        print("MLE MSE for Beta:", ml_beta_mse)
        print("MM MSE for Beta:", mm_beta_mse)
