import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax.scipy.stats import norm
from jax import grad, hessian
import seaborn as sns

def jax_BS_call(SK, sigma_tau):
    d1 = (SK / sigma_tau) + 0.5 * sigma_tau
    d2 = d1 - sigma_tau
    return jnp.exp(SK) * norm.cdf(d1) - norm.cdf(d2)

def jax_BS_digital(SK, sigma_tau):
    d1 = (SK / sigma_tau) + 0.5 * sigma_tau
    d2 = d1 - sigma_tau
    return norm.cdf(d2)

def BS_pdf(SK, sigma_tau):
    d1 = (SK / sigma_tau) + 0.5 * sigma_tau
    d2 = d1 - sigma_tau
    return norm.pdf(d2)

def first_order_greeks(Xs):
    vec_1 = lambda x, y: grad(jax_BS_call, argnums=(0, 1))(x, y)
    delta, vega =  jnp.vectorize(vec_1)(Xs[:,0], Xs[:,1])
    return delta, vega

def second_order_greek(Xs):
    vec_2 = lambda x, y : grad(grad(jax_BS_call))(x, y)
    gamma = jnp.vectorize(vec_2)(Xs[:,0], Xs[:,1])
    return gamma

def bs_log_pde_err(moneyness, ttm, d_ttm, d_x, d2_x):
    fig, ax = plt.subplots()
    PDE_err = -d_ttm + ttm * (-d_x + d2_x)
    ax.scatter(moneyness, PDE_err)
    ax.set_title("PDE Error")
    return PDE_err

def bs_pde_err(moneyness, ttm, d_ttm, d_x, d2_x):
    fig, ax = plt.subplots()
    PDE_err = -d_ttm + 0.5 * ttm * (moneyness ** 2) * d2_x
    ax.scatter(moneyness, PDE_err)
    ax.set_title("PDE Error")
    return PDE_err

def gbm_step(F, dt, sigma, Z):
    return F - (0.5 * (sigma ** 2) * dt) + sigma * Z
