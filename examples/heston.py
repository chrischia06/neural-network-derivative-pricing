import numpy as np
from scipy.integrate import quad

def heston_characteristic(u, tau, v, kappa, vbar, vol_of_vol, rho):
    """
    u - value to be evaluated at
    tau - time to maturity
    vol_of_vol - vol-of-vol, coefficient on brownian motion of volatility
    vbar - mean level of variance
    lambda - mean reversion speed
    rho - correlation between brownians
    Gatheral - Volatility Surface pg 18 - 
    """
    alpha = -0.5 * (u ** 2) - (0.5 * u * 1j)
    beta = kappa - rho * vol_of_vol * u * 1j
    gamma = 0.5 * (vol_of_vol ** 2)
    d = np.sqrt((beta ** 2) - 4 * alpha * gamma)
    rp = (beta  + d) / (2 * gamma)
    rm = (beta - d) / (2 * gamma)
    g = rm / rp
    C = kappa * (rm * tau  - 2 / (vol_of_vol ** 2) * np.log((1 - g * np.exp(-d * tau)) / (1 - g)))
    D = rm * (1 - np.exp(-d * tau)) / (1 - np.exp(-d * tau) * g) 
    return C * vbar + D * v


def heston_cdf(x, tau, v, kappa, vbar, vol_of_vol, rho):
    """
    risk-neutral probability of exercise
    """
    c_func = lambda u: heston_characteristic(u, tau, v, kappa, vbar, vol_of_vol, rho)
    return 0.5 + (1 / np.pi) * quad(lambda u: np.real(np.exp(c_func(u) + 1j * u * x) / (u * 1j)), 0, 100, limit=1000)[0]

def heston_density_quad(k, tau, v, kappa, vbar, vol_of_vol, rho):
    """
    heston pdf via integration
    """
    c_func = lambda u: heston_characteristic(u, tau, v, kappa, vbar, vol_of_vol, rho)
    return 0.5 * np.pi * quad(lambda u: np.real(np.exp(c_func(u) + 1j * u * k)), 0, 100, limit=1000)[0]
    
def density2(x, tau, v, kappa, vbar, vol_of_vol, rho, eps = 1e-6):
    """
    heston pdf via bumping
    """
    return (f(x + eps, tau, v, kappa, vbar, vol_of_vol, rho) - f(x, tau, v, kappa, vbar, vol_of_vol, rho)) / eps

# def heston_call_quad(k, tau, v, kappa, vbar, vol_of_vol, rho):
#     func = lambda u: heston_density_quad(u, tau, v, kappa, vbar, vol_of_vol, rho)
#     return quad(lambda y: np.maximum((np.exp(y) - 1.0), 0) * func(y), k, 6, limit = 30)[0]

def heston_call_quad(k, tau, v, kappa, vbar, vol_of_vol, rho):
    func = lambda u: heston_density_quad(u, tau, v, kappa, vbar, vol_of_vol, rho)
    vec_fun = np.vectorize(func)
    grid = np.linspace(k, 100, 100)
    dx = np.diff(grid)[0]
    # return np.sum(np.maximum(np.exp(grid) - 1.0, 0) * vec_func(grid) * dx)
    return quad(lambda y: np.maximum((np.exp(y) - 1.0), 0) * func(y), k, 100, limit = 30)[0]

def cir_step(Vt, kappa, vbar, dt, vol_of_vol, W_v):
    """
    Reflected Euler Scheme
    Vt : Volatilty
    kappa: Mean reversion speed
    vbar: Mean volatility level
    dt: time increment
    vol_of_vol: Volatility of the volatility process
    W_v: brownian increment
    """
    return np.abs(Vt + kappa * (vbar - Vt) * dt + vol_of_vol * np.sqrt(Vt) * W_v)

def heston_step(St, dt, Vt, W_s, W_v, rho):
    """
    Reflected Euler Scheme
    St: Forward process
    dt: time increment
    Ws: Brownian increment
    """
    return np.abs(St * (1 + dt + np.sqrt(Vt) * W_s))


# def f(x, tau, v, kappa, vbar, vol_of_vol, rho):
#     """
#     risk-neutral probability of exercise
#     """
#     c_func = lambda u: heston_characteristic(u, tau, v, kappa, vbar, vol_of_vol, rho)
#     grid = np.linspace(1e-6, 100, 10000)
#     dx = np.diff(grid)[0]
#     return 0.5 + (1 / np.pi) * np.sum(np.real(np.exp(c_func(grid) + 1j * grid * x) / (grid * 1j)) * dx)
#     # return 0.5 + (1 / np.pi) * quad(lambda u: np.real(np.exp(c_func(u) + 1j * u * x) / (u * 1j)), 0, 100, limit=1000)

# def density2(x, tau, v, kappa, vbar, vol_of_vol, rho, eps = 2 ** (-4)):
#     """
#     heston pdf via bumping
#     """
#     return (f(x + eps, tau, v, kappa, vbar, vol_of_vol, rho) - f(x, tau, v, kappa, vbar, vol_of_vol, rho)) / eps


# import jax.numpy as np

# def f(x, tau, v, kappa, vbar, vol_of_vol, rho):
#     """
#     risk-neutral probability of exercise
#     """
#     c_func = lambda u: heston_characteristic(u, tau, v, kappa, vbar, vol_of_vol, rho)
#     grid = np.linspace(1e-6, 100, 10000)
#     dx = np.diff(grid)[0]
#     return 0.5 + (1 / np.pi) * np.sum(np.real(np.exp(c_func(grid) + 1j * grid * x) / (grid * 1j)) * dx)
#     # return 0.5 + (1 / np.pi) * quad(lambda u: np.real(np.exp(c_func(u) + 1j * u * x) / (u * 1j)), 0, 100, limit=1000)

# def density2(x, tau, v, kappa, vbar, vol_of_vol, rho, eps = 2 ** (-4)):
#     """
#     heston pdf via bumping
#     """
#     return (f(x + eps, tau, v, kappa, vbar, vol_of_vol, rho) - f(x, tau, v, kappa, vbar, vol_of_vol, rho)) / eps


# def call(k, tau, v, kappa, vbar, vol_of_vol, rho):
#     func = lambda u: density(u, tau, v, kappa, vbar, vol_of_vol, rho)
#     grid = np.linspace(k, 10, 100)
#     dx = np.diff(grid)[0]
#     return np.sum(np.maximum((np.exp(grid) - 1.0), 0) * np.array([func(g) for g in grid]) * dx)
#     # return quad(lambda y: np.maximum((np.exp(y) - 1.0), 0) * func(y), k, 6, limit = 30)[0]

# space = np.linspace(-3, 3, 100)
# plt.plot(space, [call(x, tau, v, kappa, vbar, vol_of_vol, rho) for x in tqdm(space)])


# %%time
# preds = priceHestonMid(St = Xs[:,0], 
#                        K = 1, r = 0, 
#                        T = Xs[:,2], 
#                        sigma = Xs[:,1], 
#                        kappa = Xs[:,3], 
#                        theta = Xs[:,5], 
#                        volvol = Xs[:,6], 
#                        rho = Xs[:, 4])

# eps = 1e-6


# preds2 = priceHestonMid(St = Xs[:,0] + eps, 
#                        K = 1, r = 0, 
#                        T = Xs[:,2], 
#                        sigma = Xs[:,1], 
#                        kappa = Xs[:,3], 
#                        theta = Xs[:,5], 
#                        volvol = Xs[:,6], 
#                        rho = Xs[:, 4])


# # https://calebmigosi.medium.com/build-the-heston-model-from-scratch-in-python-part-ii-5971b9971cbe
# import numpy as np# Parallel computation using numba
# from numba import jit, njit, prange

# from numba import cuda
# i = complex(0,1)# To be used in the Heston pricer
# @jit
# def fHeston(s, St, K, r, T, sigma, kappa, theta, volvol, rho):
#     # To be used a lot
#     prod = rho * sigma *i *s 
    
#     # Calculate d
#     d1 = (prod - kappa)**2
#     d2 = (sigma**2) * (i*s + s**2)
#     d = np.sqrt(d1 + d2)
    
#     # Calculate g
#     g1 = kappa - prod - d
#     g2 = kappa - prod + d
#     g = g1/g2334r4fgYUb   J
    
#     # Calculate first exponential
#     exp1 = np.exp(np.log(St) * i *s) * np.exp(i * s* r* T)
#     exp2 = 1 - g * np.exp(-d *T)
#     exp3 = 1- g
#     mainExp1 = exp1*np.power(exp2/exp3, -2*theta*kappa/(sigma **2))
    
#     # Calculate second exponential
#     exp4 = theta * kappa * T/(sigma **2)
#     exp5 = volvol/(sigma **2)
#     exp6 = (1 - np.exp(-d * T))/(1 - g * np.exp(-d * T))
#     mainExp2 = np.exp((exp4 * g1) + (exp5 *g1 * exp6))
    
#     return (mainExp1 * mainExp2)# Heston Pricer (allow for parallel processing with numba)
# @jit(forceobj=True)
# def priceHestonMid(St, K, r, T, sigma, kappa, theta, volvol, rho):
#     P, iterations, maxNumber = 0,1000,100
#     ds = maxNumber/iterations
    
#     element1 = 0.5 * (St - K * np.exp(-r * T))
    
#     # Calculate the complex integral
#     # Using j instead of i to avoid confusion
#     for j in prange(1, iterations):
#         s1 = ds * (2*j + 1)/2
#         s2 = s1 - i
        
#         numerator1 = fHeston(s2,  St, K, r, T, 
#                              sigma, kappa, theta, volvol, rho)
#         numerator2 = K * fHeston(s1,  St, K, r, T, 
#                               sigma, kappa, theta, volvol, rho)
#         denominator = np.exp(np.log(K) * i * s1) *i *s1
        
#         P = P + ds *(numerator1 - numerator2)/denominator
    
#     element2 = P/np.pi
    
#     return np.real((element1 + element2))

