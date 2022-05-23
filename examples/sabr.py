from jax import grad
import jax.numpy as jnp

def cev_step(F0:float, V0:float, beta:float, dt:float, Z1:float, rho:float, Z2:float):
    
    """
    F0: value of the forward process
    V0: value of the volatility process
    beta: value of the CEV beta
    dt: timestep
    Z2: value of the brownian motion associated with the volatility process
    Z1: a brownian motion independent of Z2
    rho: corrrelation between brownian motions
    """
    return jnp.abs(F0 + 
              V0 * jnp.exp((beta - 1) * F0) * 
              jnp.sqrt(dt) * (rho * Z2 + jnp.sqrt(1 - rho ** 2) + Z1) - 
           0.5 * (V0 ** 2) * dt * jnp.exp(2 * (beta - 1) * F0))
