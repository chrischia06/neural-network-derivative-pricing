A neural network approach to pricing of a European Call Option with the Heston model
+ https://repository.urosario.edu.co/handle/10336/20712
+ MSc Thesis

Simulations of Implied Volatility and Option Pricing
using Neural Networks and Finite Difference
Methods for Heston Model
+ MSc Thesis
+ https://nur.nu.edu.kz/bitstream/handle/123456789/4697/Sukhrat%20Arziyev%20-%20Thesis.pdf?sequence=1&isAllowed=y

Alex Winter FX Volatility Calibration Using Artificial Neural Networks
+ https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3798513
+ 

Olivier Pironneau, Calibration of Heston Model with Keras (2019)
+ 
+ https://hal.sorbonne-universite.fr/hal-02273889/document

Caibration of the heston model with neural network approximations
+ https://dataspace.princeton.edu/handle/88435/dsp01c247dw13s

**Oliver Klingberg Malmer, Victor Tisell, Deep Learning and the Heston Model: Calibration & Hedging (2020)**
+ BSc Thesis
+ https://gupea.ub.gu.se/bitstream/handle/2077/65464/gupea_2077_65464_1.pdf?sequence=1



**Antal Ratku,  Dirk Neumann, Derivatives of feed-forward neural networks and their application in real-time market risk management (2022)**
+ https://link.springer.com/content/pdf/10.1007/s00291-022-00672-1.pdf
+ GitHub: https://github.com/antalratku/nn_deriv
+ The neural network selected for the following demonstration contains three
internal hidden layers of 128 nodes each, with activation functions tanh, sig-
moid, tanh, respectively. The output layer contains a single node with a sigmoid activation function.
+ Define monenyness : $m = K/s$. Then calls are bounded by 1 (i.e. using undrerlying as a numeraire)
+ Parameter Space: {"kappa": (0.1, 10.0), "theta": (0.01, 1.0), "sigma": (0.1, 2.0), "rho": (-1.0, 1.0), "v0": (0.0015, 1.5), "m": (0.1, 2.4), "tau":(0.0027, 3.0), "r": (-0.1, 0.1) "d": (0.0, 0.2)}
+ Errors are in the order 10^-3 for theta, delta, 10^-5 for vega, and 10^-2 for gamma, which seems quite high
+ Some discussion about AAD (needed for training, differentials vs neural network parameters), and analytic gradients / jacobians / Hessians for differentials vs input


Use Deep Learning to Approximate Barrier Option Prices with Heston Model
+ https://uk.mathworks.com/help/fininst/deep-learning-to-approximate-barrier-option-pricing.html

**Yi Dong, Accelerating Python for Exotic Option Pricing (2020)**

+ Article, Demo of NVIDIA functinality
+ https://developer.nvidia.com/blog/accelerating-python-for-exotic-option-pricing/
+ Asian Barriers
+ Greeks look very off
+ No discussion of accuracy, only speedup