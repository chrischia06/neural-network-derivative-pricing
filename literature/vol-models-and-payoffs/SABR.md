# NN SABR

**Katia Babbar, William Mcghee, You Wu, NN for Cross-Currency Options under the correlated SABR Model**
+ https://www.cqfinstitute.org/sites/default/files/3%20-%20Katia%20Babbar%20SABR_CrossSmile_NN_QuantInsights_2021_FINAL.pdf
+ https://www.cqfinstitute.org/content/cross-currency-options-and-correlated-sabr-model
+ Usually Beta is fixed (to be 0 or 1), given both beta and rho affect the smile, so only alpha (vol) , rho (corr), vol-of-vol need to be correlated
+ Their application is FX cross-asset smiles, but this could be applied to CMS spread options
+ Fixed Strikes
+ Dataset generation: They first considered a hypercube of 7 points in 8 dimensions for a total of 7^8 = 5,764,801, but only 2,825,977 of these are valid
+ Activation: Softplus, Regularisation: dropout, Optimizer: Adam, Learning Rate: Adaptive, 1 Hidden layer with 2000 units
+ Mentions that reducing max norm error is needed for production

**Hugues Thorin, Artificial Neural Networks for SABR Model Calibration & Hedging**
+ MSc Thesis
+ Based on McGhee's paper
+ https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3995626


**Machine learning SABR model of stochastic volatility with lookup table**

+ https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3589367
+ "How- ever, unlike the approximation schemes based on asymptotic methods – universally deemed fastest – the methodology admits arbitrary calcula- tion precision to the true pricing function without detrimental impact on time performance apart from memory access latency. The idea is plainly applicable to any function approximation or supervised learning domain with low dimension."
+ one of $\beta, \rho$ is typically fixed,  given both affect the smile
+ 200 sample paths, \Delta t  = 1 / 252
+ $\beta, \delta$ set to 1/2, 0

# SABR Theory

+ https://github.com/ynouri/pysabr
+ https://github.com/JackJacquier/SABR-Implied-Volatility
+ https://github.com/RussoMarioDamiano/VanillaSABR
+ ZABR https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1980726
+ Delta and vega hedging in the SABR and LMM-SABR models https://www.risk.net/sites/risk/files/import_unmanaged/risk.net/data/risk/pdf/technical/2008/risk_0812_technical_ird1.pdf






Shifted SABR: $dS_{t} = \sigma_{t}(S_{t} + \delta)^{\beta}, S_{0} > -\delta$


Zhang, J. (2013) SABR Enhancement-Fast Calibration, Arbitrage-
Free Extrapolation, Efficient SABR Implementaion. Available at SSRN:
https://ssrn.com/abstract=2443506

Hagan Managing Smile Risk (2002)
+ Original SABR paper

Oblój, J. (2008) Fine-tune your Smile: Correction to Hagan et al. Wilmott Magazine
+ 

https://www.uv.es/bfc/TFM2019/Julian_Arce

https://www.quantlib.org/slides/qlum15/kienitz.pdf

Finite Difference Techniques for Arbitrage Free SABR
+ https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2402001

https://hpcquantlib.wordpress.com/2019/10/12/almost-exact-sabr-interpolation-using-neural-networks-and-gradient-boosted-trees/

https://hpcquantlib.wordpress.com/2019/01/11/finite-difference-solver-for-the-sabr-model/

Explicit SABR Calibration Through Simple Expansions

+ https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2467231

Arbitrage Free SABR
+ http://janroman.dhis.org/finance/SABR/Arbitrage-Free%20SABR.pdf



+ https://github.com/ynouri/pysabr



Candidate point selection using a self-attention mechanism for generating a smooth volatility surface under the SABR model

+ https://www.sciencedirect.com/science/article/abs/pii/S0957417421000816?casa_token=x10ExvheVzsAAAAA:OdrIKmcThcSjRjQTHujCiqvbVHwbce-DZ5BlOOruI_hd0QMDwIEoN0QxVIKuZVcotAUENB1N


# NNs for SABR

Medhi Thomas, Pricing and calibration of stochastic models via neural networks

William Mcghee, An Artificial Neural Network Representation of the SABR Stochastic Volatility Model (2018)
+ https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3288882

Huges THorin artificial neural network for the SABR Model 


**DALVIR MANDARA, Artificial Neural Networks for Black-Scholes Option Pricing and Prediction of Implied Volatility for the SABR Stochastic Volatility Model (2019)**

+ MSc Thesis
+ Appliction to BS, SABR
+ Implied Vol

**Jaegi Jeon, Kyunghyun Park, Jeonggyu Huh, Extensive networks would eliminate the demand for pricing formulas (2021)**
+ https://arxiv.org/pdf/2101.09064.pdf

NN for Cross-Currency Options under the correlated SABR Model
+ https://www.cqfinstitute.org/sites/default/files/3%20-%20Katia%20Babbar%20SABR_CrossSmile_NN_QuantInsights_2021_FINAL.pdf
+ 2 SABR processes for FX cross-option smiles


Analyzing the Dynamics of the Swaption Market Using Neural Networks (2021)
+ https://www.dpublication.com/wp-content/uploads/2021/10/9-3021.pdf

https://www.cqfinstitute.org/sites/default/files/3%20-%20Katia%20Babbar%20SABR_CrossSmile_NN_QuantInsights_2021_FINAL.pdf

On the Calibration of the SABR Model and its Extensions
+ https://www.imperial.ac.uk/media/imperial-college/faculty-of-natural-sciences/department-of-mathematics/math-finance/Cheng_Luo-thesis.pdf


NN for Cross-Currency Options under the correlated SABR Mode
+ https://www.cqfinstitute.org/sites/default/files/3%20-%20Katia%20Babbar%20SABR_CrossSmile_NN_QuantInsights_2021_FINAL.pdf
+ 2 SABR Processes for FX Cross smiles


# Theory

Finite Difference Techniques for Arbitrage Free SABR
+ https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2402001

Decoding Hagan's Arbitrage Free SABR PDE Derivation
+ https://chasethedevil.github.io/post/decoding-hagans-arbitrage-free-sabr-pde-derivation/

Almost exact SABR Interpolation using Neural Networks and Gradient Boosted Trees
+ https://hpcquantlib.wordpress.com/2019/10/12/almost-exact-sabr-interpolation-using-neural-networks-and-gradient-boosted-trees/

Managing Smile Risk

+ https://lesniewski.us/papers/published/ManagingSmileRisk.pdf

Advanced Analytics for the SABR Model
+ https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2026350

FINE-TUNE YOUR SMILE CORRECTION TO HAGAN ET AL JAN OBLoJ
+ https://arxiv.org/pdf/0708.0998.pdf

On the Calibration of the SABR Model and its Extensions
+ https://www.imperial.ac.uk/media/imperial-college/faculty-of-natural-sciences/department-of-mathematics/math-finance/Cheng_Luo-thesis.pdf


Neural network approximation to the SABR option pricing model
+ Slides:https://parleyyang.github.io/Cam/LT2021/Slide.pdf
