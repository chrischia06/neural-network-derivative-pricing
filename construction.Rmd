---
title: "R Notebook"
output: html_notebook
---

## Risk Neutral Implied Distribution

**Gated Neural Networks for Option Pricing: Rationality by Design (2017)**

+ https://www.pure.ed.ac.uk/ws/portalfiles/portal/29546775/yang_yu.pdf
+ https://github.com/arraystream/fftoptionlib
+ "for deep out-of-money options or those with long maturity, NN methods perform very badly (Bennell and Sutcliffe 2004)"
+ "(Gradojevic, Gencay, and Kukolj 2009) tried to address this issue by categorising options based on their moneyness and time to maturity, and training independent NNs for each class of options."
+ 6 no-arbitrage call constraints from Follmer, Schied, Stochastic Finance
+ Softplus * sigmoid $C(K/S, 1, T - t) = \sum_{j = 1}^{J}\sigma_{1}(b_{1, j} - me^{w_{1, j}})\sigma_{2}(b_{2, j}-\tau e^{w_{2, j}})e^{w_{3, j}}$, $J$ number of hidden units   
+ they verify asymptotics and no-arb conditions
+ Then $\sum_{i = 1}^{I} y_{i}(m, \tau)w_{i}(m, \tau)$ - $I$ number of single pricing models
+ Solution to no-arb bounds and asymptotics is to generate more edge examples
+ Gated Neural Network - Sigaud, O.; Masson, C.; Filliat, D.; and Stulp, F. 2015. Gated
networks: an inventory. CoRR abs/1512.03201
+ learn-ing from hints trick (Abu-Mostafa 1993)
+ can be calibrated to prices

Chou et al., Deep Learning in Model Risk Neutral Distribution for Option Pricing (2019)

+ Uses the gated approach of Yang
+ Uses a mixture instead of a GBM
+ Taiwan Options


A Hybrid Approach to Modular and Gated Neural Networks for Option Pricing and Hedging

+ MSc Thesis
+ https://thesis.eur.nl/pub/53019/MSc_Thesis_Full.pdf

Formulation Of A Rational Option Pricing Model using Artificial Neural Networks

+ https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3724078

## Architecture

**Dugas, C.; Bengio, Y.; Belisle, F.; Nadeau, C.; and Garcia, R. 2000. Incorporating second-order functional knowledge for better option pricing. In Neural Information Processing Systems (NIPS)**

+  specific activation functions and positive weight parameter constraints such that
second-order derivatives positive. "These studies suggested that introducing econometric constraints produces better option pricing models compared to vanilla feed-forward NNs"
+ https://proceedings.neurips.cc/paper/2000/file/44968aece94f667e4095002d140b5896-Paper.pdf
+ Estimate on S&P Options


**Garcia, R. and Gençay, R. (2000). Pricing and hedging derivative securities with neural
networks and a homogeneity hint.**



**Option valuation under no-arbitrage constraints with neural networks**

+ http://eprints.nottingham.ac.uk/6540ƒ5/1/Combine.pdf





Differentiable Convex Optimization Layers

+ https://arxiv.org/pdf/1910.12430.pdf
+ https://github.com/cvxgrp/cvxpylayers


Automatic repair of convex optimization problems

+ https://web.stanford.edu/~boyd/papers/pdf/auto_repair_cvx.pdf

On the Selection of Initialization and Activation Function for Deep Neural Networks

+ https://arxiv.org/abs/1805.08266

**Garcia, R.; Ghysels, E.; and Renault, E. 2010. The Econometrics of Option Pricing. Elsevier Inc. 479–552.**

# NN Theory

**Testing the Manifold Hypothesis (2013)**
+ https://arxiv.org/abs/1310.0425


Optimizer
+ https://ruder.io/optimizing-gradient-descent/index.html#adam


https://arxiv.org/abs/2204.12446
https://old.reddit.com/r/MachineLearning/comments/un0crv/r_fullbatch_gd_generalizes_better_than_sgd/



# Soft Penalty

Ackerer, D., Tagasovska, N., and Vatter, T. (2019). Deep smoothing of the implied volatility
surface. Available at SSRN 3402942

Itkin, A. (2019). Deep learning calibration of option pricing models: some pitfalls and
solutions. arXiv:1906.03507.	


Black-Box Model Risk in Finance
+ https://arxiv.org/pdf/2102.04757.pdf	
+ relevant sections are: 2.1 How to use Neural Nets for derivatives modelling 4.2.1 Machine Learning as a numeric tool 4.22 Expert Knowledge 4.2.5 Explainability 4.2.6 Monitoring and control

# Post-Processing

Detecting and repairing arbitrage in traded option prices
+ https://arxiv.org/abs/2008.09454
+ formulate arbitrage repair as constrained linear programming
+ "The First Fundamental Theorem of Asset Pricing (FFTAP) establishes an
equivalence relation between no-arbitrage (static and dynamic) and the exis-
tence of an equivalent martingale measure (EMM). After the landmark work
of Harrison and Kreps [22], there are various versions of the FFTAP and ex-
tensions of the no-arbitrage concept (e.g. no free lunch by Kreps [28], no free
lunch with vanishing risk by Delbaen and Schachermaye"
+ Seems to be for observed call prices over (T, K)
+ effectively project calls to arbitrage-free region via LP programming
+ https://github.com/vicaws/arbitragerepair 


Arbitrage-free SVI volatility surfaces

N. Kahale. An arbitrage-free interpolation of volatilities. Risk Magazine,
17:102–106, 2004

M. Fengler. Arbitrage-free smoothing of the implied volatility surface.
Quantitative Finance, 9:417–428, 06 2009

Deep Smoothing of the Implied Volatility Surface

+ https://proceedings.neurips.cc/paper/2020/hash/858e47701162578e5e627cd93ab0938a-Abstract.html

Incorporating Prior Financial Domain Knowledge into Neural Networks for Implied Volatility Surface Prediction
+ https://arxiv.org/abs/1904.12834
+ Conditions on implied volatility
+ Smile-like activation function

Evaluation of Deep Learning Algorithms for Quadratic Hedging

+ https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4062101

