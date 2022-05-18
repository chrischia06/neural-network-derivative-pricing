# msc-thesis


**Katia Babbar, William McGhee, A DEEP LEARNING APPROACH TO EXOTIC OPTION PRICING UNDER LSVOL**

+ https://www.bayes.city.ac.uk/__data/assets/pdf_file/0007/494080/DeepLearningExoticOptionPricingLSVOL_KB_CassBusinessSchool_2019.pdf

## No Arbitrage
+ Cash settled swaptions https://www.risk.net/sites/risk/files/import_unmanaged/risk.net/data/risk/pdf/technical/2008/risk_0208_technical_briefcomm.pdf

MC 
+ Obtain all payoffs (T, K) for one set of parameters (e.g. (S0, r, sigma))
+ with Deep Hedging, Learn strategy directly from data
+ Use NN to generate MC paths via GANs, Neural SDEs

PDE
+ Solution only available at gridpoints, need some interpolation between
+ Does not scale well to large dimensions
+ Use NN to solve PDE (Neural PDEs)

Calibration
Use NNs (Horvath, Deep Learning Volatility)

Functional Approximation
Use Nns to learn output of some pricing method (e.g. Lo), e.g. from parameters. Reduces time vs NN gride

Issues
Training, Convergence
No arbitrage bounds, Asymptotics
Architecture
Generalisation
Robustness

Expectation and Price in Incomplete Markets

+ https://arxiv.org/pdf/2006.16703.pdf

Josef Teichmann

https://gtr.ukri.org/projects?ref=studentship-2435698


Differential Deep Learning for Pricing Exotic Financial Derivatives ERIK ALEXANDER ASLAKSEN JONASSON
https://www.diva-portal.org/smash/get/diva2:1591933/FULLTEXT01.pdf
https://www.sciencedirect.com/science/article/abs/pii/S0377221720310134


Backward Stochastic Differential Equations: an Introduction
https://www.mathematik.hu-berlin.de/~perkowsk/files/bsde.pdf





Mikko Pakanen
https://scholar.google.com/citations?hl=en&user=LGkKGsAAAAAJ&view_op=list_works&sortby=pubdate




Pricing Options using Deep Neural Networks from a Practical Perspective: A Compr
+ https://www.imperial.ac.uk/media/imperial-college/faculty-of-natural-sciences/department-of-mathematics/math-finance/Pu-Viola_Ruo_Han_01977026.pdf
+ https://github.com/violapu/OPNN
+ https://github.com/violapu/Optimally-Weighted-PINNs

Riskfuel
+ https://riskfuel.com/riskfuels-new-pricing-demo-the-bermudan-swaption/

Quantlib bermudan swaption
+ https://ipythonquant.wordpress.com/2015/05/02/exposure-simulation-pfe-and-cva-for-multi-callable-swaps-bermudan-swaptions-part-1-of-3/
+ https://quantlib.wordpress.com/tag/bermudan-swaption/

+ https://github.com/lballabio/QuantLib-SWIG/blob/master/Python/examples/bermudan-swaption.py

Deeply Learning Derivatives
https://arxiv.org/pdf/1809.02233.pdf
+ https://developer.nvidia.com/blog/accelerating-python-for-exotic-option-pricing/


LIBOR PROMPTS QUANTILE LEAP: MACHINE LEARNING FOR QUANTILE DERIVATIVES

Deep Learning for Exotic Option Valuation
+ https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3776322

Extensive networks would eliminate the demand for pricing formulas
+ https://www.sciencedirect.com/science/article/pii/S0950705121010698?casa_token=cIYxKlLBGzUAAAAA:Q6zidLH9WwxIWd4XcoRM70uFo1FScABAzKiKDHT85jcO_U2wFg2yWYtVofFK74W-1V93PMhB-mU
+ Deep learning SABR volatilities

On the Calibration of the SABR Model and its Extensions (MSc Thesis)
+ https://www.imperial.ac.uk/media/imperial-college/faculty-of-natural-sciences/department-of-mathematics/math-finance/Cheng_Luo-thesis.pdf

Option Pricing With Machine Learning
Daniel Alexandre Bloch
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3486224



Differential Deep Learning for Pricing Exotic Financial Derivatives
https://www.diva-portal.org/smash/get/diva2:1591933/FULLTEXT01.pdf


Physics-Informed Neural Networks and Option Pricing
Andreas Louskos

+ https://math.dartmouth.edu/theses/undergrad/2021/Louskos-thesis.pdf

Derivatives Pricing via Machine Learning
+ https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3352688


Deep Learning for Exotic Option Valuation

+ https://jfds.pm-research.com/content/4/1/41
+ https://arxiv.org/abs/2103.12551

Machine Learning for Quantitative Finance: Fast Derivative Pricing, Hedging and Fitting
+ Gaussian Process
+ https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3191050


Pricing Exotic Options with Flow-based Generative Networks
+ https://www.researchgate.net/profile/Jeonggyu-Huh/publication/353479642_Pricing_Exotic_Options_with_Flow-based_Generative_Networks/links/60ff7e7e169a1a0103bc4d71/Pricing-Exotic-Options-with-Flow-based-Generative-Networks.pdf

Deep learning exotic derivatives
+ https://www.diva-portal.org/smash/record.jsf?pid=diva2%3A1515504&dswid=-3976
+ https://www.bloomberg.com/professional/blog/ghosts-in-the-machines-neural-nets-exotic-options-and-model-risk/

Exotic Derivatives and Deep Learning
+ https://www.diva-portal.org/smash/record.jsf?pid=diva2%3A1210175&dswid=775


A machine learning approach to portfolio pricing and risk management for high-dimensional problems
+ https://arxiv.org/pdf/2004.14149v4.pdf

 Evolutionary Algorithms and Computational Methods for Derivatives Pricing 
 + https://discovery.ucl.ac.uk/id/eprint/10068568/

 Multi-Asset Spot and Option Market Simulation
 + https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3980817	


 Risk-Neutral Market Simulation
 + https://arxiv.org/abs/2202.13996

Simulating spot and equity option markets using rough path signatures
+ https://gateway.newton.ac.uk/presentation/2021-03-16/29953

Deep Hedging: Learning Risk-Neutral Implied Volatility Dynamics
+ https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3808555

Deep hedging: learning to simulate equity option markets
+ https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3470756

Deep Hedging: Learning to Remove the Drift under Trading Frictions with Minimal Equivalent Near-Martingale Measures
+ https://arxiv.org/abs/2111.07844


Ruf, Wang

• Homogeneity hint. Garcia and Genc ̧ay [1998, 2000] incorporate a homogeneity hint by considering
an ANN consisting of two parts, one controlled by moneyness and the other controlled by time-to-
maturity.
• Shape-restricted outputs. Dugas et al. [2001, 2009], Lajbcygier [2004], Yang et al. [2017], Huh
[2019], and Zheng et al. [2019] enforce certain no-arbitrage conditions such as monotonicity and
convexity of the ANN pricing function by fixing an appropriate architecture

 Data augmentation. Yang et al. [2017] and Zheng et al. [2019] create additional synthetic options to
help with the training of ANNs.
• Loss penalty. Itkin [2019] and Ackerer et al. [2019] add various penalty terms to the loss function.
Those terms present no-arbitrage conditions. For example, parameter configurations that allow for
calendar arbitrage are being penalise