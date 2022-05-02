# msc-thesis


Neural networks for option pricing and hedging:
a literature review

http://eprints.lse.ac.uk/104341/1/Ruf_Wang_Literature_Review.pdf


## Neural Americans

Bernard Lapeyre, Jérôme Lelong. Neural network regression for Bermudan option pricing.
Monte Carlo Methods Appl. 27 (2021), no. 3, 227–247.

https://arxiv.org/pdf/1907.06474.pdf

2. Calypso Herrera, Florian Krach, Pierre Ruyssen, and Josef Teichmann, 2021. ”Optimal
Stopping via Randomized Neural Networks,” Papers 2104.13669, arXiv.org.
3. Becker, Sebastian; Cheridito, Patrick; Jentzen, Arnulf; Welti, Timo. Solving high-dimensional
optimal stopping problems using deep learning. European J. Appl. Math. 32 (2021), no. 3,
470–514. MR4253974 Add to clipboard
4. John Tsitsiklis and Benjamin Van Roy. Regression methods for pricing complex american-
style options. IEEE Transactions on Neural Networks, 12(4):694–703, 200

A new approach for American option pricing: The Dynamic Chebyshev method
https://arxiv.org/abs/1806.05579
https://www.youtube.com/watch?v=FWp1X8m5XX4

## Bermudans
 
+ Bermudan Swaptions in Gaussian HJM One-Factor Model: Analytical and Numerical Approaches https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1287982
+ Pricing Bermudan Swaptions on the LIBOR Market Model
using the Stochastic Grid Bundling Method https://www.nag.com/doc/techrep/pdf/tr2_15.pdf
+ Evaluating Sensitivities of Bermudan Swaptions https://www.maths.ox.ac.uk/system/files/legacy/12804/Schlenkrich_20.3.11.pdf
+ https://arxiv.org/abs/2201.02587

Deep Learning-Based BSDE Solver for LiborMarket Model with Application to Bermudan Swaption Pricing and Hedging

+ https://www.cefpro.com/wp-content/uploads/2019/07/1807.06622.pdf

+ Valuation of Bermudan swaptions with a one-factor
Hull-White model https://essay.utwente.nl/61747/1/MSc_P_Nikolopoulos.pdf

+ Pricing and Hedging American-Style Options with Deep Learning https://mdpi-res.com/d_attachment/jrfm/jrfm-13-00158/article_deploy/jrfm-13-00158-v2.pdf?version=1595308716

+ Deep Optimal Stopping https://jmlr.org/papers/volume20/18-232/18-232.pdf

+ Learning Bermudans https://arxiv.org/abs/2105.00655	

## Neural SDE

+ Samuel N. Cohen, Christoph Reisinger, Sheng Wang, Arbitrage-free neural-SDE market models https://arxiv.org/abs/2105.11053
+ https://github.com/vicaws/neuralSDE-marketmodel
+ Samuel N. Cohen, Christoph Reisinger, Sheng Wang, Detecting and repairing arbitrage in traded option prices https://arxiv.org/abs/2008.09454
+ https://github.com/vicaws/arbitragerepair
+ Samuel N. Cohen, Christoph Reisinger, Sheng Wang, Estimating risks of option books using neural-SDE market models (2022) https://arxiv.org/abs/2202.07148
+ Timothy DeLise Neural Options Pricing https://arxiv.org/abs/2105.13320
+ Neural SDEs as Infinite-Dimensional GANs http://proceedings.mlr.press/v139/kidger21b/kidger21b.pdf
+ Neural Stochastic Partial Differential Equations https://arxiv.org/abs/2110.10249

## Neural PDEs

+ Sirignano Deep PDEs DGM: A deep learning algorithm for solving partial differential equations https://arxiv.org/abs/1708.07469
+ Neural Q-learning for solving elliptic PDEs (2022) https://arxiv.org/abs/2203.17128
+ The Deep Parametric PDE Method: Application to Option Pricing https://arxiv.org/pdf/2012.06211.pdf https://github.com/LWunderlich/DeepPDE/blob/main/TwoAssetsExample/DeepParametricPDEExample.ipynb
+ DNN Expression Rate Analysis of High-Dimensional PDEs: Application to Option Pricing https://link.springer.com/article/10.1007/s00365-021-09541-6
+ Physics-Informed Neural Networks and Option Pricing Andreas Louskos https://math.dartmouth.edu/theses/undergrad/2021/Louskos-thesis.pdf
+ Deep Learning of High-dimensional Partial Differential Equations
+ https://medium.com/data-analysis-center/solving-differential-equations-using-neural-networks-with-pydens-e831f1a115f


## No Arbitrage
+ Cash settled swaptions https://www.risk.net/sites/risk/files/import_unmanaged/risk.net/data/risk/pdf/technical/2008/risk_0208_technical_briefcomm.pdf


## Universal Approximators

+ https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4077158


​​Pricing options on flow forwards by neural networks in Hilbert space (2022)
https://arxiv.org/abs/2202.11606
https://www.youtube.com/watch?v=jOye0mznjqc



Projection of Functionals and Fast Pricing of Exotic Options
https://arxiv.org/abs/2111.03713


## Deep Hedging

+ Deep Learning Algorithms for Hedging with Frictions https://arxiv.org/pdf/2111.01931v3.pdf

+ Asian Options https://github.com/sanj909/Hedging-Asian-Options

Path functionals

## Initial Margin

+ https://mdpi-res.com/d_attachment/engproc/engproc-07-00041/article_deploy/engproc-07-00041.pdf?version=1634624574
+ https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3357626
+ https://arxiv.org/pdf/2002.04563.pdf
+ https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3071506
+ https://education.wbstraining.com/pluginfile.php/7794/mod_resource/content/3/DIM%20webinar%20v5.pdf
+ Deep xVA solver - A neural network based counterparty
credit risk management framework http://dse.univr.it/home/workingpapers/wp2020n7.pdf


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

Deep Arbitrage-Free Learning in a Generalized HJM Framework via Arbitrage-Regularization

+ https://mdpi-res.com/d_attachment/risks/risks-08-00040/article_deploy/risks-08-00040-v2.pdf?version=1587701662


https://www.pure.ed.ac.uk/ws/portalfiles/portal/29546775/yang_yu.pdf

Option valuation under no-arbitrage constraints with neural networks

http://eprints.nottingham.ac.uk/65405/1/Combine.pdf

https://www.tandfonline.com/doi/full/10.1080/14697688.2018.1490807


Differential Deep Learning for
Pricing Exotic Financial Derivatives
ERIK ALEXANDER ASLAKSEN JONASSON
https://www.diva-portal.org/smash/get/diva2:1591933/FULLTEXT01.pdf
https://www.sciencedirect.com/science/article/abs/pii/S0377221720310134



Deep Local Volatility
https://mdpi-res.com/d_attachment/risks/risks-08-00082/article_deploy/risks-08-00082.pdf?version=1596445027


https://scholar.google.com/citations?hl=en&user=ES1uzHIAAAAJ&view_op=list_works&sortby=pubdate


Machine Learning in Finance:
Applications of Continuous Depth
and Randomized Neural Networks
	


Optimal Stopping via Randomized Neural Networks
+ https://arxiv.org/pdf/2104.13669.pdf
https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/529739/3/thesis_calypso_herrera_20220128_withoutCV.pdf


https://people.maths.ox.ac.uk/hambly/PDF/Papers/RL-finance.pdf

Distributional Reinforcement Learning


https://www.imperial.ac.uk/mathematics/postgraduate/msc/mathematical-finance/project-and-thesis/	


https://www.imperial.ac.uk/media/imperial-college/faculty-of-natural-sciences/department-of-mathematics/math-finance/TOMAS_MEHDI_01390785.pdf


https://www.mathematik.hu-berlin.de/~perkowsk/files/bsde.pdf


https://www.imperial.ac.uk/media/imperial-college/faculty-of-natural-sciences/department-of-mathematics/math-finance/Qingxin_Geng_Thesis.pdf


https://www.imperial.ac.uk/media/imperial-college/faculty-of-natural-sciences/department-of-mathematics/math-finance/SHI-WINTER_01256799.pdf

https://scholar.google.com/citations?hl=en&user=LGkKGsAAAAAJ&view_op=list_works&sortby=pubdate


https://www.imperial.ac.uk/media/imperial-college/faculty-of-natural-sciences/department-of-mathematics/math-finance/Tang-Xiaofu_01904015.pdf

https://www.imperial.ac.uk/media/imperial-college/faculty-of-natural-sciences/department-of-mathematics/math-finance/Cheng_Luo-thesis.pdf


Pricing Options using Deep Neural Networks from a Practical Perspective: A Compr
https://www.imperial.ac.uk/media/imperial-college/faculty-of-natural-sciences/department-of-mathematics/math-finance/Pu-Viola_Ruo_Han_01977026.pdf
https://github.com/violapu/OPNN
https://github.com/violapu/Optimally-Weighted-PINNs

CORNELIS OOSTERLEE


https://ipythonquant.wordpress.com/2015/05/02/exposure-simulation-pfe-and-cva-for-multi-callable-swaps-bermudan-swaptions-part-1-of-3/


Riskfuel
+ https://riskfuel.com/riskfuels-new-pricing-demo-the-bermudan-swaption/
+ 


https://quantlib.wordpress.com/tag/bermudan-swaption/

https://github.com/lballabio/QuantLib-SWIG/blob/master/Python/examples/bermudan-swaption.py

Deeply Learning Derivatives
https://arxiv.org/pdf/1809.02233.pdf


LIBOR PROMPTS QUANTILE LEAP:
MACHINE LEARNING FOR QUANTILE DERIVATIVES


Deep Learning for Exotic Option Valuation
+ https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3776322

Accuracy of deep learning in calibrating HJM forward curves
https://link.springer.com/article/10.1007/s42521-021-00030-w	

Extensive networks would eliminate the demand for pricing formulas
+ https://www.sciencedirect.com/science/article/pii/S0950705121010698?casa_token=cIYxKlLBGzUAAAAA:Q6zidLH9WwxIWd4XcoRM70uFo1FScABAzKiKDHT85jcO_U2wFg2yWYtVofFK74W-1V93PMhB-mU

Option Pricing With Machine Learning
Daniel Alexandre Bloch
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3486224

Black-Box Model Risk in Finance
+ https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3782412	


Deep MVA: Deep Learning for Margin Valuation Adjustment of Callable Products
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3634059


Differential Deep Learning for Pricing Exotic Financial Derivatives
https://www.diva-portal.org/smash/get/diva2:1591933/FULLTEXT01.pdf

Machine Learning and Option Implied Information
+ https://spiral.imperial.ac.uk/bitstream/10044/1/57953/5/Yu-Z-2018-PhD-Thesis.pdf


Dalvir Singh Mandara https://www.datasim.nl/application/files/8115/7045/4929/1423101.pdf

The Performance of Artificial Neural Networks on Rough Heston Model https://www.datasim.nl/application/files/3516/0614/0758/Finalthesis.pdf

The Swap Market Model with Local Stochastic Volatility
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2912558


The Co-Terminal Swap Market Model with Bergomi Stochastic Volatility
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3237914