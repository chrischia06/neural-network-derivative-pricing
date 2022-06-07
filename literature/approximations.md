**Machine Learning in Finance: The Case of Deep Learning for Option Pricing (2017), Das and Culkin (2017)**


+ Stock price (S) $10 – $500, Strike price (K) $7 – $650, Maturity (T ) 1 day to 3 years,, Dividend rate (q) 0% – 3%, Risk free rate (r) 1% – 3%, Volatility (σ) 5% – 90% (annualised?)
+ Use homogeneity of Black-Scholes
+ 4 hidden layers of 100 neurons eaach, LeakyReLU, ELU, ReLU, ELU, dropout 0.25
+ 240,000 option prices vs 60,000 prices

**Supervised Deep Neural Networks (DNNs) for Pricing/Calibration of Vanilla/Exotic Options Under Various Different Processes (2019)**   

+ GBM, Heston, Variance Gamma, VGSA
+ Mean Squared Error
+ Moneyness S_{0} / K, Time to maturity T, risk-free rate, dividend rate
+ S/K [0.8, 1.2], T [1, 3], q = [0, 0.03], r=  [0.01, 0.03],    
+ Quasi Monte Carlo sampling: Halton sequence
+ Train directly on closed-form prices
+ MSE
+ Suggest (but do not explore) ConvNets for Asian / Path-dependent Options
+ "Li and Yuan (2017) have introduced identity mapping, by which SGD always converges to
the global minimum for a 2-layered neural network with ReLU activation function un-
der the standard weight initialization scheme"
+ "Under similar realistic assumptions,
Kawaguchi’s studies showed that all local minima are global minima using nonlinear acti-
vation functions" K. Kawaguchi, Deep learning without poor local minima, arXiv:1605.07110v3
+ "M. Soltanolkotabi, A. Javanmard, J. D. Lee, Theoretical insights into the optimization landscape of over-parameterized shallow neural networks, arXiv preprint arXiv:1707.04926"
+ 4 different training sizes, which are 40,000, 80,000, 160,000, and 240,00
+ Barriers, Americans
+ Test Cases: Interpolation, (2) deep-out-of-the-money, and (3) longer maturity
+ Moneyness v Call prices look linear as opposed to convex
+ Discussion of width vs depth


Derivatives Pricing via Machine Learning 	Tingting Ye, Liangliang Zhang (2019)
+ https://www.scirp.org/journal/paperinformation.aspx?paperid=94637
+ https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3352688
+  BSDEs, PIDEs
+ Hilbert space formulation
+ Does a plot of prediction v price
+ Experiments: Heston for a single set of parameters

Function approximation for option pricing and risk management Methods, theory and applications.	Christian Potz	2020	https://qmro.qmul.ac.uk/xmlui/handle/123456789/71039

+ Chebyshev polynomial have certain useful properties


Zeron, M. and I. Ruiz (2017). Chebyshev Methods for Ultra - efficient Risk Calculations.
MoCaX Intelligence Working Paper.

Gaß, M., K. Glau, M. Mahlstedt, and M. Mair (2015, May). Chebyshev Interpolation for
Parametric Option Pricing. ArXiv e-prints

Deeply Learning Derivatives
+ Ryan Ferguson, Andrew Green
+ https://arxiv.org/abs/1809.02233
+ Basket option
+ Riskfuel, Scotiabank
+ https://www.linkedin.com/feed/update/urn:li:activity:6658758503500390400/
+ https://azure.microsoft.com/en-us/blog/azure-gpus-with-riskfuels-technology-offer-20-million-times-faster-valuation-of-derivatives/
+ https://www.youtube.com/watch?v=ewCDNokzzOU
+ Basket option


Pricing and hedging derivative securities with neural networks and a homogeneity hint, Rene Garcia, Ramazan Gencay (2000)
+ http://yoksis.bilkent.edu.tr/doi_getpdf/articles/10.1016-S0304-4076(99)00018-4.pdf
+ One set of parameters, GBM

Pricing and Hedging Derivative Securities with Neural Networks: Bayesian Regularization, Early Stopping, and Bagging, Ramazan Gençay and Min Qi (2001)

+ 


Option Pricing with Modular Neural Networks

+ http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.87.3384&rep=rep1&type=pdf
+ Nikola Gradojevic, Ramazan Geņcay, Dragan Kukolj
+ the modular neural network seems to be similar to a gated neural network or ensemble methods


"Learning Minimum Variance Discrete Hedging Directly from Market	Ke Nian, Thomas Coleman, Yuying Li	2017

+ Learning from market data
+ radial basis function , MSE



Option Pricing Using Artificial Neural Networks : an Australian Perspective	Tobias Hahn	2014	https://pure.bond.edu.au/ws/portalfiles/portal/18243185/Option_Pricing_Using_Artificial_Neural_Networks.pdf


A neural network approach to option pricing F. Mostafa & T. Dillon
+ https://www.witpress.com/Secure/elibrary/papers/CF08/CF08008FU1.pdf

Pricing and Hedging Derivative Securities with Neural Networks: Bayesian Regularization, Early Stopping, and Bagging Ramazan Gençay and Min Qi 2001
+ https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.379.4379&rep=rep1&type=pdf
+ "Modern" NN training techniques, but remarkable thing is the paper is dated 2001!



**Daniel Bloch Option Pricing With Machine Learning (2019)**
+ https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3486224


Differential Deep Learning for Pricing Exotic Financial Derivatives
+ https://www.diva-portal.org/smash/get/diva2:1591933/FULLTEXT01.pdf

**Differential Machine Learning, Huge, Savine (2020)**
+ https://arxiv.org/pdf/2005.02347.pdf


The CV Makes the Difference – Control Variates for Neural Networks
+ https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3527314

Neural Networks with Asymptotics Control
+ https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3544698

Supervised deep learning in computational finance
+ S Liu PHd Thesis
+ https://pure.tudelft.nl/ws/portalfiles/portal/87037282/Thesis_Liu_final_upload.pdf


Karatas T., Oskoui A., Hirsa A., Supervised deep neural networks for pricing/calibration of vanilla/exotic
options under various different processes

**Deep Learning for Exotic Option Valuation (2021)**

+ https://arxiv.org/abs/2103.12551
+ Get parameters of market (volatility) model and use it to price exotics
+ Prices (or vol surface) of exotics need to be consistent with vanillas
+ "We create a neural network where the inputs are the volatility surface points and the exotic option parameters and the target is the price."
+ **Neural Network**: *Optimizer*: Adam, *Loss*: Mean Absolute Error, EarlyStopping with patience 50
+ https://www.cqfinstitute.org/sites/default/files/Talk%209_John%20Hull_Valuing%20Exotic%20Options%20and%20Estimating%20Model%20Risk%20Quant%20Insights.pdf





# Asymptotics

**Hideharu Funahashi, Artificial neural network for option pricing with and without asymptotic correction (2020)**
+ https://www.tandfonline.com/doi/abs/10.1080/14697688.2020.1812702?journalCode=rquf20
+ Funahashi, H., A chaos expansion approach under hybrid volatility models. Quant. Finance, 2014, "The residual term, D, is a smooth and infinitely differentiable function, which is a sum of the multiplication of polynomials by a CDF and PDF of the standard Brownian motion."


A neural network-based framework for financial model calibration
+ https://mathematicsinindustry.springeropen.com/articles/10.1186/s13362-019-0066-7

Option Pricing
+ https://link.springer.com/chapter/10.1007/978-3-319-51668-4_7