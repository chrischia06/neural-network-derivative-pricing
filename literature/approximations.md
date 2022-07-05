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






Deeply Learning Derivatives
+ Ryan Ferguson, Andrew Green
+ https://arxiv.org/abs/1809.02233
+ Basket option
+ Riskfuel, Scotiabank
+ https://www.linkedin.com/feed/update/urn:li:activity:6658758503500390400/
+ https://azure.microsoft.com/en-us/blog/azure-gpus-with-riskfuels-technology-offer-20-million-times-faster-valuation-of-derivatives/
+ https://www.youtube.com/watch?v=ewCDNokzzOU
+ Basket option
+ https://www.youtube.com/watch?v=LTADvz49ork








"Learning Minimum Variance Discrete Hedging Directly from Market	Ke Nian, Thomas Coleman, Yuying Li	2017

+ Learning from market data
+ radial basis function , MSE



Option Pricing Using Artificial Neural Networks : an Australian Perspective	Tobias Hahn	2014	https://pure.bond.edu.au/ws/portalfiles/portal/18243185/Option_Pricing_Using_Artificial_Neural_Networks.pdf


A neural network approach to option pricing F. Mostafa & T. Dillon
+ https://www.witpress.com/Secure/elibrary/papers/CF08/CF08008FU1.pdf




**Daniel Bloch Option Pricing With Machine Learning (2019)**
+ https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3486224


Supervised deep learning in computational finance
+ S Liu PHd Thesis
+ https://pure.tudelft.nl/ws/portalfiles/portal/87037282/Thesis_Liu_final_upload.pdf


Karatas T., Oskoui A., Hirsa A., Supervised deep neural networks for pricing/calibration of vanilla/exotic
options under various different processes



Option Pricing
+ https://link.springer.com/chapter/10.1007/978-3-319-51668-4_7




Pricing path-dependent exotic options with flow-based generative networks

**Hyun-Gyoon Kim, Se-JinKwon, Jeong-HoonKim, Jeonggyu Huh**

+ https://www.sciencedirect.com/science/article/abs/pii/S1568494622003532

**Pricing options with exponential Lévy neural network**

+ https://www.sciencedirect.com/science/article/pii/S0957417419301617

 Evolutionary Algorithms and Computational Methods for Derivatives Pricing 
 + https://discovery.ucl.ac.uk/id/eprint/10068568/


Bounds on multi asset derivatives via neural networks (2020)
+ European Gaussian basket options under copulas

Pricing options on flow forwards by neural networks in Hilbert space

+ https://arxiv.org/abs/2202.11606

Accelerating Python for Exotic Option Pricing (NVIDIA)
+ Asian Barrier Options
+ mostly consider speedups, no mention of the accuracy 

Deep Structural Estimation
+ Just function approximation under parameters, but under econometric name of structural estimation
+ Discussion on activation functions
+ 

 Unbiased Deep Solvers for Linear Parametric PDEs
 + https://www.tandfonline.com/doi/full/10.1080/1350486X.2022.2030773


**Lucio Fernandez-Arjona, Damir Filipovic, A machine learning approach to portfolio pricing and risk management for high-dimensional problems (2022)**


Deep ReLU network expression rates for option prices in high-dimensional, exponential Lévy models
+ https://link.springer.com/article/10.1007/s00780-021-00462-7


Jeonggyu Huh, Pricing options with exponential Lévy neural network (2019)
+ https://www.sciencedirect.com/science/article/pii/S0957417419301617
+ Levy Models


Option Pricing With Machine Learning
Daniel Alexandre Bloch
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3486224

**Learning the random variables in Monte Carlo simulations with stochastic gradient descent** Machine learning for parametric PDEs and financial derivative pricing 

+ Authors present that in the BS case the Learning Random Variables (LRV) approach outperforms ANN approximation and (Q)MC methods in the L1 and Linfinity norms
+ However, the LRV approach appears to be much more slower

Pricing options on flow forwards by neural networks in Hilbert space (2022)
https://arxiv.org/abs/2202.11606
https://www.youtube.com/watch?v=jOye0mznjqc

Computations in Option Pricing Engines
+ Undergraduate paper
+ https://web.wpi.edu/Pubs/E-project/Available/E-project-050120-154222/unrestricted/Computations-in-Option-Pricing-Engines-MendoncaFilho-Phongsopa-Wotton.pdf

 Financial Option Valuation by Unsupervised Learning with Artificial Neural Networks 
 + https://www.mdpi.com/2227-7390/9/1/46

 Option Pricing Using Artificial Neural Networks
 + MSC Thesis
 + https://research-api.cbs.dk/ws/portalfiles/portal/59791650/625879_Masters_Thesis_MHP.pdf

 Using Deep Neural Networks to Price Basket and Rainbow Options
 + https://github.com/christiangruener/MasterThesis


Deep calibration of financial models: turning theory into practice
+ https://link.springer.com/content/pdf/10.1007/s11147-021-09183-7.pdf


Parameter estimation method of option pricing model based on convolutional neural network in high frequency financial trading
+ https://link.springer.com/article/10.1007/s10479-022-04582-x

RLOP: RL Methods in Option Pricing from a Mathematical Perspective
+ https://arxiv.org/abs/2205.05600

"Ratku, A., Neumann, D. Derivatives of Feed-Forward Neural Networks and their Application in Real-Time Market Risk Management"
+ https://github.com/antalratku/nn_deriv