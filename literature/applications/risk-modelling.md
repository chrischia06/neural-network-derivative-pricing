# PFE


Kristoffer Andersson, Cornelis Oosterlee, A deep learning approach for computations of exposure profiles for high-dimensional Bermudan options (2021)

+ https://www.sciencedirect.com/science/article/pii/S0096300321004215
+ https://arxiv.org/abs/2003.01977

PIERRE HENRY-LABORDeRE, Optimal Posting of Collateral with Recurrent Neural Networks (2018
+ https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3140327

# XVA / CCR

Kathrin Glau, Ricardo Pachon, Christian Potz, Speed-up credit exposure calculations for pricing and risk management

Alessandro Gnoatto, Athena Picarelli, Christoph Reisinger, Deep xVA solver -- A neural network based counterparty credit risk management framework (2020)

+ https://arxiv.org/abs/2005.02633
+ http://dse.univr.it/home/workingpapers/wp2020n7.pdf
+ https://www.youtube.com/watch?v=O9bEx_RXhyI

**Balance Sheet XVA by Deep Learning and GPU**
+ https://math.maths.univ-evry.fr/crepey/papers/Deep-XVA-Analysis-SHORT.pdf
+ https://www.youtube.com/watch?v=Pmo3syXg2tc


**Sven Welack, Artificial Neural Network Approach to Counterparty Credit Risk and XVA (2019)**
+ https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3312944

## CVA

Neural Network for CVA: Learning Future Values
+ https://arxiv.org/abs/1811.08726

A deep learning-based high-order operator splitting method for high-dimensional nonlinear parabolic PDEs via Malliavin calculus: application to CVA computation
+ https://ieeexplore.ieee.org/abstract/document/9776096

Deep learning for CVA computations of large portfolios of financial derivatives
+ https://www.sciencedirect.com/science/article/pii/S0096300321004884


Deep learning for CVA computations of large portfolios of financial derivatives Kristoffer Andersson, Cornelis W. Oosterlee (2021)
https://www.sciencedirect.com/science/article/pii/S0096300321004884

## MVA

Deep MVA: Deep Learning for Margin Valuation Adjustment of Callable Products
+ https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3634059
+ IHS Markit



# VaR

Pietro Rossi, Flavio Cocco, Giacomo Bormetti, Deep learning Profit & Loss (2020)

+ https://arxiv.org/abs/2006.09955

Learning Value-at-Risk and Expected Shortfall
+ https://perso.lpsm.paris/~crepey/papers/learning-var-es.pdf

Neural Networks and Value at Risk
+ https://arxiv.org/abs/2005.01686

Estimating Future VaR from Value Samples and Applications to Future Initial Margin
+ https://arxiv.org/abs/2104.11768

DeepVaR: a framework for portfolio risk assessment leveraging probabilistic deep neural networks
+ https://link.springer.com/article/10.1007/s42521-022-00050-0

Estimating Value-at-Risk Using Neural Networks
+ https://link.springer.com/chapter/10.1007/978-3-642-60327-3_28


# Initial Margin / ISDA SIMM

Initial Margin Simulation with Deep Learning
+ https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3357626


Deep Learning-Based Method for Computing Initial Margin
+ https://www.mdpi.com/2673-4591/7/1/41
+ https://mdpi-res.com/d_attachment/engproc/engproc-07-00041/article_deploy/engproc-07-00041.pdf?version=1634624574

**Deep Primal-Dual Algorithm for BSDEs: Applications of Machine Learning to CVA and IM**

+ https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3071506


## Regresion approaches

Regression Sensitivities for Initial Margin Calculations (2016)
+ https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2763488

Alexandre Antonov, Serguei Issakov, and Andrew McClelland, Efficient SIMM-MVA Calculations for Callable Exotics
+ https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3040061


Asif Lakhany, Amber Zhang, Efficient ISDA Initial Margin Calculations Using Least Squares Monte-Carlo (2021)
+ https://arxiv.org/abs/2110.13296

Mathematical Foundations of Regression Methods for Approximating the Forward Initial Margin
+ https://arxiv.org/pdf/2002.04563.pdf

Computing MVA via regression and principal component analysis
+ https://www.d-fine.com/fileadmin/user_upload/pdf/inights/whitepapers/Computing-MVA-via-regression-and-principal-component-analysis-2017.pdf

The Impact of Initial Margin on Derivatives Pricing with an Application of Machine Learning
+ https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3427230

# CCAR 

Heng Z. Chen, An interpretable Comprehensive Capital Analysis and Review (CCAR) neural 
network model for portfolio loss forecasting and stress testing  

+ "• An interpretable NN model is proposed for the CCAR loss forecasting and stress testing per regulatory requirement. • The model interpretability is achieved by solving constrained optimization at a small cost to the performance in mean squared errors. • Based on a time series charge-offs dataset from a major US credit cards company, the interpretable NN model outperforms the benchmarking ARIMA model and maintains its interpretability."

+ "This paper proposes an interpretable nonlinear neural network (NN) model that translates business regulatory requirements into model constraints. The model is then compared with linear and nonlinear NN models without the constraint for Comprehensive Capital Analysis and Review (CCAR) loss forecasting and scenario stress testing. Based on a monthly time series data set of credit card portfolio chargeoffs, the model outperforms the benchmark linear model in mean squared errors, and the improvement increases with network architecture complexity. However, the NN models could be vulnerable to overfitting, which could make the model uninterpretable. The constrained NN model ensures model interpretability at a small cost to model performance. Thus, it is insufficient to measure the model’s statistical performance without ensuring model interpretability and clear CCAR scenario narratives."
+ Link: https://www.researchgate.net/profile/Heng-Chen-14/publication/353938125_An_interpretable_Comprehensive_Capital_Analysis_and_Review_CCAR_neural_network_model_for_portfolio_loss_forecasting_and_stress_testing/links/61817714eef53e51e11d88a1/An-interpretable-Comprehensive-Capital-Analysis-and-Review-CCAR-neural-network-model-for-portfolio-loss-forecasting-and-stress-testing.pdf
+ Slides: https://www.mathworks.com/content/dam/mathworks/mathworks-dot-com/company/events/conferences/matlab-computational-finance-conference-nyc/2019/towards-interpretable-neural-networks-hsbc-h-chen.pdf
+ Seems to be using a neural network as a time series model

CCAR-Consistent∗Yield Curve Stress Testing: From Nelson-Siegel to Machine Learning
+ https://www.risk.net/journal-of-risk-model-validation/7878286/comprehensive-capital-analysis-and-review-consistent-yield-curve-stress-testing-from-nelson-siegel-to-machine-learning
+ "In stress testing, especially when using a top-down approach, it is common to design scenarios for a core set of variables. When applying these scenarios to a specific area, one needs to translate the changes in the core variables to additional variables that better capture specific risk. In this paper, we clearly define a scenario translation problem for interest rate variables. The same idea can be applied to other situations when one needs to expand a list of stressed variables. Given the need to estimate additional variables, considerable effort is given to determining the most efficient approach incorporating both computational time and accuracy of results. By examining three different methods one can choose the approach that best suits their needs. Our analysis results in three viable alternatives to solving the translation problem for interest rate variables. ANN did not significantly outperform PCA, with its performance likely constrained by dataset size. PCA draws a good balance between complexity and performance, compared to the other two approaches. Outside of CCAR, ANN is promising when simulated data is utilized and inherent factors in the dataset become more complex. Its flexibility and ability to generalize makes it a good candidate for generic stress testing problems."
+ https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3408228