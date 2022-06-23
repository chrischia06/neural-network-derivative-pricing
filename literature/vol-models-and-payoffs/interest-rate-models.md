# Hull-White

**Andres Hernandez (2016), Model Calibration with Neural Networks**
**Andres Hernandez (2017) Model Calibration: Global Optimizer vs. Neural Network**

+ Paper: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2812140
+ Paper2: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2996930
+ slides: https://www.quantlib.org/slides/qlum17/hernandez.pdf
+ Hull white, bermudan swaptions, compared vs Quantlib
+ Code: https://github.com/Andres-Hernandez/CalibrationNN
+ Risk.net article: https://www.risk.net/derivatives/5288126/model-calibration-with-neural-networks
+ Model output seems to be the Hull-White parameters, inputs are 13 × 12 swaption ATM matrix, and an IR curve
+ FFN vs CNN architectures. "Due to the success of CNNs in image recognition problems, it was natural to try them, but for the small ’image’ size, it did not provide much improvement over simpler FNNs, while increasing the learning time significantly."
+ Some discussion of architectures search: Grid search, manual search
+ This (when input dimensionality is large) "is where neural networks excel, and where other methods can falter, e.g. interpolation tables"
+ {"optimizer":"NAdam", "learning_rate":1e-3, "dropout_rate":0.2, "EarlyStopping": 50, "activation":"elu", "hidden_layers": 4, "hidden_units":64}
+ Dataset size: 150,000
+ "Targets", the Hull-White paramters Obtained against a Levenberg–Marquardt local optimizer 
+ "Hence, sampling from a parametrized correlation structure, could extend the life-span of a trained model"



**Jorg Kienitz, Sarp Kaya Acar, Qian Liang, Nikolai Nowaczyk, Deep Option Pricing - Term Structure Models	 (2019)**

+ https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3498398
+ Discussion of Control variate
+ Hull-White
+ https://www.youtube.com/watch?v=pm1nIPfNI9I
+ https://www.wbstraining.com/wp-content/uploads/2020/05/New-Stochastic-Volatility-Models-Kienitz.pdf

## 2-Factor Hull-White

**Andres Hernandez, Model Calibration: Global Optimizer vs. Neural Network (2017)**
+ 2-Factor Hull-White

**Deep calibration of interest rates model (2021)**

+ https://arxiv.org/abs/2110.15133
+ 2 factor hull white
+ output is the model parameters
+ "The methods we propose perform very quickly (less than 0.3 seconds for 2 000 calibrations) and have low errors and good fitting."

**Luca SABBIONI, Neural Network calibration of the two-additive factor Gaussian model. A Machine Learning approach to Swaption pricing (2018)**

+  MSc Thesis at Banca IMI
+ https://www.politesi.polimi.it/bitstream/10589/142618/3/2018_10_Sabbioni.pdf	

**Calibrating the Mean-Reversion Parameter in the Hull-White Model Using Neural Networks: Methods and Protocols**

+ https://www.researchgate.net/publication/330915787_Calibrating_the_Mean-Reversion_Parameter_in_the_Hull-White_Model_Using_Neural_Networks_Methods_and_Protocols
+ MSC Thesis version at ING: https://scripties.uba.uva.nl/document/659005



# HJM

**Accuracy of Deep Learning in Calibrating HJM Forward Curves (2021)**

+ SPDEs
+ Calibration to full HJM is very difficult
+ Pointwise vs Grid
+ Code: https://github.com/silvialava/HJM_calibration_with_NN
+ Commodities/Energy markets
+ **FRED ESPEN BENTH, NILS DETERING, LUCA GALIMBERTI, NEURAL NETWORKS IN FReCHET SPACES**
+ https://arxiv.org/abs/2109.13512
+ Related Work: **FRED ESPEN BENTH, NILS DETERING, LUCA GALIMBERTI, PRICING OPTIONS ON FLOW FORWARDS BY NEURAL NETWORKS IN HILBERT SPACE**


**Anastasis Kratsios, Cody B. Hyndman, Deep Learning in a Generalized HJM-type Framework Through Arbitrage-Free Regularization (2019)**
+ Nelson Siegel



# LIBOR Market Model

**Differential Machine Learning**
+ deltix.io/diff-ml.html

**Neural calibration of the DDSVLMM interest rates model**




# Uncategorised

DeepPricing: pricing convertible bonds based on financial time-series generative adversarial networks
+ https://jfin-swufe.springeropen.com/articles/10.1186/s40854-022-00369-y

# Credit Modelling

Gerardo Manzo, Xiao Qiao, Deep Learning Credit Risk Modeling, The Journal of Fixed Income Fall 2021, jfi.2021.1.121; DOI: 

+ https://doi.org/10.3905/jfi.2021.1.121
+ https://github.com/gmanzog/DeepLearningCreditRiskModeling
+ Seems to be a NN with 3 hidden layers trained on Heston
+ "This article demonstrates how deep learning can be used to price and calibrate models of credit risk. Deep neural networks can learn structural and reduced-form models with high degrees of accuracy. For complex credit risk models with no closed-form solutions available, deep learning offers a conceptually simple and more efficient alternative solution. This article proposes an approach that combines deep learning with the unscented Kalman filter to calibrate credit risk models based on historical data; this strategy attains an in-sample R-squared of 98.5% for the reduced-form model and 95% for the structural model.""
+ "Key Findings: 1)Neural networks can approximate solutions to credit risk models, precisely capturing the relationship between model inputs and credit spreads. 2) Compared to standard techniques, the approximate solutions are more computationally efficient. 3) Neural networks can be used to accurately calibrate structural and reduced-form models of credit risk.

The Vasicek Credit Risk Model: A Machine Learning Approach
+ https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3836719

The generalized Vasicek credit risk model: A Machine Learning approach☆
+ https://www.sciencedirect.com/science/article/pii/S1544612321005705

Machine Learning Vasicek Model Calibration with Gaussian Processes











Artificial neural networks for interest and exchange rates models calibration
+ 2 Factor Hull-White
+ https://www.politesi.polimi.it/bitstream/10589/151717/3/Tesi%20Elia%20Mazzoni.pdf
+ MSc Thesis


Fast direct calibration of interest rate derivatives pricing models
+ 2 Factor Hull-White
+ https://dl.acm.org/doi/abs/10.1145/3383455.3422534