# Neural Network Construction Methods for Derivatives Modellings

Investment Banks and Market Makers make heavy use of pricing functions for derivatives. 

For different applications, there may be a different tradeoff between

Towards this, neural networks can be applied in several ways, not limited on:

1. Pre-train a neural network offline on prices / greeks. This can be done for any vanilla Europeans, or even weakly path-dependent payoffs (barriers, asians) but likely not for more complex payoff structures (e.g. Americans), given the need for one neural net per payoff contract. (For the American case, a neural network is needed for each exercise structure)
2. Solve a pricing problem on-the-fly, at a significantly quicker speed (and sufficient accuracy) than existing methods. 


Given unlimited compute power, we could brute-force. However, is there a way


+ [Neural Network Theory](literature/nn-theory): Neural Network Theory: Universal Approximation, Activation Functions

# Volatility / Market Models and Payoffs

## Interest Rate Models

+ [Interest Rate Models](literature/vol-models-and-payoffs/interest-rate-models.md)

**Interest Rate Model Calibration**
+ Andres Hernandez, Model Calibration with Neural Networks (2016) examines calibrating Hull-White with a neural network

**Credit Risk Modelling**
+ Gerardo Manzo, Xiao Qiao, Deep Learning Credit Risk Modeling (2021) explores calibrating credit risk models to Heston and other vol models with Deep Learning, and has code available at: https://github.com/gmanzog/DeepLearningCreditRiskModeling

# Applications

## Risk Modelling

+ [Risk Modelling](literature/applications/risk-modelling.md)

**XVA**
+ ALESSANDRO GNOATTO, ATHENA PICARELLI, AND CHRISTOPH REISINGER, DEEP XVA SOLVER – A NEURAL NETWORK BASED COUNTERPARTY CREDIT RISK MANAGEMENT FRAMEWORK (2021)





## Alternative Methods

Aside from using neural networks

+ [Gaussian Processes](literature/gaussian-process): Using Gaussian Process Regression for derivatives pricing and risks
+ [Tensor Methods](literature/tensor-methods): Using tensor methods.


# TOdo

+ add numeraire, stoch control formulations
+ Abstract