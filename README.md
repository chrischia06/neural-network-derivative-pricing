# Neural Network Construction Methods for Derivatives Modellings

Investment Banks and Market Makers make heavy use of pricing functions for derivatives. 

For different applications, there may be a different tradeoff between

Towards this, neural networks can be applied in several ways, not limited on:

1. Pre-train a neural network offline on prices / greeks. This can be done for any vanilla Europeans, or even weakly path-dependent payoffs (barriers, asians) but likely not for more complex payoff structures (e.g. Americans), given the need for one neural net per payoff contract. (For the American case, a neural network is needed for each exercise structure)
2. Solve a pricing problem on-the-fly, at a significantly quicker speed (and sufficient accuracy) than existing methods. 


Given unlimited compute power, we could brute-force. However, is there a way


+ [Neural Network Theory](literature/nn-theory): Neural Network Theory: Universal Approximation, Activation Functions

## Alternative Methods

Aside from using neural networks

+ [Gaussian Processes](literature/gaussian-process): Using Gaussian Process Regression for derivatives pricing and risks
+ [Tensor Methods](literature/tensor-methods): Using tensor methods.