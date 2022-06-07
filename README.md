# msc-thesis

**Notes**:

 + Pre-train a neural network offline on prices / greeks. This can be done for any vanilla Europeans, or even weakly path-dependent payoffs (barriers, asians) but likely not for more complex payoff structures (e.g. Americans), given the need for one neural net per payoff contract. (For the American case, a neural network is needed for each exercise structure)
 + Solve a pricing problem on-the-fly, at a significantly quicker speed (and sufficient accuracy) than existing methods. 



+ [Gaussian Processes](gaussian-process): Using Gaussian Process Regression for derivatives pricing and risks