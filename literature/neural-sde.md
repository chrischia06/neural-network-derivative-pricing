# Neural SDEs

**Summary**:

+ They can be used as a generative model (i.e. simulate MC paths)
+ There is also a connection with GANs

**Robust Neural SDE**

+ https://github.com/msabvid/robust_nsde

**Timothy DeLise, Neural Options Pricing (2021)**

+  https://arxiv.org/abs/2105.13320
+ Abstract: "We treat neural SDEs as universal Itô process approximators. In this way we can lift all assumptions on the form of the underlying price process, and compute theoretical option prices numerically. We propose a variation of the SDE-GAN approach by implementing the Wasserstein distance metric as a loss function for training. Furthermore, it is conjectured that the error of the option price implied by the learnt model can be bounded by the very Wasserstein distance metric that was used to fit the empirical data. "


**Neural SDEs as Infinite-Dimensional GANs**

+ http://proceedings.mlr.press/v139/kidger21b/kidger21b.pdf


**Neural Stochastic Partial Differential Equations** 

+ https://arxiv.org/abs/2110.10249






**Efficient and Accurate Gradients for Neural SDEs**

+ https://proceedings.neurips.cc/paper/2021/hash/9ba196c7a6e89eafd0954de80fc1b224-Abstract.html

**Neural SDEs as Infinite-Dimensional GANs**

+ http://proceedings.mlr.press/v139/kidger21b.html






**LEI FAN, MACHINE LEARNING METHODS FOR PRICING AND HEDGING FINANCIAL
DERIVATIVES,  Phd Thesis (2021)**
https://www.ideals.illinois.edu/bitstream/handle/2142/113868/FAN-DISSERTATION-2021.pdf?sequence=1&isAllowed=y

+ Supervised by Sirignano
+ A sort of neural-SDE + neural PDE hybrid approach 
+ e.g. use a NN to learn a pricing map f(K, t) to the option / implied vol surface. Then a NN-local volatility diffusion term can be obtainend by considering Dupire's formula
+ use a NN to learn drift dynamics directly, the generate many MC paths, then compute MSE beween MC prices and call surface (standard neural-SDEs)
+ Not entirely clear what the MSE / MAPE metric is on. Forecast errors all exceed 10% MAPE (the lowest is for the Neural SDE). If the metric is not how well different vol models can fit / calibrate to the surface?  Or suggesting that out of sample MC prices (which suggests that NN SDEs have more realistic volatility dynamics) are better when parameters are unchanged (no recalibration of BS, local, heston, and NN SDEs).
+ Author suggests recalibrating the NN leads to better results
+ NN SDEs can be used to obtain delta hedges (standard pathwise differential for MC)
+ NN SDEs can be used on multiple payoffs
+ The oos testing approach is based on using different underlyings



Option Pricing by Neural Stochastic Differential Equations: A Simulation-Optimization Approach
+ https://ieeexplore.ieee.org/abstract/document/9715493?casa_token=4lhAPjJbRPsAAAAA:TBm6r04u-9zvxwjHPZGpKYZLIw63GH9JvbxCz4B0Sq0LOWHxlsxMVCLapKgzrX0VkOFS2zdgp6E


**Samuel N. Cohen, Christoph Reisinger, Sheng Wang Detecting and repairing arbitrage in traded option prices** 

+ https://arxiv.org/abs/2008.09454
+ https://github.com/vicaws/arbitragerepair
+ Observed surface for call (mid)-prices for equity index options
+ Use linear programming to project prices to no-arb polytope

**Samuel N. Cohen, Christoph Reisinger, Sheng Wang, Arbitrage-free neural-SDE market models**
+ https://arxiv.org/abs/2105.11053
+ https://github.com/vicaws/neuralSDE-marketmodel
+ Uses a neural SDE as a factor model for the call-price
+ Project arbitragable prices into a no-arb polytope

**Samuel N. Cohen, Christoph Reisinger, Sheng Wang, Estimating risks of option books using neural-SDE market models (2022)**

+ https://arxiv.org/pdf/2202.07148v1.pdf
+ https://arxiv.org/abs/2202.07148