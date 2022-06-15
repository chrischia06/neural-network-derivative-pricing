# Deep Hedging

**Summary**: 

+ Use a neural network as an approximation for a hedging strategy. Buehler et al popularised the idea in their paper "Deep Hedging". 
+ This can be thought of as semi-supervised learning, or reinforcement learning, although not in the sense of Q-learning or other RL methods. 
+ In terms of derivatives pricing, this can be viewed as pricing via replication in an incomplete, discrete time version of some volatility / market model. 
+ Sensitivities can be obtained by automatic differentiation of the neural network (i.e. MC pathwise differentials). 
+ However, both pricingn aand sensitivities these require simulation of MC paths, making it likely too slow for pricing / risk applications. 
+ However, the strategy may be more optimal than a model hedge. Furthermore, the hedging strategy can account for transaction costs, and high dimensionality.
+ In Buehler et al Deep Hedging, they provide examples for only fixed vol model parameters (e.g. rho, vol-of-vol, kappa, theta in Heston), but this approach can likely be extended to account for multiple vol model parameters


**Tobias Pedersen - A deep study of deep hedging**

**Ziheng Chen, RLOP: RL Methods in Option Pricing from a Mathematical Perspective (2022)**
+ **Code**: https://github.com/owen8877/RLOP




**Evaluation of Deep Learning Algorithms for Quadratic Hedging (2022)**
+ https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4062101

The Efficient Hedging Frontier with Deep Neural Networks

+ https://arxiv.org/abs/2104.05280

Deep Learning Algorithms for Hedging with Frictions

+  https://arxiv.org/pdf/2111.01931v3.pdf
+ https://github.com/InnerPeas/ML-for-Transaction-Costs

+ Asian Options https://github.com/sanj909/Hedging-Asian-Options


Neural networks for option pricing and hedging:
a literature review

http://eprints.lse.ac.uk/104341/1/Ruf_Wang_Literature_Review.pdf


https://people.maths.ox.ac.uk/hambly/PDF/Papers/RL-finance.pdf

Distributional Reinforcement Learning

Deep Hedging under Reweighted asset measure
+ https://www.imperial.ac.uk/media/imperial-college/faculty-of-natural-sciences/department-of-mathematics/math-finance/SHI-WINTER_01256799.pdf

Option hedging with risk averse reinforcement learning
+ https://dl.acm.org/doi/abs/10.1145/3383455.3422532?casa_token=48hPdNZdT7YAAAAA:lohnJA3tspR8NZ_mT7aQl_tUAt8iHmxJtEBxuf8aW7LO61CR-NTYQCUglWuubt87NMbfQ16U_viNBg

pfhedge

+ https://github.com/pfnet-research/pfhedge


Pricing and Hedging of Derivatives by Unsupervised Deep Learning
+ https://www.imperial.ac.uk/media/imperial-college/faculty-of-natural-sciences/department-of-mathematics/math-finance/Peixuan_Qin_01772192.pdf


Equal risk pricing of derivatives with deep hedgingOpen Data
+ https://www.tandfonline.com/doi/full/10.1080/14697688.2020.1806343?casa_token=P8sk_uBGcKEAAAAA%3A-qVUFHsp9jQIOKrVBIRMVz4wao8veOeNVoSfr2moTOK9WHkGngUl8trV2Csxj94JeJ4k0xTyohg 

## Reinforcement Learning

Summary: These papers focus on using reinforcement learning

Kolm, P. N. and Ritter, G. (2019). Dynamic replication and hedging: A reinforcement
learning approach.

Mikkilä, O. (2020). Optimal hedging with continuous action reinforcement learning.


Optimal Option Hedging with Policy Gradient	Bo Xiao, Wuguannan Yao, Xiang Zhou	2021	https://ieeexplore.ieee.org/abstract/document/9679868?casa_token=dg7OXWH65U8AAAAA:UuoGeEqiHOFnnvRxpSeeMnw9zf1S9hDyeFlzo7KRjCrbEyWKHcS4U4CH05kh8EwlAN6rCGothkmbhA