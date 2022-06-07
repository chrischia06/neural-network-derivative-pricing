# Gaussian Process Regression

**Jan De Spiegeleer, Dilip B. Madan, Sofie Reyners, Wim Schoutens, Machine Learning for Quantitative Finance: Fast Derivative Pricing, Hedging and Fitting (2018)**

+ Straightforward application of GPRs to replace the pricing function. In the set of second slides, they also consider gradient boosting machines, although the latter does not have smooth derivatives.
+ Standard: Sample parameters from parameter subspace,
+ *Remark*: Doesn't seem like GPRs can scale to millions of samples
+ Link: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3191050
+ Slides: (1) https://kuleuvencongres.be/eaj2018/documents/presentations/2-reyners-p2-gpr.pdf, (2) https://staff.fnwi.uva.nl/p.j.c.spreij//winterschool/19slidesReyners.pdf


**Stéphane Crépey, Matthew Dixon, Gaussian Process Regression for Derivative Portfolio Modeling and Application to CVA Computations, (2019)**

+ Link: https://arxiv.org/abs/1901.11081
+ Argues that their approach captures the joint dependencies in a portfolio, by considering multi-output GP
+ https://github.com/mfrdixon/GP-CVA

**Mike Ludkovski, Yuri Saporito, KrigHedge: Gaussian Process Surrogates for Delta Hedging (2021)**

+ Link: https://arxiv.org/abs/2010.08407
+ 
+ "Among our key take-aways are the recommendation to use Matern kernels, the benefit of including virtual training points to capture boundary conditions, and the significant loss of fidelity when training on stock-path-based datasets."


Machine learning for pricing American options in high-dimensional Markovian and non-Markovian models.


Use of Kernel Methods for Dynamic Hedging Incomplete Markets 
+ https://www.imperial.ac.uk/media/imperial-college/faculty-of-natural-sciences/department-of-mathematics/math-finance/Tang-Xiaofu_01904015.pdf

**Dynamically Controlled Kernel Estimation for XVA Pricing and Options Replication**
+ https://www.imperial.ac.uk/media/imperial-college/faculty-of-natural-sciences/department-of-mathematics/math-finance/Qingxin_Geng_Thesis.pdf
+ https://www.risk.net/cutting-edge/7929776/podcast-ubss-gordon-lee-on-conditional-expectations-and-xvas



**Joerg Kienitz, GMM DCKE - Semi-Analytic Conditional Expectations (2021)**
+ Gaussian Mean Mixture


**Ludovic Goudenège, Andrea Molent, Antonino Zanette, Machine Learning for Pricing American Options in High-Dimensional Markovian and non-Markovian models (2019)**

+ https://arxiv.org/abs/1905.09474
+ American exercise
+ Fixed set of parameters

