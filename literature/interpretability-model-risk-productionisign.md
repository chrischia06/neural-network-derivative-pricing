# Interpretability, Model Risk, Productionising

These papers are about the model risk, how to interpret neural networks for finance, and how to productionise models

**Interpretability in deep learning for finance (2021)**
+ https://arxiv.org/abs/2104.09476
+ https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3829947
+ MSc Thesis version: https://www.imperial.ac.uk/media/imperial-college/faculty-of-natural-sciences/department-of-mathematics/math-finance/Xiaoshan_Huang_Thesis.pdf
+ Heston
+ 10000 datapoints, parameter space from Horvath et al. (2019) and Roeder and Dimitrof, sampled uniformly in ever ydimension
+ Uniform grid of strikes and maturities
+ FCNN the data is scaled in a range from 0 to 1 before being inputted to the network, while
for the CNN the data is scaled to have a mean of 0 and a variance of 1
+ Local and global interpretability

Jorg Kietniz how deep are financial models?	







Black-Box Model Risk in Finance
+ https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3782412

https://www.sciencedirect.com/science/article/pii/S0378426609001824


https://assets.kpmg/content/dam/kpmg/pdf/2016/06/KPMG-Whitepaper-Model-Risk-Management-2016.pdf

https://www.pwc.com/gx/en/financial-services/pdf/fs-model-monitoring.pdf

http://people.maths.ox.ac.uk/obloj/RT2018/Talks/Talks%20Tue/Morini%20RTQF%20Oxford.pdf

Hanging Up the Phone - Electronic Trading in Fixed Income Markets and its Implications

The new OTC derivatives landscape: (more) transparency, liquidity, and electronic trading (2020)


## Productionising

**Making Deep Learning Go Brrrr From First Principles**
+ https://horace.io/brrr_intro.html