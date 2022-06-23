
# Industry Usage

DEEP LEARNING FOR DERIVATIVES PRICING: FROM THEORY TO PRACTICE (2021)
+ RiskFuel / ScotiaBank
+ Slides: https://www.cqfinstitute.org/sites/default/files/twood-cqfi-dld-20210406.pdf
+ https://www.risk.net/awards/7736276/technology-innovation-of-the-year-scotiabank
+ Linkedin Discussion https://www.linkedin.com/feed/update/urn:li:activity:6658758503500390400/
+ https://azure.microsoft.com/en-us/blog/azure-gpus-with-riskfuels-technology-offer-20-million-times-faster-valuation-of-derivatives/

JPM
+ Hans Buehler , Deep Hedging https://www.risk.net/awards/7926411/quant-of-the-year-hans-buehler
+ https://www.risk.net/our-take/6880376/the-machines-are-coming-for-your-pricing-models
+ https://www.risk.net/derivatives/6875321/deep-hedging-and-the-end-of-the-black-scholes-era
+ https://www.risk.net/awards/7930626/derivatives-house-of-the-year-jp-morgan

Wells Fargo - Deep BSDE
+ Andres Hernandez - PWC - Interest Rate Calibration
+ William McGhee - Natwest - SABR Calibration
+ Huge, Savine - Danske Bank - Differential method, VaR / Risk calculations
+ Pierre Henry-Labordere - Societe Generale - Optimal posting of collateral with recurrent neural networks https://www.risk.net/our-take/6875086/fishing-collateral-with-neural-nets
+ Piterbarg - NatWest - https://www.risk.net/our-take/7733011/setting-boundaries-for-neural-networks	
+ https://www.risk.net/our-take/7851386/how-xva-quants-learned-to-stop-worrying-and-trust-the-machine
+ CCAR (Paywalled Articles) https://www.risk.net/risk-management/5355856/banks-apply-machine-learning-to-ccar-models https://www.risk.net/risk-management/7881926/more-banks-flirt-with-machine-learning-for-ccar-but-risks-persist

Citi - Neural networks for Exotic Options and Risk

+ PRDC Power reverse dual-currency note
+ Use PCA to visualise sensitivty to factor
+ Slides: https://cfvod.kaltura.com//api_v3//index.php//service//attachment_attachmentAsset//action//serve//attachmentAssetId//1_bcp5ml06//ks//djJ8MjkzNTc3MXyULy7YMB0XZeGfB09MNz6Di1IhyGEQ_WzsIizeRtroXPpyUcl5kMdc0UV230qrUaQfN-wE_pzbirhjDmSh6B9MDw83nHrLqeV006bNz5CyzDChgK0w7EaSUWi4Mlf1NJkRSyKrq_9h5ozidP0Qkzr578vs-CyFrTbnDntLRV8C5418FkeFT1X6Yo3JpuGVQGMGSz-3GV1JPMS7vbMkRFNl
+ Video: https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s32060/

https://cfe.columbia.edu/sites/default/files/content/MLinFin21%20-%20Giovanni%20Faonte_0.pdf

https://www.turing.ac.uk/sites/default/files/2019-04/artificial_intelligence_in_finance_-_turing_report_0.pdf

Alexei Kondratyev - Standard Chartered - modelling curve dynamics using NNs instead of PCA https://www.risk.net/our-take/5667601/how-machine-learning-could-aid-interest-rate-modelling

+ GANs https://arxiv.org/abs/2110.02742

 Barclays (and others) strive for machine learning at quantum speed - https://www.risk.net/technology/7934366/barclays-and-others-strive-for-machine-learning-at-quantum-speed

Quantum Machine Learning in Finance: Time Series Forecasting https://arxiv.org/pdf/2202.00599.pdf

Giovanni Faonte - Goldman Sachs - https://cfe.columbia.edu/sites/default/files/content/MLinFin21%20-%20Giovanni%20Faonte_0.pdf
https://www.youtube.com/watch?v=Gc99AAMCQxY&list=FLXt_X9MajR7-7wPyszir8aw&index=100


# Surveys

Extensive networks would eliminate the demand for pricing formulas
+ https://arxiv.org/pdf/2101.09064.pdf
+ Fittingn NNN to Option Prices for BS / Heston not that meaningful given the ability to obtain closed form prices already. SABR is slightly more meaningful, as both the SABR approximation and NN approximation
+ Authors analyse SABR
+ FDM is more accurate than MC, but possibly biased. Whereas MC simulation is unbiased?
+ Cites, Ferguson and Green and their empirical observation that NNs can denoise MC samples.
+ For a better goodness-of-fit, one should consider several factors, such as data size, network architecture, and the optimization method. Therefore, we try to reduce the MSFE in our experiment by generating enormous data and tuning various hyperparameters for a considerably accurate fit.
+ "We could not find any mentions about the number of epochs, the batch size, the weight initialization method, and the loss values for the training and test datasets. In particular, the types of neural networks tested in the research are fairly limited because the number of the hidden layers for the networks is fixed at one"

