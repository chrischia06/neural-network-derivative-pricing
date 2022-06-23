**Summary**: 
+ To the best of knowledge, using deep approximations for rough volatility are the only way to achieve calibration / pricing in a feasibly fast time
+

# NNs for rough volatility

Deep learning volatility: a deep neural network perspective on pricing and calibration in (rough) volatility models

+ https://www.tandfonline.com/doi/abs/10.1080/14697688.2020.1817974?casa_token=D8Rma9PRVBEAAAAA%3AfAhvc8rCvbq9O8qORcIibtAr4TKTqzfot-G73C96ZYSVC_vJCF3gaJHSd9y-ZRiJW22KXfcQZKU&journalCode=rquf20

**Mathieu Rosenbaum, Jianfei Zhang, Deep calibration of the quadratic rough Heston model**

+ https://arxiv.org/abs/2107.01611v2

**Chun Kiat Ong, The Performance of Artificial Neural Networks on Rough Heston Model (2020)**
+ MSc Thesis
+ https://www.datasim.nl/application/files/3516/0614/0758/Finalthesis.pdf


Deep Curve-dependent PDEs for affine rough volatility
+ https://arxiv.org/abs/1906.02551

Henry Stone, Calibrating rough volatility models: a convolutional neural network approach (2019)
+ https://www.tandfonline.com/doi/abs/10.1080/14697688.2019.1654126?casa_token=EVhA3y_F21sAAAAA%3A08YC6pPCM-KYQYVfccVT2x4A5Vnetpfv2EzNVnQmPGbKuHG5V9ygUYdblpmGzO7VmysUCtdzRzE&journalCode=rquf20

Christian Bayer, Jinniao Qiu, and Yao Yao, Pricing Options under Rough Volatility with Backward SPDEs
+ https://epubs.siam.org/doi/abs/10.1137/20M1357639?casa_token=8oy3f_jSAs8AAAAA:AoF0X5CIrFqTRNiYWgsNfiDdnKvS9eO8jGo2MgNypfPEazRYsud_epK4b0uoptSquh1IxeKG


Athul AR, Richard Obonyo, Modelling and Calibration Techniques For Fractional Black Scholes Model (2020)
+ https://www.researchgate.net/publication/345858415_Modelling_and_Calibration_Techniques_For_Fractional_Black_Scholes_Model
+ Student thesis

**Medhi Thomas, Pricing and calibration of stochastic models via neural networks**
+ https://www.imperial.ac.uk/media/imperial-college/faculty-of-natural-sciences/department-of-mathematics/math-finance/TOMAS_MEHDI_01390785.pdf
+ Medhi Thomas MSc Thesis

**Deep Learning (Rough) Volatility**

+ Video: https://www.youtube.com/watch?v=O03erV5nYXA
+ https://arxiv.org/abs/1901.09647
+ Article by bocconi students https://bsic.it/rough-volatility/


**Christian Bayer, Benjamin Stemper, Deep calibration of rough stochastic volatility models (2018)**

+ Rough (Bergomi) models presents itself as opportunity, given the lack of analtyical approximations in many cases, necessitating MC
+ https://github.com/roughstochvol
+ https://github.com/bstemper/deep_rough_calibration  
+ Cut up one dataset into train, val, test
+ Sample more from dense parameter regions (can be obtained from historical data)
+ Relu activation
+ Feature scaling,  weight initialisation, regularisation, batch norm
+ Bayesian approach to parameters

**Dirk Roeder, Georgi Dimitroff, Volatility model calibration with neural networks a comparison between direct and indirect methods (2020)**

+ https://github.com/roederd/volatility_model_calibration_with_nn
+ Strike-Maturity grid is fixed (so interpolation through grid points)


DEEP PPDES FOR ROUGH (LOCAL) STOCHASTIC VOLATILITY
+ https://www.ma.imperial.ac.uk/~ajacquie/index_files/JO%20-%20RoughVolPPDE_Final.pdf
+ https://github.com/msabvid/Deep-PPDE


Deep calibration of the quadratic rough Heston model
+ https://arxiv.org/abs/2107.01611

the Performance of Artificial Neural Networks on Rough Heston Model
+ https://www.datasim.nl/application/files/3516/0614/0758/Finalthesis.pdf