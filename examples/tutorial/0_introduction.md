
## Foreword

In high stakes applications, people want predictions that they can trust. An important aspect of trust is accurate uncertainty quantification --- a prediction should contain accurate information about what it knows, and what it does not know. 

Uncertainty quantification is a very active area of research, with hundreds of new publications each year in the machine learning community alone. This tutorial aims to cover both classical methods for uncertainty quantification (such as scoring rules, calibration, conformal inference) and some recent developments (such as multi-calibration, decision calibration, advanced conformal methods). This tutorial will also focus on how to use the torchuq software package to easily implement these methods with a few lines of code. Specifically, we will go through the following topics: 

1. How to represent uncertainty, and how to learn different uncertainty predictions from data.
2. How to evaluate and visualize the quality of uncertainty predictions, such as calibration, coverage, scoring rules, etc. 
3. How to obtain calibrated probability in i.i.d. / distribution shift / online adversarial setups. 
4. How to measure group calibration or multi-calibration and how to implement algorithms to achieve them.

We hope this tutorial will give you an overview of frequentist uncertainty quantification. This tutorial does not cover other paradigms such as Bayesian methods, belief theory, betting theory, etc. 

**Background** This tutorial aims to be as self contained as possible and will introduce the key concepts as we go. Required backgrounds include undergraduate level understanding of machine learning / statistics, and familiarity with Pytorch (if you have not used Pytorch before I would recommend first going through the [basic tutorial](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)). 




## Outline of the Tutorial

The tutorial consists of the following topics. You can click the link to access each tutorial. All tutorials are made with jupyter, so you can also download the repo to run the tutorial interactively. 

1. Regression 
    a. Representing and evaluating predictions
    b. Learning predictions with proper scoring rules
    c.1 Conformal inference
    c.2 Conformal calibration
    d. Online prediction beyond exchangeability 
    
2. Classification 
    a. Representing and evaluating predictions
    b. Calibration and conformal inference
    c. Zoo of calibration definitions
    
3. Advanced topics
    a. Multicalibration and fairness
    b. Decision making under uncertainty
    
    
