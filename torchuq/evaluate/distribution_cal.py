
import numpy as np
from sklearn.metrics import log_loss
import torch
def discrete_cal_loss(y_true, y_prob):
    """ 
    Log loss for probabilistic output over discrete outcomes 
    """
    return log_loss(y_true, y_prob)

def discrete_cal_score(y_true, y_prob):
    """ 
    Calibration score for probabilistic output over discrete outcomes (computed as weighted sum) 

    https://arxiv.org/pdf/2112.07184
    """
    cdf = np.cumsum(y_prob, axis=1)
    
    cdf = np.array([cdf[i][y_true[i]] for i in range(len(y_true))])
    taus = np.linspace(0, 1, num=11)
    result=0
    for tau in taus:
      tau_hat = (cdf<=tau).sum()/(len(cdf))
      # weight of each bucket is also tau_hat. can be set to 1
      result+=((tau_hat-tau)**2)*tau_hat

    return result

def compute_empirical_cdf(cdf):
    """
    This function takes the output CDF as produced by probabilistic model corresponding to each observed outcome in the dataset to produce empirical CDF 
    This will be used to perform quantile calibration in Algorithm 1 of Kuleshov 2018
    """

    empirical_cdf = (torch.Tensor([torch.sum(cdf<=p) for p in cdf]))/len(cdf)

    return empirical_cdf

def comparison_quantile_calibration_scores(x_data, y_data, taus=np.linspace(0, 1, num=11), model=None, old_quantiles=None):
    """ 
    Computes weighted calibration scores before and after applying recalibrator in the continuous outcome setting where distribution is featurized using equispaced quantiles. 
    """
    # Todo: remove redundant parts
    if x_data is None and y_data is None:
        return (0, 0)
    
    if old_quantiles is None:
      old_quantiles = x_data
    
    num_buckets = len(taus)
    expected_tau = taus
    empirical_tau = np.zeros(num_buckets)
    empirical_tau_old = np.zeros(num_buckets)
    for tau_i, tau in enumerate(expected_tau):
        q_old = old_quantiles[:, tau_i]
        if model:
          q_new = model.model.predict((tau, x_data))[0]
        else:
          q_new = q_old
       

        # Todo: write efficient code
        for outcome_i, outcome in enumerate(y_data):

            if(outcome<=q_new[outcome_i]):
                empirical_tau[tau_i]+=1
            if(outcome<=q_old[outcome_i]):
                empirical_tau_old[tau_i]+=1
        empirical_tau[tau_i]/=len(y_data)
        empirical_tau_old[tau_i]/=len(y_data)

        empirical_tau[len(empirical_tau)-1]=1
        empirical_tau_old[len(empirical_tau)-1]=1
        cal_score_before = (((empirical_tau_old-expected_tau)**2)*empirical_tau_old).sum()
        cal_score_after = (((empirical_tau-expected_tau)**2)*empirical_tau).sum()

    return (cal_score_before, cal_score_after)

def comparison_param_calibration_scores(x_q_data, x_data, y_data, taus=np.linspace(0, 1, num=11), model=None, old_quantiles=None):
    """ 
    Computes weighted calibration scores before and after applying recalibrator in the continuous outcome setting where distribution is featurized using parameters of the distribution. 
    """
    if x_data is None and y_data is None:
        return (0, 0)
    if old_quantiles is None:
      old_quantiles = x_q_data
    num_buckets = len(taus)
    expected_tau = taus
    empirical_tau = np.zeros(num_buckets)
    empirical_tau_old = np.zeros(num_buckets)
    for tau_i, tau in enumerate(expected_tau):
       
        q_old = old_quantiles[:, tau_i]

        if model:
          q_new = model.model.predict((tau, x_data))[0]
        else:
          q_new = q_old

        # Todo: write a more efficient code below
        for outcome_i, outcome in enumerate(y_data):

            if(outcome<=q_new[outcome_i]):
                empirical_tau[tau_i]+=1
            if(outcome<=q_old[outcome_i]):
                empirical_tau_old[tau_i]+=1
        empirical_tau[tau_i]/=len(y_data)
        empirical_tau_old[tau_i]/=len(y_data)


        empirical_tau[len(empirical_tau)-1]=1
        empirical_tau_old[len(empirical_tau)-1]=1

        cal_score_before = (((empirical_tau_old-expected_tau)**2)*empirical_tau_old).sum()
        cal_score_after = (((empirical_tau-expected_tau)**2)*empirical_tau).sum()
    return (cal_score_before, cal_score_after)

def comparison_quantile_check_score(x_data, y_data, taus=np.linspace(0, 1, num=11), model=None, quant_calibrated_outcome=None):
    """ 
    Computes check scores before and after applying recalibrator in the continuous outcome setting where distribution is featurized using equispaced quantiles. 
    """
    if x_data is None and y_data is None:
        return 0, 0
    check_score_before, check_score_after = 0, 0
    for i, tau in enumerate(taus):
        temp_before = check_score(tau, x_data[:, i], y_data).mean()
        if model:
          calibrated_outcome = model.model.predict((tau, x_data))[0]
        else:
          calibrated_outcome = quant_calibrated_outcome[:, i]
          # calibrated_outcome = x_data[:, i]
        temp_after = check_score(tau, calibrated_outcome, y_data).mean()
        check_score_before += temp_before
        check_score_after += temp_after
  
    return check_score_before, check_score_after.detach()



def comparison_param_check_score(x_q_data, x_data, y_data, taus=np.linspace(0, 1, num=11), model=None):
    """ 
    Computes check scores before and after applying recalibrator in the continuous outcome setting where distribution is featurized using parameters of the distribution. 
    """
    if x_data is None and y_data is None:
        return 0, 0
    check_score_before, check_score_after = 0, 0
    for i, tau in enumerate(taus):
        temp_before = check_score(tau, x_q_data[:, i], y_data).mean()
        if model:
          calibrated_outcome = model.model.predict((tau, x_data))[0]
        else:
          calibrated_outcome = x_q_data[:, i]
        temp_after = check_score(tau, calibrated_outcome, y_data).mean()
        check_score_before += temp_before
        check_score_after += temp_after
  
    return check_score_before, check_score_after.detach()


def check_score(tau, inverse_cdf, y):
    """ 
    Computes check scores

    """
    selector = (inverse_cdf>y.squeeze()).int()

    return (selector*(inverse_cdf - y.squeeze())*(1-tau) + (1-selector)*(y.squeeze() - inverse_cdf)*(tau))