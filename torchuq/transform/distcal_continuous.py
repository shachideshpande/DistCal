import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import pdb
from tqdm import tqdm
from .basic import Calibrator
from torchuq.evaluate.distribution_cal import *
from .. import _get_prediction_device


def convert_normal_to_quantiles(mean, std_dev, num_buckets):
    """ Converts prediction in the form of Normal distribution over outcomes to equispaced quantiles equal to num_buckets

        Args:
            mean (tensor): a batch of scalar means
            std_dev (tensor): a batch of scalar outcomes
            
        Returns:
            tensor: batch of cdf predictions represented as equispaced quantiles with the shape [batch_size, num_buckets]  
        """
    normal_dist = Normal(mean, std_dev)
    quantiles = normal_dist.icdf(torch.tensor([[i*(1.0/(num_buckets-1))] for i in range(0, num_buckets, 1)])).T
    quantiles[:, 0] = mean - 6*std_dev
    quantiles[:, -1] = mean + 6*std_dev
    return quantiles

def convert_normal_cdf_to_quantiles(mean, std_dev, cdf_values):
    """ Converts prediction in the form of Normal distribution over outcomes to equispaced quantiles equal to num_buckets

        Args:
            mean (tensor): a batch of scalar means
            std_dev (tensor): a batch of scalar outcomes
            
        Returns:
            tensor: batch of cdf predictions represented as equispaced quantiles with the shape [batch_size, num_buckets]  
        """
    normal_dist = Normal(mean, std_dev)
    
    quantiles=torch.zeros((mean.shape[0], cdf_values.shape[0]))

    for i, cdf_value in enumerate(cdf_values):
        x = normal_dist.icdf(cdf_value)
        quantiles[:, i] = x.permute(*torch.arange(x.ndim - 1, -1, -1))
        # quantiles[:, i] = normal_dist.icdf(cdf_value).T
    quantiles[:, -1] = mean+6*std_dev 
    return quantiles

class EarlyStopper:
    """ Early stopping after validation loss begins to degrade
    """
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class ContinuousQuantileRecalibrator(nn.Module):
    """ Recalibrator model for continuous prediction and CDF featurized as equispaced quantiles

    ContinuousQuantileRecalibrator takes as input cdf prediction over continuous outcome, provided in the form of equispaced quantiles and the quantile for which CDF value is queried. The outcome is recalibrated CDF value corresponding to the input quantile. 

    Args:
        num_buckets (int): Number of equispaced buckets as featurization of input CDF forecast

    """
    def __init__(self, num_buckets=11):
        super(ContinuousQuantileRecalibrator, self).__init__()
        self.num_buckets = num_buckets
        self.linear_relu_stack = nn.Sequential(
                              nn.Linear(num_buckets+1, num_buckets*2),
                              nn.ReLU(),
                              nn.Linear(num_buckets*2, 2),
                              )


    def forward(self, inputs, targets):
        """ Forward pass, also computes check score as loss function and adds l2 regularization
        """
        tau, quantiles = inputs
        mu, sigma = self.predict(inputs)
        return check_score(tau, mu, targets).mean() +self.add_l2_regularization(0.005)

    def add_l1_regularization(self, l1_lambda):
        """ l1 regularization
        """
        l1_reg = 0
        for param in self.parameters():
            l1_reg = l1_reg + torch.norm(param, 1)
        return l1_lambda * l1_reg

    def add_l2_regularization(self, l2_lambda):
        """ l2 regularization
        """
        l2_reg = 0
        for param in self.parameters():
            l2_reg = l2_reg + torch.norm(param, 2)**2
        return l2_lambda * l2_reg

    def predict(self, inputs):
        """ Computes the CDF output corresponding to input quantile and input CDF
        """
        tau, quantiles = inputs
        inputs = torch.cat((torch.ones(quantiles.shape[0], 1)*tau, quantiles), 1)
        outcome = self.linear_relu_stack(inputs)
        mu, logsigma = outcome[:, 0], outcome[:, 1]
        sigma = torch.exp(logsigma)
        return mu, sigma


class ContinuousNormalRecalibrator(nn.Module):
    """ Recalibrator model for continuous prediction and normally distributed uncertainty envelope. CDF featurized using mean and std deviation

    ContinuousQuantileRecalibrator takes as input cdf prediction over continuous outcome, provided in the form of mean and std deviation along with the the quantile for which CDF value is queried. The outcome is recalibrated CDF value corresponding to the input quantile. 

    Args:
        num_params (int): Number of parameters required to represent the distribution. Here, num_params will be 2, but this class can be modified to represent other parametric distributions. 

    """
    def __init__(self, num_params=2):
        super(ContinuousNormalRecalibrator, self).__init__()
        self.num_params = num_params
        self.linear_relu_stack = nn.Sequential(
                              nn.Linear(num_params+1, num_params*2),
                              nn.ReLU(),
                              nn.Linear(num_params*2, 2),
                              )


    def forward(self, inputs, targets):
        """ Forward pass, also computes check score as loss function and adds l2 regularization
        """
        tau, params = inputs
        mu, sigma = self.predict(inputs)
        return check_score(tau, mu, targets).mean() +self.add_l2_regularization(0.001)

    def add_l1_regularization(self, l1_lambda):
        """ l1 regularization
        """
        l1_reg = 0
        for param in self.parameters():
            l1_reg = l1_reg + torch.norm(param, 1)
        return l1_lambda * l1_reg

    def add_l2_regularization(self, l2_lambda):
        """ l2 regularization
        """
        l2_reg = 0
        for param in self.parameters():
            l2_reg = l2_reg + torch.norm(param, 2)**2
        return l2_lambda * l2_reg

    def predict(self, inputs):
        """ Computes the CDF output corresponding to input quantile and input CDF
        """
        tau, params = inputs
        inputs = torch.cat((torch.ones(params.shape[0], 1)*tau, params), 1)
        outcome = self.linear_relu_stack(inputs)
        mu, logsigma = outcome[:, 0], outcome[:, 1]
        sigma = torch.exp(logsigma)
        return mu, sigma

class DistCalibrator(Calibrator):
    """ The class to recalibrate a continuous prediction with quantile regression

    DistCalibrator takes as input cdf prediction over continuous outcome, provided in the form of equispaced quantiles. The outcome is also provide in the form of same number of equispaced quantiles

    Args:
        verbose (bool): if verbose=True print detailed messsages
        num_buckets (int): number of equispaced quantiles as featurization of CDF
        num_params (int): number of parameters representing input distribution
        quantile_input (bool): If True, then input CDF is featurized as 'num_buckets' quantiles, else featurized using 'num_params' parameters of distribution

    """
    def __init__(self, num_buckets=11, num_params=2, quantile_input=True, verbose=False):
        super(DistCalibrator, self).__init__(input_type='continuous')
        self.verbose = verbose
        self.model = None
        self.quantile_input=quantile_input
        self._set_recalibrator(num_buckets, num_params)

    def _set_recalibrator(self, num_buckets=11, num_params=2):
        """ Sets the recalibrator such that input CDF is represented either using 'num_buckets' equispaced quantiles or 'num_params' parameters
        """
        self.num_buckets = num_buckets
        if self.quantile_input:
            self.model = ContinuousQuantileRecalibrator(num_buckets=num_buckets)
            
        else:
            self.model = ContinuousNormalRecalibrator(num_params=num_params)
            self.num_params = num_params


    def train(self, train_predictions, train_labels, val_predictions=None, val_labels=None, *args, **kwargs):
        """ Trains the recalibrator using independent calibration dataset

        Args:
            train_predictions (tensor): Training batch of continuous outcome cdf predictions with shape [batch_size, num_buckets]
            train_labels (tensor): Training batch of continuous labels with shape [batch_size]
            val_predictions (tensor): Holdout batch of continuous outcome cdf predictions with shape [batch_size, num_buckets]
            val_labels (tensor): Holdout batch of continuous labels with shape [batch_size]
        """
        # Note: can employ cross-val based approach here if needed in future. For now, assume that we have independent calibration dataset and evaluate on holdout
        # Set train parameters
        num_epochs = kwargs.get('num_epochs', 1500)
        log_interval = kwargs.get('log_interval', 100)
        patience=kwargs.get('es_patience', 50)
        min_delta=kwargs.get('es_delta', 0.0005)
        lr = kwargs.get('lr', 0.0001)
        reps = kwargs.get('repetitions', 100)
        best_loss = None
        best_model = None
        # Train the recalibrator
        for iteration in range(reps):
            # self._set_recalibrator(num_buckets=20)
            
            X_train = train_predictions
            y_train = train_labels
            X_val = val_predictions
            y_val = val_labels


            taus = np.linspace(0, 1, num=self.num_buckets)
            
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

            early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)
            # for epoch in tqdm(range(num_epochs)):
            for epoch in (range(num_epochs)):
		        
                for tau in taus:
                    loss = self.model((tau, X_train), y_train)
		        
                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if self.quantile_input:
                        _, cal_after_val = comparison_quantile_calibration_scores(X_val, y_val, taus, X_val)
                    else:
                        _, cal_after_val = comparison_param_calibration_scores(X_val, y_val, taus, X_val)
                        
                if(epoch%log_interval==0): 
                    pass
                    # can print/collect logs here

                if X_val and early_stopper.early_stop(cal_after_val):
                    break
            
            if best_loss is None:
                best_loss = loss
            elif loss<=best_loss:
                best_model = self.model
                best_loss = loss
            
        # self.model = best_model

            

    def __call__(self, predictions, *args, **kwargs):
        """ Use the learned model to calibrate the predictions. 

        Only use this after calling DistCalibrator.train. 

        Args:
            predictions (tensor): a batch of continuous cdf predictions, with output shape [batch_size, num_buckets]

        Returns:
            tensor: the calibrated continuous prediction, with shape [batch_size, num_buckets]
        """
        taus = np.linspace(0, 1, num=self.num_buckets)
        outcome_predictions = torch.cat([self.model.predict((tau, predictions))[0].reshape(-1, 1) for tau in taus], axis=1)
        return outcome_predictions



    def to(self, device):
        """ Move all assets of this class to a torch device. 

        Args:
            device (device): the torch device (such as torch.device('cpu'))
        """
        device = _get_prediction_device(device)
        if self.model is not None:
            self.model.to(device)
        self.device = device
        return self




