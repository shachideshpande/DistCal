import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import numpy as np
import pdb
from .basic import Calibrator
from .. import _get_prediction_device


class DiscreteDistCalibrator(Calibrator):
    """ The class to recalibrate probabilistic prediction over discrete outcomes

    DiscreteDistCalibrator takes as input probabilistic outcome over K classes (p_1, p_2,..,p_K). The recalibrated outcome is also provide in the same form. 

    Args:
        verbose (bool): if verbose=True print detailed messsages 

    """
    def __init__(self, verbose=False, platt_scaling=False):
        super(DiscreteDistCalibrator, self).__init__(input_type='continuous')
        self.verbose = verbose
        if(platt_scaling==True):
            # self.model = LogisticRegression(random_state=0, C=5, solver='liblinear', penalty='l1', max_iter=1000)
            self.model = LogisticRegression(random_state=0, C=5, max_iter=1000)
        else:
            self.model = MLPClassifier(random_state=0, max_iter=1000, activation='tanh') # learning_rate='adaptive'

    def train(self, train_predictions, train_labels, val_predictions=None, val_labels=None, *args, **kwargs):
        """ Trains the recalibrator using independent calibration dataset

        Args:
            train_predictions (tensor): Training batch of probabilistic outcomes over K classes in shape [batch_size, K]
            train_labels (tensor): Training batch of K discrete labels with shape [batch_size]
            val_predictions (tensor): Holdout batch of probabilistic outcomes over K classes in shape [batch_size, K]
            val_labels (tensor): Holdout batch of K discrete labels with shape [batch_size]
        """
        

        self.model.fit(train_predictions, train_labels)


    def __call__(self, predictions, *args, **kwargs):
        """ Use the learned model to calibrate the predictions. 

        Only use this after calling DiscreteDistCalibrator.train. 

        Args:
            predictions (tensor): a batch of probabilistic outcomes over K classes in shape [batch_size, K]

        Returns:
            tensor: the calibrated probabilistic outcomes over K classes in shape [batch_size, K]

        """
        return self.model.predict_proba(predictions)



    

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




