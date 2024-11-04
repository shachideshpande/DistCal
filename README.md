## Distribution Calibration with TorchUQ

> We extend TorchUQ, an extensive library for uncertainty quantification (UQ) based on pytorch, to implement distribution calibration.



## Installation 

To run the code, you can install the dependencies with the following command:

```bash
conda env create -f env_nobuilds.yml
```

## Example code run

Below, we use Bayesian Ridge Regression as the base model to predict the mean and standard deviation of Gaussian as the outcome distribution. 

We use object of the *DistCalibrator* class to train a recalibrator that takes the probabilistic outcome from the base model and outputs the recalibrated distribution represented by equispaced quantiles. In this example, we use 20 equispaced quantiles to featurize the outcome distribution. 

We use an independent calibration dataset to train the DistCalibrator. We evaluate the quality of probabilistic uncertainty with check score and calibration score as defined [here](https://arxiv.org/pdf/2112.07184). 

```
import torch
from torchuq.transform.distcal_continuous import *
from sklearn.linear_model import BayesianRidge
from torchuq.evaluate.distribution_cal import *
from torchuq.dataset.regression import *

dataset = get_regression_datasets("cal_housing", val_fraction=0.2, test_fraction=0.2, split_seed=0, normalize=True, verbose=True)

train_dataset, cal_dataset, test_dataset = dataset
X_train, y_train = train_dataset[:][0], train_dataset[:][1]
X_cal, y_cal = cal_dataset[:][0], cal_dataset[:][1]
X_test, y_test = test_dataset[:][0], test_dataset[:][1]

quantiles_cal = convert_normal_to_quantiles(mean_cal, std_dev_cal, num_buckets)
quantiles_test = convert_normal_to_quantiles(mean_test, std_dev_test, num_buckets)

# Train the base  model
reg = BayesianRidge().fit(X_train, y_train)

# Train the distribution calibration model on an independent calibration dataset
calibrator = DistCalibrator(num_buckets = num_buckets, quantile_input=True, verbose=True)
calibrator.train(quantiles_cal, torch.Tensor(y_cal))
```


