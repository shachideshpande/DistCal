import torch, torchuq
from torchuq.evaluate import distribution 
from torchuq.transform.conformal import ConformalCalibrator 
from torchuq.dataset import create_example_regression

predictions, labels = create_example_regression()

calibrator = ConformalCalibrator(input_type='distribution', interpolation='linear')
calibrator.train(predictions, labels)
adjusted_predictions = calibrator(predictions)

distribution.plot_density_sequence(predictions, labels, smooth_bw=10);
distribution.plot_density_sequence(adjusted_predictions, labels, smooth_bw=10);

 
