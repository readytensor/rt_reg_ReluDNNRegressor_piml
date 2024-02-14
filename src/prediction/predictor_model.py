import os
import warnings
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from piml.models import ReluDNNRegressor
from sklearn.exceptions import NotFittedError
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")


PREDICTOR_FILE_NAME = "predictor.joblib"


class Regressor:
    """A wrapper class for the ReluDNNRegressor.

    This class provides a consistent interface that can be used with other
    regressor models.
    """

    model_name = "ReluDNNRegressor"

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        feature_types: Optional[List[str]] = None,
        hidden_layer_sizes: Tuple[int, int] = (40, 40),
        dropout_prob: float = 0.0,
        max_epochs: int = 1000,
        learning_rate: float = 0.001,
        batch_size: int = 500,
        batch_size_inference: int = 10000,
        l1_reg: float = 1e-05,
        val_ratio: float = 0.2,
        n_epoch_no_change: int = 20,
        iht: bool = False,
        phase_epochs: int = 50,
        threshold: float = 0.1,
        verbose: bool = False,
        random_state: int = 0,
        **kwargs,
    ):
        """Construct a new ReluDNNRegressor.

        Args:
            feature_names (Optional[List[str]]): The list of feature names.
            feature_types (Optional[List[str]]): The list of feature types. Available types include “numerical” and “categorical”.
            hidden_layer_sizes (Tuple[int, int]): A list of hidden layer sizes.
            dropout_prob (float): Dropout probability.
            max_epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for model training.
            batch_size (int): Batch size for training.
            batch_size_inference (int):
                The batch size used in the inference stage.
                It is imposed to avoid out-of-memory issue when dealing very large dataset.

            l1_reg (float): lambda parameter for L1 Regularization.
            val_ratio (float): validation ratio for early stopping.
            n_epoch_no_change (int): Stops training is loss doesn't improve for last n_epoch_no_change epochs. This is required when early_stop is True.
            iht (bool): Whether to perform IHT (Iterative Hard Thresholding) or not.
            phase_epochs (int): No of phase 1 and phase 2 epochs for IHT, required when IHT is True.
            threshold (int): Threshold value for performing IHT, required when IHT is True.
            verbose (bool): Whether to display training statistics (loss) or not.
            random_state (int): Determines random number generation for weights and bias initialization.
        """
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.dropout_prob = dropout_prob
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.batch_size_inference = batch_size_inference
        self.l1_reg = l1_reg
        self.val_ratio = val_ratio
        self.n_epoch_no_change = n_epoch_no_change
        self.iht = iht
        self.phase_epochs = phase_epochs
        self.threshold = threshold
        self.verbose = verbose
        self.random_state = random_state
        self.kwargs = kwargs
        self.model = self.build_model()
        self._is_trained = False

    def build_model(self) -> ReluDNNRegressor:
        """Build a new regressor."""
        model = ReluDNNRegressor(
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            hidden_layer_sizes=self.hidden_layer_sizes,
            dropout_prob=self.dropout_prob,
            max_epochs=self.max_epochs,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            batch_size_inference=self.batch_size_inference,
            l1_reg=self.l1_reg,
            val_ratio=self.val_ratio,
            n_epoch_no_change=self.n_epoch_no_change,
            iht=self.iht,
            phase_epochs=self.phase_epochs,
            threshold=self.threshold,
            verbose=self.verbose,
            random_state=self.random_state,
            **self.kwargs,
        )
        return model

    def fit(self, train_inputs: pd.DataFrame, train_targets: pd.Series) -> None:
        """Fit the regressor to the training data.

        Args:
            train_inputs (pandas.DataFrame): The features of the training data.
            train_targets (pandas.Series): The labels of the training data.
        """
        self.model.fit(train_inputs, train_targets)
        self._is_trained = True

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict regression targets for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted regression targets.
        """
        return self.model.predict(inputs)

    def evaluate(self, test_inputs: pd.DataFrame, test_targets: pd.Series) -> float:
        """Evaluate the regressor and return the r-squared score.

        Args:
            test_inputs (pandas.DataFrame): The features of the test data.
            test_targets (pandas.Series): The targets of the test data.
        Returns:
            float: The r-squared score of the regressor.
        """
        if self.model is not None:
            return self.model.score(test_inputs, test_targets)
        raise NotFittedError("Model is not fitted yet.")

    def save(self, model_dir_path: str) -> None:
        """Save the regressor to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Regressor":
        """Load the regressor from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Regressor: A new instance of the loaded regressor.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return (
            f"Model name: {self.model_name} ("
            f"feature_names: {self.feature_names}, "
            f"feature_types: {self.feature_types}, "
            f"fit_intercept: {self.fit_intercept}, "
            f"l1_regularization: {self.l1_regularization}, "
            f"l2_regularization: {self.l2_regularization})"
        )


def train_predictor_model(
    train_inputs: pd.DataFrame, train_targets: pd.Series, hyperparameters: dict
) -> Regressor:
    """
    Instantiate and train the predictor model.

    Args:
        train_X (pd.DataFrame): The training data inputs.
        train_y (pd.Series): The training data targets.
        hyperparameters (dict): Hyperparameters for the regressor.

    Returns:
        'Regressor': The regressor model
    """
    regressor = Regressor(**hyperparameters)
    regressor.fit(train_inputs=train_inputs, train_targets=train_targets)
    return regressor


def predict_with_model(regressor: Regressor, data: pd.DataFrame) -> np.ndarray:
    """
    Predict regression targets for the given data.

    Args:
        regressor (Regressor): The regressor model.
        data (pd.DataFrame): The input data.

    Returns:
        np.ndarray: The predicted regression targets.
    """
    return regressor.predict(data)


def save_predictor_model(model: Regressor, predictor_dir_path: str) -> None:
    """
    Save the regressor model to disk.

    Args:
        model (Regressor): The regressor model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Regressor:
    """
    Load the regressor model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Regressor: A new instance of the loaded regressor model.
    """
    return Regressor.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Regressor, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the regressor model and return the r-squared value.

    Args:
        model (Regressor): The regressor model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The targets of the test data.

    Returns:
        float: The r-sq value of the regressor model.
    """
    return model.evaluate(x_test, y_test)
