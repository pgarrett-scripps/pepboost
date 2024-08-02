import logging
from typing import Union, List
from xgboost import XGBRegressor

import numpy as np

from .base import _predict, _fine_tune, _train, _uniplot, _r_squared, BasePredictor
from .constants import DEFAULT_RT_MODEL


class RtPredictor(BasePredictor):

    def __init__(self, model: Union[str, XGBRegressor] = DEFAULT_RT_MODEL, verbose: bool = False):

        if isinstance(model, str):
            self.load_model(model)
        else:
            self.model = model

        if not isinstance(self.model, XGBRegressor):
            raise ValueError("Model path must be a string or an instance of XGBRegressor.")

        self.verbose = verbose
        self.logger = logging.getLogger('RtPredictor')
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)
        self.logger.info("ChargePredictor initialized.")

    def predict(self, sequences: List[str]) -> np.ndarray:
        return _predict(model=self.model,
                        sequences=sequences,
                        charges=None,
                        logger=self.logger)

    @classmethod
    def train(cls, sequences: List[str],
              retention_times: List[float]) -> XGBRegressor:

        model = XGBRegressor()
        return _train(model=model,
                      sequences=sequences,
                      labels=retention_times,
                      charges=None)

    def fine_tune(self, sequences: List[str],
                  retention_times: List[float]) -> XGBRegressor:

        return _fine_tune(model=self.model,
                          sequences=sequences,
                          labels=retention_times,
                          charges=None)

    def uniplot(self, sequences: List[str],
                retention_times: List[float], ) -> str:

        return _uniplot(model=self.model,
                        sequences=sequences,
                        labels=retention_times,
                        plot_name='Retention Time',
                        charges=None)

    def r_squared(self, sequences: List[str],
                  retention_times: List[float]) -> float:

        return _r_squared(model=self.model,
                          sequences=sequences,
                          labels=retention_times,
                          charges=None)
