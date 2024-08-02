import logging
from typing import Union, List
from xgboost import XGBRegressor

import numpy as np

from .base import _predict, _fine_tune, _train, _uniplot, _r_squared, BasePredictor
from .constants import DEFAULT_CHARGE_MODEL


class ChargePredictor(BasePredictor):

    def __init__(self, model: Union[str, XGBRegressor] = DEFAULT_CHARGE_MODEL, verbose: bool = False):

        if isinstance(model, str):
            self.load_model(model)
        else:
            self.model = model

        if not isinstance(self.model, XGBRegressor):
            raise ValueError("Model path must be a string or an instance of XGBRegressor.")

        self.verbose = verbose
        self.logger = logging.getLogger('ChargePredictor')
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)
        self.logger.info("ChargePredictor initialized.")

    def predict(self, sequences: List[str], charges: List[int]) -> np.ndarray:
        return _predict(model=self.model,
                        sequences=sequences,
                        charges=charges,
                        logger=self.logger)

    @classmethod
    def train(cls, sequences: List[str],
              charges: List[int],
              intensities: List[float]) -> XGBRegressor:

        model = XGBRegressor()
        return _train(model=model,
                      sequences=sequences,
                      labels=intensities,
                      charges=charges)

    def fine_tune(self, sequences: List[str],
                  charges: List[int],
                  intensities: List[float]) -> XGBRegressor:

        return _fine_tune(model=self.model,
                          sequences=sequences,
                          labels=intensities,
                          charges=charges)

    def uniplot(self, sequences: List[str],
                charges: List[int],
                intensities: List[float], ) -> str:

        return _uniplot(model=self.model,
                        sequences=sequences,
                        labels=intensities,
                        plot_name='Intensity',
                        charges=charges)

    def r_squared(self, sequences: List[str],
                  charges: List[int],
                  intensities: List[float], ) -> float:

        return _r_squared(model=self.model,
                          sequences=sequences,
                          labels=intensities,
                          charges=charges)

