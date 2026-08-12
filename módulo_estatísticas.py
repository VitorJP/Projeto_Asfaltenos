# Importação de bibliotecas do python
import os
import numpy as np
import pandas as pd
import scipy as scp
from dataclasses import dataclass, field, fields


@dataclass
class Erros:

    valores_experimental: np.ndarray
    valores_modelo: np.ndarray

    @property
    def erros(self):
        return self.valores_experimental - self.valores_modelo

    @property
    def AD(self):
        # Absolute Deviation
        return np.abs(self.erros)

    @property
    def AAD(self):
        # Average Absolute Deviation
        return np.nanmean(np.abs(self.erros))

    @property
    def ARD(self):
        # Aboslute Relative Deviation
        return np.abs(self.erros / self.valores_experimental)

    @property
    def AARD(self):
        # Average Absolute Relative Deviation
        return np.nanmean(np.abs(self.erros / self.valores_experimental))

    @property
    def MSE(self):
        # Mean Squared Error
        return np.nanmean(self.erros ** 2) / 2

    @property
    def RMSE(self):
        # Root Mean Squared Error
        return np.sqrt(np.nanmean(self.erros**2))

    @property
    def std(self):
        # Standard Deviation
        return np.std(self.erros, ddof=1)

    @property
    def variância(self):
        return np.var(self.erros, ddof=1)

    @property
    def r2_score(self):
        SS_res = np.sum((self.valores_experimental - self.valores_modelo)**2)
        SS_tot = np.sum((self.valores_experimental - np.nanmean(self.valores_experimental))**2)
        return 1 - (SS_res/SS_tot)

    @property
    def bias(self):
        return np.nanmean(self.erros)
