# Importação de bibliotecas do python
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, fields


def absolute_deviations(valores_exp, valores_modelo):
    return np.abs(valores_exp - valores_modelo)


def average_absolute_deviation(valores_exp, valores_modelo):
    return np.nanmean(absolute_deviations(valores_exp, valores_modelo))


def absolute_relative_deviations(valores_exp, valores_modelo):
    return absolute_deviations(valores_exp, valores_modelo) / valores_exp


def average_absolute_relative_deviation(valores_exp, valores_modelo):
    return np.nanmean(absolute_relative_deviations(valores_exp, valores_modelo))
