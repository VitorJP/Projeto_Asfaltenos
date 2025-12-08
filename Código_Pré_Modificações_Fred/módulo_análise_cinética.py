# Importação de bibliotecas do python
import numpy as np
import scipy as scp
from scipy.constants import R  # m3*Pa/mol*K
from pyswarms.single import GlobalBestPSO
from itertools import combinations, permutations

# Importação de outros módulos deste projeto
from módulo_composições import normalizar_composição


def cinética_de_precipitação(tempos, yield_eq_max, x_solv, parâmetros_cinéticos):

    k1, k2, k3, k4, k5 = parâmetros_cinéticos[0], parâmetros_cinéticos[1], parâmetros_cinéticos[2], \
                         parâmetros_cinéticos[3], parâmetros_cinéticos[4]

    tempo_eq = k1 * (x_solv ** k2)
    yield_eq = yield_eq_max / (1 + k3 * np.exp(-(x_solv - k4)/k5))

    yield_cinética = []
    for t in tempos:
        yield_t = yield_eq * (1 - np.exp(- t / tempo_eq))
        yield_cinética.append(yield_t)
    yield_cinética.append(yield_eq)

    return np.array(yield_cinética)


def fator_de_correção_de_Saidoun():
    pass


def deposição_cumulativa_asfaltenos():
    pass


# ******************************************************************************************************************** #
#  ATENÇÃO: O CÓDIGO A SEGUIR SERÁ EXECUTADO APENAS QUANDO ESTE MÓDULO FOR RODADO COMO SCRIPT PRINCIPAL.               #
#           O CÓDIGO A SEGUIR SERVE PARA CONFERIR SE AS FUNÇÕES DESTE MÓDULO FUNCIONAM CORRETAMENTE.                   #
# ******************************************************************************************************************** #
# INÍCIO DO TESTE
if __name__ == "__main__":
    from módulo_gráficos import plotar_yield_curves_cinéticas

    par_cin = [5, -2, 2.987, 0.663, 0.030]
    lista_tempos = [2, 4, 6, 8, 24]
    max_yield_eq = 0.10
    lista_x_solv = np.linspace(0.05, 0.95)
    curvas_yield_cinéticas = cinética_de_precipitação(lista_tempos, max_yield_eq, lista_x_solv, par_cin)

    yield_experimental = np.zeros(len(lista_x_solv))

    plotar_yield_curves_cinéticas(lista_x_solv, lista_tempos, yield_experimental, curvas_yield_cinéticas)

# FIM DO TESTE
# ******************************************************************************************************************** #
