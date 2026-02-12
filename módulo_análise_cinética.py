# Importação de bibliotecas do python
import numpy as np
import scipy as scp
from scipy.constants import R  # m3*Pa/mol*K

# Importação de outros módulos deste projeto
from módulo_composições import normalizar_composição


def calcular_yield_tempo_infinito(x_solv, x_onset, yield_max, ks_x):
    return yield_max / (1 + ks_x[0] * np.exp(-(x_solv - x_onset)/ks_x[1]))


def calcular_yields_temporais(tempos, x_solv, x_onset, yield_max, ks_x, ks_t):
    tempos = np.asarray(tempos)[:, None]
    tau = ks_t[0] * (x_solv ** ks_t[1])
    yields_t_inf = calcular_yield_tempo_infinito(x_solv, x_onset, yield_max, ks_x)
    return yields_t_inf * (1 - np.exp(- tempos / tau))


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

    par_cin_tempo_infinito = [683.75, 0.030]
    par_cin_temporais = [0.2801, -8.132]
    lista_t = [2, 4, 6, 8, 16, 24]
    max_yield_eq, onset_x_value = 0.063419, 0.50
    lista_x_solv = np.linspace(0.05, 0.95)

    yields_t_infinito = calcular_yield_tempo_infinito(lista_x_solv, onset_x_value, max_yield_eq, par_cin_tempo_infinito)
    yields_ao_longo_do_tempo_calc = calcular_yields_temporais(lista_t, lista_x_solv, onset_x_value, yields_t_infinito,
                                                              par_cin_tempo_infinito, par_cin_temporais)

    yields_experimentais = np.zeros(len(lista_x_solv))
    yields_ao_longo_do_tempo_exp = np.zeros((len(lista_t), len(lista_x_solv)))

    informações_auxiliares = ['predicao', 'TESTE']
    plotar_yield_curves_cinéticas(lista_x_solv, lista_t, yields_experimentais, yields_ao_longo_do_tempo_exp,
                                  yields_ao_longo_do_tempo_calc, informações_auxiliares)

# FIM DO TESTE
# ******************************************************************************************************************** #
