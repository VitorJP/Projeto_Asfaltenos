# Importação de bibliotecas do python
import numpy as np
import scipy as scp
from scipy.constants import R  # m3*Pa/mol*K


def obter_ponto_de_onset(xs_precipitante, yield_curve):
    onset = 0
    for i_onset in range(len(yield_curve)):
        if yield_curve[i_onset] > 0.005:
            onset = xs_precipitante[i_onset] if i_onset == 0 else xs_precipitante[i_onset - 1]
            break
    return onset


def calcular_yields_tempo_infinito(xs_precipitante, params):
    return params.yield_max / (1 + params.kc1 * np.exp(-(xs_precipitante - params.onset)/params.kc2))


def calcular_yields_temporais(tempos, xs_precipitante, yields_tempo_infinito, params):
    tempos = np.asarray(tempos)[:, None]
    taus = params.kt1 * (xs_precipitante ** params.kt2)
    return yields_tempo_infinito * (1 - np.exp(- tempos / taus)), taus


def criar_curva_de_equilíbrio(xs_prec_exp, x_prec_onset, yield_max, ks_x, x_max=95):
    xs_prec_eq = np.linspace(0, x_max/100, x_max+1)
    xs_prec_eq = np.unique(np.sort(np.concatenate[xs_prec_eq, xs_prec_exp]))
    yields_eq = calcular_yield_tempo_infinito(xs_prec_eq, x_prec_onset, yield_max, ks_x)

    return xs_prec_eq, yields_eq


def fator_de_correção_de_Saidoun(rho_liq, rho_asf, Ra=9.5, Rp=1.3, Df=2.45):
    phi_asf = (Ra / Rp) ** (Df - 3)
    rho_eff = 1 / ((phi_asf / rho_asf) + ((1 - phi_asf) / rho_liq))
    x_asf = phi_asf * (rho_asf / rho_eff)
    fator_correção = x_asf + (1 - x_asf) * (rho_liq / rho_asf)

    return fator_correção


def deposição_cumulativa_asfaltenos():
    pass


# ******************************************************************************************************************** #
#  ATENÇÃO: O CÓDIGO A SEGUIR SERÁ EXECUTADO APENAS QUANDO ESTE MÓDULO FOR RODADO COMO SCRIPT PRINCIPAL.               #
#           O CÓDIGO A SEGUIR SERVE PARA CONFERIR SE AS FUNÇÕES DESTE MÓDULO FUNCIONAM CORRETAMENTE.                   #
# ******************************************************************************************************************** #
# INÍCIO DO TESTE
if __name__ == "__main__":
    from módulo_exibição_de_resultados import plotar_yield_curves_cinéticas

    par_cin = [683.75, 0.030, 0.2801, -8.132]
    lista_t = [2, 4, 6, 8, 16, 24]
    max_yield_eq, onset_x_value = 0.063419, 0.50
    lista_x_solv = np.linspace(0.05, 0.95)

    yields_t_infinito = calcular_yield_tempo_infinito(lista_x_solv, onset_x_value, max_yield_eq, par_cin_t_inf)
    yields_t_calc = calcular_yields_temporais(lista_t, lista_x_solv, onset_x_value, yields_t_infinito,
                                              par_cin[0], par_cin[1], par_cin[2], par_cin[3])

    yields_experimentais = np.zeros(len(lista_x_solv))
    yields_t_exp = np.zeros((len(lista_t), len(lista_x_solv)))

    informações_auxiliares = ['predicao', 'TESTE']
    plotar_yield_curves_cinéticas(lista_x_solv, lista_t, yields_experimentais, yields_t_exp,
                                  yields_t_calc, informações_auxiliares)

# FIM DO TESTE
# ******************************************************************************************************************** #
