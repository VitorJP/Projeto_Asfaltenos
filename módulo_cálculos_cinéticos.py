# Importação de bibliotecas do python
import numpy as np
import scipy as scp
from scipy.constants import R  # m3*Pa/mol*K
from dataclasses import dataclass

@dataclass
class parâmetros_precipitação:

    onset: float
    yield_max: float
    kc1: float
    kc2: float
    kt1: float
    kt2: float

@dataclass
class modelo_precipitação:
    parâmetros: parâmetros_precipitação

    def obter_ponto_de_onset(self, xs_precipitante, yield_curve, tol=0.001):
        onset = 0
        for i_onset in range(len(yield_curve)):
            if yield_curve[i_onset] > tol:
                onset = xs_precipitante[i_onset] if i_onset == 0 else xs_precipitante[i_onset - 1]
                break
        return onset
    
    def calcular_yields_tempo_infinito(self, xs_precipitante):
            return self.parâmetros.yield_max / (1 + self.parâmetros.kc1 * np.exp(-(xs_precipitante - self.parâmetros.onset)/self.parâmetros.kc2))

    def calcular_yields_temporais(self, tempos, xs_precipitante, yields_tempo_infinito):
        tempos = np.asarray(tempos)[:, None]
        taus = self.parâmetros.kt1 * (xs_precipitante ** self.parâmetros.kt2)
        return yields_tempo_infinito * (1 - np.exp(- tempos / taus)), taus

    def criar_curva_de_equilíbrio(self, xs_prec_exp, x_max=95):
        xs_prec_eq = np.linspace(0, x_max/100, x_max+1)
        xs_prec_eq = np.unique(np.sort(np.concatenate((xs_prec_eq, xs_prec_exp))))
        yields_eq = self.calcular_yields_tempo_infinito(xs_prec_eq)

        return xs_prec_eq, yields_eq

@dataclass
class cálculos_físicos:

    @staticmethod
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

    params = parâmetros_precipitação(
            onset=0.50,                     
            yield_max=0.063419,
            kc1=683.75,
            kc2=0.030,
            kt1=0.2801,
            kt2=-8.132
            )
    
    modelo = modelo_precipitação(parâmetros=params)

    from módulo_resultados import plotar_yield_curves_cinéticas

    lista_t = [2, 4, 6, 8, 16, 24]
    lista_x_solv = np.linspace(0.05, 0.95)

    yields_t_infinito = modelo.calcular_yields_tempo_infinito(lista_x_solv)
    yields_t_calc, taus = modelo.calcular_yields_temporais(lista_t, lista_x_solv, yields_t_infinito)

    yields_experimentais = np.zeros(len(lista_x_solv))
    yields_t_exp = np.zeros((len(lista_t), len(lista_x_solv)))

    informações_auxiliares = ['predicao', 'TESTE', 'Resultados_Cinética']
    plotar_yield_curves_cinéticas(lista_x_solv, lista_t, yields_experimentais, yields_t_exp,
                                  yields_t_calc, informações_auxiliares)

# FIM DO TESTE
# ******************************************************************************************************************** #
