# Importação de bibliotecas do python
import numpy as np
import pandas as pd
import scipy as scp
from copy import deepcopy

# Importação de módulos internos
from módulo_composições import FasesSistema
from módulo_equilíbrio_líquido_líquido import calcular_ELL
from módulo_cálculos_cinéticos import calcular_yields_tempo_infinito, calcular_yields_temporais
from módulo_cálculos_erros import average_absolute_relative_deviation


# Subfunção
def diferença_média(valor_calc, valor_exp):
    return np.nanmean(np.abs(valor_calc - valor_exp))


# Função
def regressão_equilíbrio(valores_otimização, *args):
    # Desempacotando os *args
    params_base = args[0]
    T, SARA, precipitante, ws_exp, yields_exp, n_dados_exp = args[1]
    propriedades = deepcopy(args[2])
    config = args[3]

    # Alocando os valores dos parâmetros possíveis de regressão
    params = deepcopy(params_base)
    params.atualizar(params.variáveis_regressão, valores_otimização)

    # Propriedades dos agregados de asfaltenos e dos componentes do sistema
    propriedades.adicionar_asfaltenos(T, params, config)

    # Composição global do sistema em termos de [Solvente, S, A, R, Asf0, Asf1, ...]
    sistema = FasesSistema.inicializar(ws_exp, SARA, n_dados_exp, propriedades)

    # Cálculo de Equilíbrio Líquido-Líquido
    for i in range(n_dados_exp):
        sistema.betas[i], sistema.fase_leve.xs[i, :], sistema.fase_pesada.xs[i, :], _ = calcular_ELL(
            T, sistema.fase_global.xs[i], propriedades)

    # Expressão matemática a ser minimizada: diferenças entre os yields calculados e experimentais
    return diferença_média(sistema.yields_calc(propriedades.MMs), yields_exp)


# Função
def regressão_cinética(valores_otimização, *args):
    # Desempacotando os *args
    params_base = args[0]
    tempos, ws_precipitante, yields_t_exp = args[1]
    yields_eq = args[2]

    params = deepcopy(params_base)
    params.atualizar(params.variáveis_regressão, valores_otimização)

    # Cálculo das equações cinéticas
    yields_eq_calc = calcular_yields_tempo_infinito(ws_precipitante, params)
    yields_t_calc, _ = calcular_yields_temporais(tempos, ws_precipitante, yields_eq_calc, params)

    # Expressão matemática a ser minimizada: diferenças entre os yields calculados e experimentais
    return diferença_média(yields_eq_calc, yields_eq) + diferença_média(yields_t_calc, yields_t_exp)
