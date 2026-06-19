# Importação de bibliotecas do python
import numpy as np


def input_database():
    nome_planilha = 'Rolim_A_hept'

    return nome_planilha


def input_tipo_cálculos():
    """Decisões de quais métodos de cálculo serão aplicados e quais funções serão regredidas."""

    regressão_equilíbrio = True

    cálculo_cinética = 'regressão'
    # opções:   (não)
    #           (predição)
    #           (regressão)

    algoritmo_otimização = 1
    # opções:   (1) Nelder-Mead
    #           (2) Brute-force com resultado servindo de chute inicial para Nelder-Mead
    #           (3) L-BFGS-B
    #           (4) Powell

    if algoritmo_otimização == 2:
        raise ValueError('O algoritmo de Brute-force com resultado servindo de chute inicial para Nelder-Mead ainda'
                         'não está implementado.')

    x_yield_curve = 'massa'
    # opções:   (massa)
    #           (molar)
    #           (volume)
    #           (solubilidade)

    tipo_cálculo_MM_agregados = 'superior'
    # opções:   (superior)
    #           (médio)

    método_integração_FDP_Gamma = 'trapezios'
    # opções:   (quadratura)
    #           (trapezios)

    return (regressão_equilíbrio, cálculo_cinética, algoritmo_otimização, x_yield_curve,
            tipo_cálculo_MM_agregados, método_integração_FDP_Gamma)


def input_correlações_densidades_e_parâmetros_de_solubilidade():
    """Escolha de quais correlações aplicar para as densidades (rho) e parâmetros de solubilidade (delta)
    do precipitante, e frações SARA (Saturados, Aromáticos, Resinas e Asfaltenos)."""

    correlação_delta_precipitante = 'Akbarzadeh'
    # opções:   (Akbarzadeh)
    #           (Vargas)

    correlação_rho_saturados = 'Akbarzadeh'
    # opções:   (Alves)
    #           (Akbarzadeh)
    #           (Yanes)

    correlação_delta_saturados = 'Akbarzadeh'
    # opções:   (Akbarzadeh)
    #           (Tharanivasan)
    #           (Yanes)

    correlação_rho_aromáticos = 'Akbarzadeh'
    # opções:   (Alves)
    #           (Akbarzadeh)
    #           (Yanes)

    correlação_delta_aromáticos = 'Akbarzadeh'
    # opções:   (Akbarzadeh)
    #           (Yanes)

    correlação_rho_resinas = 'Yanes'
    # opções:   (Yanes)

    correlação_delta_resinas = 'Yanes'
    # opções:   (Yanes)

    correlação_rho_asfaltenos = 'Alboudwarej'
    # opções:   (Alboudwarej)
    #           (Barrera)

    correlação_delta_asfaltenos = 'Tharanivasan'
    # opções:   (Tharanivasan)
    #           (Barrera)

    return (correlação_delta_precipitante,
            correlação_rho_saturados, correlação_delta_precipitante,
            correlação_rho_aromáticos, correlação_delta_aromáticos,
            correlação_rho_resinas, correlação_delta_resinas,
            correlação_rho_asfaltenos, correlação_delta_asfaltenos)


def input_variáveis_regressão():
    """Dicionário de parâmetros matemáticos que são utilizados pelo código."""

    variáveis_equilíbrio = {
        # Parâmetros da Distribuição de Massa Molar de Asfaltenos
        "n_agregados": 30,
        "MW_min": 750,
        "MW_max": 30000,
        "MW_avg": 3620,
        "alfa": 3.5,

        # Parâmetros da correlação de Barrera para delta de asfaltenos
        "A'_Barrera": 0.0,
        "c_Barrera": 0.647,
        "d_Barrera": 0.0495,
    }

    vairáveis_cinética = {
        # Parâmetros cinéticos de Saidoun
        "kw1": 683.75,
        "kw2": 0.030,
        "kt1": 5.0,
        "kt2": -2.0,
    }

    return variáveis_equilíbrio, vairáveis_cinética


def input_controle_regressão(regressao_equilibrio, correlação_delta_agregados, calculo_cinetica):
    """Dicionário de boolean para definir se cada parâmetro será fixo ou regredido."""

    controle_regressão_equilíbrio = {
        # Parâmetros da Distribuição de Massa Molar de Asfaltenos
        "MW_min": False,
        "MW_max": False,
        "MW_avg": True,
        "alfa": True,

        # Parâmetros da correlação de Barrera para delta de asfaltenos
        "A'_Barrera": False,
        "c_Barrera": False,
        "d_Barrera": False,
    }

    controle_regressão_cinética = {
        # Parâmetros cinéticos de Saidoun
        "kw1": True,
        "kw2": True,
        "kt1": True,
        "kt2": True,
    }

    # Validação do controle dos parâmetros com base no tipo de cálculo
    if not regressao_equilibrio:
        if controle_regressão_equilíbrio["MW_min"] or controle_regressão["MW_max"] or controle_regressão["MW_avg"] \
                or controle_regressão["alfa"]:
            mensagem = 'ATENÇÃO: Corrija o input!'
            mensagem += '\nPara regredir os parâmetros da distribuição da massa molar de asfaltenos, ' \
                        'é preciso que a regressão_equilíbrio seja TRUE como input de tipo de cálculo.'
            raise ValueError(mensagem)

    if calculo_cinetica != 'regressão':
        if controle_regressão_cinética["kw1"] or controle_regressão_cinética["kw2"] \
                or controle_regressão_cinética["kt1"] or controle_regressão_cinética["kt2"]:
            mensagem = 'ATENÇÃO: Corrija o input!'
            mensagem += '\nPara regredir os parâmetros cinéticos de Saidoun, ' \
                        'é preciso que a regressão_cinética seja TRUE como input de tipo de cálculo.'
            raise ValueError(mensagem)

    if not correlação_delta_agregados == 'Barrera':
        if controle_regressão_equilíbrio["A'_Barrera"] or controle_regressão_equilíbrio["c_Barrera"] \
                or controle_regressão_equilíbrio["d_Barrera"]:
            mensagem = 'ATENÇÃO: Corrija o input!'
            mensagem += '\nPara regredir os parâmetros da correlação de solubilidade de Barrera, ' \
                        'é preciso que a correlação de Barrera seja a selecionada para cálculo de ' \
                        'delta dos agregados de asfaltenos.'
            raise ValueError(mensagem)

    regressão_equilíbrio = [nome for nome, regredir in controle_regressão_equilíbrio.items() if regredir]
    regressão_cinética = [nome for nome, regredir in controle_regressão_cinética.items() if regredir]

    return regressão_equilíbrio, regressão_cinética

