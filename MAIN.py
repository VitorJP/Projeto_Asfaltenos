# ==================================================================================================================== #
# PARTE 0 - IMPORTAÇÕES DE BIBLIOTECAS DO PYTHON E DE OUTROS MÓDULOS DESTE PROJETO

# 0.1 - Bibliotecas
import os
import numpy as np
import pandas as pd
import scipy as scp
from tabulate import tabulate

# 0.2 - Módulos
from módulo_leitura_dados import ler_dados_experimentais, ler_dados_cinéticos, ler_lista_datasets
from módulo_configurações import Configurações
from módulo_parâmetros import Parâmetros
from módulo_propriedades import Propriedades
from módulo_composições import FasesSistema
from módulo_equilíbrio_líquido_líquido import calcular_ELL
from módulo_equilíbrio_multifásico import calcular_equilíbrio_multifásico
from módulo_cálculos_cinéticos import calcular_yields_tempo_infinito, calcular_yields_temporais, obter_ponto_de_onset
from módulo_exibição_de_resultados import exibir_resultados_equilíbrio, exibir_resultados_cinética


# ==================================================================================================================== #
def executar_modelagem_principal(nome_planilha, diretório):
    # ================================================================================================================ #
    # PARTE 1: LEITURA DE INFORMAÇÕES BÁSICAS

    # 1.1 - Declaração do dataset em análise
    print(f"\nDATASET: {nome_planilha}")

    # 1.2 - Configurações e Parâmetros do código
    # Obs: para alterar estas configurações e parâmetros, é preciso mudar nos módulos
    # 'configurações' e 'parâmetros' (ainda a ser melhorado)
    config = Configurações.inicializar()
    params = Parâmetros.inicializar()

    # 1.3 - Informações experimentais do sistema
    diretório_do_xlsx = os.path.join(diretório, 'Dados de Entrada', 'database_yield_curves.xlsx')
    SARA, T, precipitante, ws_exp, yields_exp, n_dados_exp = ler_dados_experimentais(diretório_do_xlsx, nome_planilha)

    # ================================================================================================================ #
    # PARTE 2: MODELAGEM TERMODINÂMICA
    # 2.1 - Propriedades dos Saturados, Aromáticos, Resinas e Precipitante/Alcano
    # Obs: Estrutura do array: [Precipitante, S, A, R]
    propriedades = Propriedades.inicializar(T, precipitante, config)

    # 2.2 - Regressão dos parâmetros dos asfaltenos
    if config.cálculo.tipo_cálculo_equilíbrio == 'regressão':
        dados_experimentais = (T, SARA, precipitante, ws_exp, yields_exp, n_dados_exp)
        params.equilíbrio.regredir(dados_experimentais, propriedades, config)

    # 2.3 - Predição da Curva de Solubilidade
    # 2.3.1 - Propriedades dos agregados de asfaltenos e dos componentes do sistema
    # Obs: Estrutura do array: [Precipitante, S, A, R, Asf0, Asf1, ...]
    propriedades.adicionar_asfaltenos(T, params.equilíbrio, config)

    # 2.3.2 - Composição global do sistema
    sistema = FasesSistema.inicializar(ws_exp, SARA, n_dados_exp, propriedades)

    # 2.3.3 - Cálculos das composições de ELL e yields de asfaltenos p/ cada i-ésimo dado experimental
    for i in range(n_dados_exp):
        sistema.betas[i], sistema.fase_leve.xs[i, :], sistema.fase_pesada.xs[i, :], _ = calcular_ELL(
            T, sistema.fase_global.xs[i], propriedades)
    # sistema.validação()

    # 2.4 - Exibição dos Resultados de Yield Curve
    exibir_resultados_equilíbrio(nome_planilha, config, sistema, propriedades, params.equilíbrio, yields_exp, SARA)

    # ================================================================================================================ #
    # PARTE 3: MODELAGEM CINÉTICA

    if config.cálculo.tipo_cálculo_cinética != 'não':
        # 3.1 - Parâmetros do Modelo Cinético
        # 3.1.1 - Leitura dos dados experimentais
        tempos, yields_temp_exp = ler_dados_cinéticos(diretório_do_xlsx, nome_planilha)  # dados temporais

        # 3.1.2 - Parâmetros experimentais da yield curve
        params.cinética.yield_max = max(sistema.yields_calc(propriedades.MMs))
        params.cinética.onset = obter_ponto_de_onset(ws_exp[:, 0], sistema.yields_calc(propriedades.MMs))

        # 3.2 - Regressão dos parâmetros cinéticos
        if config.cálculo.tipo_cálculo_cinética == 'regressão':
            dados_experimentais = (tempos, ws_exp[:, 0], yields_temp_exp)
            params.cinética.regredir(dados_experimentais, sistema.yields_calc(propriedades.MMs), config)

        # 3.3 - Predição do Modelo Cinético
        yields_eq_calc = calcular_yields_tempo_infinito(ws_exp[:, 0], params.cinética)
        yields_temp_calc, taus = calcular_yields_temporais(tempos, ws_exp[:, 0], yields_eq_calc, params.cinética)

        # 3.4 - Exibição dos Resultados de Yield Curves Cinéticas
        exibir_resultados_cinética(config, nome_planilha, tempos, ws_exp[:, 0], taus, params.cinética,
                                   yields_eq_calc, yields_temp_exp, yields_temp_calc)
    # ================================================================================================================ #


# CÓDIGO INICIAL
diretório_deste_módulo = os.path.dirname(__file__)
database = ler_lista_datasets(diretório_deste_módulo)

executar_tudo = False
if executar_tudo:
    for db in database:
        executar_código_principal(db, diretório_deste_módulo)
else:
    dataset = 'Powers_VB-TC51'
    if dataset not in database:
        raise ValueError("Nome da planilha inválida")
    executar_modelagem_principal(dataset, diretório_deste_módulo)
