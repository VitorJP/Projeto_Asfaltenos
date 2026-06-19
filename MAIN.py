# ======================================================================================================================
# PARTE 0 - IMPORTAÇÕES DE BIBLIOTECAS DO PYTHON E DE OUTROS MÓDULOS DESTE PROJETO

# 0.1 - Bibliotecas
import os
import numpy as np
import pandas as pd
import scipy as scp
from tabulate import tabulate

# 0.2 - Módulos
from módulo_inputs import input_database, input_tipo_cálculos, \
    input_correlações_densidades_e_parâmetros_de_solubilidade, input_variáveis_regressão, input_controle_regressão
from módulo_leitura_dados import ler_variáveis_entrada_código, ler_dados_experimentais, ler_dados_cinéticos
from módulo_composições import normalizar_composição, fracionar_composição_SARA, \
    converter_fração_molar_para_fração_volumétrica
from módulo_propriedades_precipitante import calcular_propriedades_precipitante
from módulo_propriedades_frações_SAR import calcular_propriedades_saturados, calcular_propriedades_aromáticos, \
    calcular_propriedades_resinas
from módulo_distribuição_massa_molar import gerar_distribuição_massa_molar
from módulo_propriedades_agregados import calcular_propriedades_agregados
from módulo_equilíbrio_líquido_líquido import calcular_ELL, calcular_yield_asfaltenos
from módulo_equilíbrio_multifásico import calcular_equilíbrio_multifásico
from módulo_análise_cinética import calcular_yields_tempo_infinito, calcular_yields_temporais, \
    obter_parâmetros_experimentais_da_yield_curve, criar_curva_de_equilíbrio
from módulo_gráficos import plotar_yield_curves, plotar_distribuição_massa_molar, plotar_yield_curves_cinéticas


# ======================================================================================================================
# PARTE 1 - LEITURA DE INFORMAÇÕES BÁSICAS

# 1.1 - Dados de entrada do código
diretório_deste_módulo = os.path.dirname(__file__)
diretório_do_txt = os.path.join(diretório_deste_módulo, 'Dados de Entrada', 'variáveis_entrada_código.txt')

nome_planilha = input_database()

(regressão_equilíbrio, cálculo_cinética, algoritmo_otimização, x_yield_curve, tipo_cálculo_MM_agregados,
 método_integração_FDP_Gamma) = input_tipo_cálculos()

(correlação_delta_precipitante, correlação_densidade_saturados, correlação_delta_saturados,
 correlação_densidade_aromáticos, correlação_delta_aromáticos, correlação_densidade_resinas,
 correlação_delta_resinas, correlação_densidade_agregados,
 correlação_delta_agregados) = input_correlações_densidades_e_parâmetros_de_solubilidade()

dict_variáveis_equilibrio, dict_variáveis_cinética = input_variáveis_regressão()
variáveis_regressão_equilíbrio, variáveis_regressão_cinética = input_controle_regressão(
    regressão_equilíbrio, correlação_delta_agregados, cálculo_cinética)

tipo_cálculo_cinética = 'regressao1'

# (n_agregados, MWmin, MWmax, alfa, MWavg, tipo_cálculo_MM_agregados, método_integração_FDP_Gamma,
#  correlação_delta_precipitante, correlação_densidade_saturados, correlação_delta_saturados,
#  correlação_densidade_aromáticos, correlação_delta_aromáticos,
#  correlação_densidade_resinas, correlação_delta_resinas,
#  correlação_densidade_agregados, correlação_delta_agregados,
#  Alinha_delta_agregados, c_delta_agregados, d_delta_agregados,
#  kt1_cinético, kt2_cinético, kw1_cinético, kw2_cinético,
#  tipo_cálculo_equilíbrio, tipo_cálculo_cinética, x_yield_curve,
#  tipo_regressão_equilíbrio, algoritmo_otimização, nome_planilha) = ler_variáveis_entrada_código(diretório_do_txt)

# 1.2 - Informações experimentais do sistema
diretório_do_xlsx = os.path.join(diretório_deste_módulo, 'Dados de Entrada', 'dados_experimentais_codigo_completo.xlsx')
SARA, T, precipitante, ws_simplificados, yields_exp = ler_dados_experimentais(diretório_do_xlsx, nome_planilha)
SARA = normalizar_composição(SARA)  # normalização da composição SARA

# ======================================================================================================================
# PARTE 2 - PROPRIEDADES DO SOLVENTE, SATURADOS, AROMÁTICOS E RESINAS

# 2.1 - Inicialização dos arrays de massas molares, densidades, parâmetros de solubilidade e volumes molares
# de todos os componentes do sistema
# Obs: Estrutura do array: [Solvente, S, A, R, Asf0, Asf1, Asf2, ...]

n_agregados = dict_variáveis_equilibrio["n_agregados"]
MMs, rhos, deltas, Vs = [np.zeros(4 + n_agregados) for _ in range(4)]

# 2.2 - Propriedades do precipitante/alcano
MMs[0], rhos[0], deltas[0], Vs[0] = calcular_propriedades_precipitante(T, precipitante, correlação_delta_precipitante)

# 2.3 - Propriedades dos saturados, aromáticos e resinas
MMs[1], rhos[1], deltas[1], Vs[1] = calcular_propriedades_saturados(
    T, correlação_densidade_saturados, correlação_delta_saturados)
MMs[2], rhos[2], deltas[2], Vs[2] = calcular_propriedades_aromáticos(
    T, correlação_densidade_aromáticos, correlação_delta_aromáticos)
MMs[3], rhos[3], deltas[3], Vs[3] = calcular_propriedades_resinas(
    T, correlação_densidade_resinas, correlação_delta_resinas)

# ======================================================================================================================
# PARTE 3 - REGRESSÃO DOS PARÂMETROS TERMODINÂMICOS
# Este bloco será pulado caso tipo_cálculo_programa == 'predicao'

if regressão_equilíbrio:

    # 3.0 - Criação de uma função auxiliar para a Função Objetiva
    def atualizar_parâmetros(base, parâmetros_livres, valores_otimização):
        parâmetros = base.copy()

        for nome, valor in zip(parâmetros_livres, valores_otimização):
            parâmetros[nome] = valor

        return parâmetros

    # 3.1 - Criação da Função Objetivo
    def F_obj(valores_otimização, *args):

        # Desempacotando os *args principais para a regressão dos parâmetros
        parâmetros_completo = args[0]
        parâmetros_regressão = args[1]
        params = atualizar_parâmetros(parâmetros_completo, parâmetros_regressão, valores_otimização)

        # Alocando os valores dos parâmetros possíveis de regressão
        n_agregados = params["n_agregados"]
        MWmin = params["MW_min"]
        MWmax = params["MW_max"]
        MWavg = params["MW_avg"]
        alfa = params["alfa"]
        Alinha_delta_agregados = params["A'_Barrera"]
        c_delta_agregados = params["c_Barrera"]
        d_delta_agregados = params["d_Barrera"]

        # 3.1.2 - Desempacotando os *args (outros argumentos da função 'F_obj' a serem passados pra função 'minimize')
        T, SARA, ws_simplificados, yields_exp = args[2]
        MMs, rhos, deltas, Vs = args[3]
        tipo_cálculo_MM_agregados, método_integração_FDP_Gamma = args[4]

        # 3.1.3 - Propriedades dos agregados de asfaltenos
        MMsagregados, wsagregados, xsagregados = gerar_distribuição_massa_molar(
            alfa, MWavg, n_agregados, MWmin, MWmax, tipo_cálculo_MM_agregados, método_integração_FDP_Gamma)
        wsagregados, xsagregados = normalizar_composição(wsagregados), normalizar_composição(xsagregados)
        rhosagregados, deltasagregados, Vsagregados = calcular_propriedades_agregados(
            T, MMsagregados, correlação_densidade_agregados, correlação_delta_agregados,
            Alinha_delta_agregados, c_delta_agregados, d_delta_agregados)
        
        # 3.1.4 - Alocação das propriedades dos agregados de asfaltenos
        # nos arrays que contêm as propriedades de todos os componentes do sistema
        MMs[4:4 + n_agregados] = MMsagregados[0:n_agregados]*1e-3  # kg/mol
        rhos[4:4 + n_agregados] = rhosagregados[0:n_agregados]
        deltas[4:4 + n_agregados] = deltasagregados[0:n_agregados]
        Vs[4:4 + n_agregados] = Vsagregados[0:n_agregados]
        
        # 3.1.5 - Composição global do sistema em termos de [Solvente, S, A, R, Asf0, Asf1, ...] (base mássica e molar)
        _, xs_completo = fracionar_composição_SARA(ws_simplificados, SARA, wsagregados, MMs)
        xs_completo = np.apply_along_axis(func1d=normalizar_composição, axis=1, arr=xs_completo)

        # 3.1.6 - Cálculo de equilíbrio líquido-líquido
        n_dados_exp = yields_exp.shape[0]
        yields_calc = np.zeros(n_dados_exp)
        for i_comp in range(n_dados_exp):
            betarr, xs_leve, xs_pesada, _ = calcular_ELL(T, xs_completo[i_comp], deltas, Vs, xsagregados, MMs)
            yields_calc[i_comp] = calcular_yield_asfaltenos(betarr, xs_leve, xs_pesada, MMs)

        # 3.1.7 - Expressão matemática a ser minimizada
        yields_diferenças = np.abs(yields_calc - yields_exp)  # diferenças entre os yields calculados e experimentais

        return np.nanmean(yields_diferenças)

    # 3.2 - Minimização da Função Objetivo para Regressão dos Parâmetros
    # 3.2.1 - Chutes iniciais dos parâmetros a serem estimados
    chute_inicial = np.array([dict_variáveis_equilibrio[nome] for nome in variáveis_regressão_equilíbrio])

    # 3.2.2 - Atribuição de valores para os *args (outros argumentos da função 'F_obj' para a função 'minimize')
    dados_experimentais = (T, SARA, ws_simplificados, yields_exp) 
    propriedades_componentes = (MMs, rhos, deltas, Vs)
    definições_distribuição_massa_molar = (tipo_cálculo_MM_agregados, método_integração_FDP_Gamma)

    argumentos_otimização = (dict_variáveis_equilibrio, variáveis_regressão_equilíbrio, dados_experimentais,
                             propriedades_componentes, definições_distribuição_massa_molar)

    # 3.2.3 - Declaração dos intervalos limites para as variáveis
    limites = {
        "n_agregados": (1, 30),
        "MW_min": (600, 1800),
        "MW_max": (5000, 30000),
        "MW_avg": (1800, 10000),
        "alfa": (1.15, 80),
        "c_Barrera": (0.634, 0.672),
        "A'_Barrera": (0.0, 0.3),
        "d_Barrera": (0.0445, 0.0545),
    }
    limites_parametros = [limites[nome] for nome in variáveis_regressão_equilíbrio]

    # 3.3 - Otimização da Função Objetivo
    # 3.3.1 - Configuração do algoritmo de otimização
    if algoritmo_otimização == 1:  # 3.3.1.1 - Nelder-Mead
        sol = scp.optimize.minimize(F_obj, chute_inicial, method="Nelder-Mead",
                                    args=argumentos_otimização)
    elif algoritmo_otimização == 2:  # 3.3.1.2 - Brute-force
        sol = 0
        pass  # Obs: ainda falta ser implementado
    elif algoritmo_otimização == 3:  # 3.3.1.3 - L-BFGS-B
        sol = scp.optimize.minimize(F_obj, chute_inicial, method="L-BFGS-B", bounds=limites_parâmetros,
                                    args=argumentos_otimização)
    elif algoritmo_otimização == 4:  # 3.3.1.4 - Powell
        sol = scp.optimize.minimize(F_obj, chute_inicial, method="Powell", bounds=limites_parâmetros,
                                    args=argumentos_otimização)
    else:  # Caso Erro
        sol = 0
        raise ValueError(f"Problema na escolha da variável algoritmo_otimização. Ela deve ser um número inteiro entre "
                         f"1 e 4. Valor inserido: {algoritmo_otimização}")

    # 3.3.2 - Alocação dos parâmetros estimados
    dict_variáveis_equilibrio = atualizar_parâmetros(dict_variáveis_equilibrio, variáveis_regressão_equilíbrio, sol.x)

# ======================================================================================================================
# PARTE 4 - PREDIÇÃO DA CURVA DE SOLUBILIDADE

# 4.1 - Propriedades dos agregados de asfaltenos
# 4.1.1 - Massas molares, frações mássicas e frações molares
MMsagregados, wsagregados, xsagregados = gerar_distribuição_massa_molar(
    dict_variáveis_equilibrio["alfa"], dict_variáveis_equilibrio["MW_avg"], dict_variáveis_equilibrio["n_agregados"],
    dict_variáveis_equilibrio["MW_min"], dict_variáveis_equilibrio["MW_max"],
    tipo_cálculo_MM_agregados, método_integração_FDP_Gamma)
wsagregados, xsagregados = normalizar_composição(wsagregados), normalizar_composição(xsagregados)

# 4.1.2 - Densidades, parâmetros de solubilidades e volumes molares
rhosagregados, deltasagregados, Vsagregados = calcular_propriedades_agregados(
    T, MMsagregados, correlação_densidade_agregados, correlação_delta_agregados,
    dict_variáveis_equilibrio["A'_Barrera"], dict_variáveis_equilibrio["c_Barrera"],
    dict_variáveis_equilibrio["d_Barrera"])

# 4.1.3 - Alocação das propriedades dos agregados de asfaltenos nos arrays das propriedades dos componentes do sistema
n_agregados = dict_variáveis_equilibrio["n_agregados"]
MMs[4:4 + n_agregados] = MMsagregados[0:n_agregados]*1e-3  # kg/mol
rhos[4:4 + n_agregados] = rhosagregados[0:n_agregados]
deltas[4:4 + n_agregados] = deltasagregados[0:n_agregados]
Vs[4:4 + n_agregados] = Vsagregados[0:n_agregados]

# 4.2 - COMPOSIÇÃO GLOBAL DO SISTEMA EM TERMOS DE [Solvente, S, A, R, Asf0, Asf1, ...] (base mássica e base molar)
ws_completo, xs_completo = fracionar_composição_SARA(ws_simplificados, SARA, wsagregados, MMs)
ws_completo = np.apply_along_axis(func1d=normalizar_composição, axis=1, arr=ws_completo)
# normalização das frações mássicas de cada linha
xs_completo = np.apply_along_axis(func1d=normalizar_composição, axis=1, arr=xs_completo)
# normalização das frações molares de cada linha

# 4.3 - CÁLCULO DE EQUILÍBRIO LÍQUIDO-LÍQUIDO
# 4.3.1 - Nº de dados experimentais
n_dados_exp = yields_exp.shape[0]

# 4.3.2 - Inicialização dos arrays que armazenarão os resultados dos cálculos de ELL
betas = np.zeros(n_dados_exp)  # betas de Rachford-Rice
xsL, xsH = [np.zeros((n_dados_exp, 4 + n_agregados)) for _ in range(2)]
# composição da fase leve, composição da fase pesada
somaxsL, somaxsH = [np.zeros(n_dados_exp) for _ in range(2)]
# soma da composição da fase leve, soma da composição da fase pesada
n_it = np.zeros(n_dados_exp)  # nº de iterações

# 4.3.3 - Inicialização do array que armazenará os yields calculados
yields_calc = np.zeros(n_dados_exp)

# 4.3.4 - Cálculos das composições de ELL e yields de asfaltenos p/ cada i-ésimo dado experimental
for i in range(n_dados_exp):
    betas[i], xsL[i, :], xsH[i, :], n_it[i] = calcular_ELL(T, xs_completo[i], deltas, Vs, xsagregados, MMs)
    yields_calc[i] = calcular_yield_asfaltenos(betas[i], xsL[i, :], xsH[i, :], MMs)
    somaxsL[i], somaxsH[i] = np.round(xsL[i, :].sum(), decimals=8), np.round(xsH[i, :].sum(), decimals=8)

    if np.abs(somaxsL[i] - 1.0) > 1e6 and np.abs(somaxsH[i] - 1.0) > 1e6:
        raise ValueError("Erro nas composições das fases leve e pesada.")
    elif np.abs(somaxsL[i] - 1.0) > 1e6:
        raise ValueError("Erro na composição da fase leve.")
    elif np.abs(somaxsH[i] - 1.0) > 1e6:
        raise ValueError("Erro na composição da fase pesada.")

# ======================================================================================================================
# PARTE 6 - REGRESSÃO DOS DADOS CINÉTICOS

if cálculo_cinética != 'não':

    # 6.1 - Parâmetros do Modelo Cinético
    # 6.1.3 - Dados Experimentais Temporais
    w_precipitante = ws_completo[:, 0]
    tempos, yields_temp_exp = ler_dados_cinéticos(diretório_do_xlsx, nome_planilha)

    # 6.1.1 - Parâmetros experimentais da yield curve
    w_onset, yield_max = obter_parâmetros_experimentais_da_yield_curve(w_precipitante, yields_calc)


    def atualizar_parâmetros(base, parâmetros_livres, valores_otimização):
        parâmetros = base.copy()

        for nome, valor in zip(parâmetros_livres, valores_otimização):
            parâmetros[nome] = valor

        return parâmetros

    if cálculo_cinética == 'regressão':

        # 6.3 - Otimização dos Parâmetros Cinéticos de Tempo Infinito
        def F_obj(valores_otimização, *args):

            # Desempacotando os *args principais para a regressão dos parâmetros
            parâmetros_completo = args[0]
            parâmetros_regressão = args[1]
            params = atualizar_parâmetros(parâmetros_completo, parâmetros_regressão, valores_otimização)

            # Alocando os valores dos parâmetros possíveis de regressão
            kw1, kw2, kt1, kt2 = params["kw1"], params["kw2"], params["kt1"], params["kt2"]

            # Desempacotando os *args (outros argumentos da função 'F_obj' a serem passados pra função 'minimize')
            w_onset, yield_max = args[2]
            w_precipitante, yields_eq_exp, tempos, yields_t_exp = args[3]

            yields_eq = calcular_yields_tempo_infinito(w_precipitante, w_onset, yield_max, kw1, kw2)
            yields_t, _ = calcular_yields_temporais(tempos, w_precipitante, w_onset, yield_max, kw1, kw2, kt1, kt2)

            erro_eq = np.nanmean(np.abs(yields_eq - yields_eq_exp))
            erro_t = np.nanmean(np.abs(yields_t - yields_t_exp))

            return erro_eq + erro_t

        # Minimização da Função Objetivo para Regressão dos Parâmetros
        # Chutes iniciais dos parâmetros a serem estimados
        chute_inicial = np.array([dict_variáveis_cinética[nome] for nome in variáveis_regressão_cinética])

        # Atribuição de valores para os *args (outros argumentos da função 'F_obj' para a função 'minimize')
        parâmetros_experimentais_yield_curve = (w_onset, yield_max)
        dados_experimentais = (w_precipitante, yields_calc, tempos, yields_temp_exp)

        argumentos_otimização = (dict_variáveis_cinética, variáveis_regressão_cinética,
                                 parâmetros_experimentais_yield_curve, dados_experimentais)

        # Otimização dos Parâmetros
        sol = scp.optimize.minimize(F_obj, chute_inicial, method="Nelder-Mead",
                                    args=argumentos_otimização)

        # Alocação dos parâmetros estimados
        dict_variáveis_cinética = atualizar_parâmetros(dict_variáveis_cinética, variáveis_regressão_cinética, sol.x)

    # 6.5 - Predição do Modelo Cinético
    yields_eq = calcular_yields_tempo_infinito(w_precipitante, w_onset, yield_max,
                                               dict_variáveis_cinética["kw1"], dict_variáveis_cinética["kw2"])
    yields_temp_calc, taus = calcular_yields_temporais(tempos, w_precipitante, w_onset, yield_max,
                                                       dict_variáveis_cinética["kw1"], dict_variáveis_cinética["kw2"],
                                                       dict_variáveis_cinética["kt1"], dict_variáveis_cinética["kt2"])

    print("w_onset: ", w_onset)
    print("yield_max: ", yield_max)
    print("taus:", taus)
    print("PARÂMETROS ESTIMADOS: ", dict_variáveis_cinética)

    # 6.6 - Criação do Gráfico de Curvas Cinéticas em diferentes tempos
    informações_auxiliares = [cálculo_cinética, nome_planilha]
    plotar_yield_curves_cinéticas(w_precipitante, tempos, yields_eq, yields_temp_exp, yields_temp_calc,
                                  informações_auxiliares)

# ==================================================================================================================== #
# PARTE 7 - EXIBIÇÃO DOS RESULTADOS

# 7.0 - Escolha do referencial das yield curves para geração dos gráficos
match x_yield_curve:
    case "molar":
        eixo_x = xs_completo[:, 0]
        x_label = "Fração Molar precipitante"
    case "massa":
        eixo_x = ws_completo[:, 0]
        x_label = "Fração Mássica precipitante"
    case "volume":
        phis_completo = converter_fração_molar_para_fração_volumétrica(xs_completo, Vs)
        eixo_x = phis_completo[:, 0]
        x_label = "Fração Volumétrica precipitante"
    case "solubilidade":
        phis_L = converter_fração_molar_para_fração_volumétrica(xsL, Vs)
        deltas_L = (phis_L * deltas[None, :]).sum(axis=1)  # Pa**0.5
        eixo_x = deltas_L * 1e-3  # MPa**0.5
        x_label = "Parâmetro de Solubilidade"
    case _:  # Em caso de erro, utilizar fração mássica como padrão.
        eixo_x = ws_completo[:, 0]
        x_label = "Fração Mássica precipitante"

# 7.1 - Criação dos dados de desvios para o sistema
# 7.1.1 - Se há dados experimentais de yields para o sistema em questão
no_experimental_data = True if all(yield_exp == 0 for yield_exp in yields_exp) else False

# 7.1.2 - Criação de listas com os resultados formatados
DAs = None if no_experimental_data else np.abs(yields_exp - yields_calc)  # desvios absolutos fracionais
DMA = None if no_experimental_data else np.nanmean(DAs)  # média dos desvios absolutos fracionais

DAs_formatado = ["nao disponivel" for yield_calc in yields_calc] if no_experimental_data \
    else [f"{100*DA:.2f}%" for DA in DAs]
DMA_formatado = "nao disponivel" if no_experimental_data else f"{100*DMA:.4f}%"

eixo_x_formatado = [f"{100*valor_x:.2f}%" for valor_x in eixo_x] if x_yield_curve != 'solubilidade' else \
    [f"{valor_x:.2f} MPa^0.5" for valor_x in eixo_x]
yields_exp_formatado = ["nao disponivel" for yield_calc in yields_calc] if no_experimental_data \
    else [f"{100*yield_exp:.2f}%" for yield_exp in yields_exp]
yields_calc_formatado = [f"{100*yield_calc:.2f}%" for yield_calc in yields_calc]
betas_formatado = [f"{beta:.4e}" for beta in betas]

# 7.2 - Criação e impressão de Dataframe com os resultados
df_resultados = pd.DataFrame(
    {f"  {x_label}  ": eixo_x_formatado,
     "  Yield (exp.)  ": yields_exp_formatado,
     "  Yield (calc.)  ": yields_calc_formatado,
     "  DA (%)  ": DAs_formatado,
     "  Beta  ": betas_formatado,
     # "  somaxsL  ": somaxsL,
     # "  somaxsH  ": somaxsH,
     # "  qte. iteracoes  ": list(map(int, n_it))
     })
print(f"\n| DESVIO MEDIO ABSOLUTO NOS YIELDS (%): {DMA_formatado}")
if regressão_equilíbrio:
    print(f"PARÂMETROS ESTIMADOS: {variáveis_regressão_equilíbrio}")
    print(f"PARÂMETROS ESTIMADOS: {sol.x}")
    print(f"{tabulate(df_resultados, headers = df_resultados.columns, tablefmt = 'pretty', showindex = False)}")

# 7.3 - Criação dos gráficos: yield curves e distribuição de massa molar
informações_auxiliares = [DMA_formatado, regressão_equilíbrio, variáveis_regressão_equilíbrio,
                          algoritmo_otimização, x_yield_curve, nome_planilha]
plotar_yield_curves(eixo_x, yields_exp, yields_calc, SARA[-1], informações_auxiliares)
plotar_distribuição_massa_molar(MMsagregados, xsagregados, dict_variáveis_equilibrio["alfa"],
                                dict_variáveis_equilibrio["MW_avg"], informações_auxiliares)

# ==================================================================================================================== #
