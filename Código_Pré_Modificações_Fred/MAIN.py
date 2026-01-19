# ======================================================================================================================
# PARTE 0 - IMPORTAÇÕES DE BIBLIOTECAS DO PYTHON E DE OUTROS MÓDULOS DESTE PROJETO

# 0.1 - Bibliotecas
import os
import numpy as np
import pandas as pd
import scipy as scp
from tabulate import tabulate

# 0.2 - Módulos 
from módulo_leitura_dados import ler_variáveis_entrada_código, ler_dados_experimentais
from módulo_composições import normalizar_composição, fracionar_composição_global
from módulo_propriedades_solvente import calcular_propriedades_solvente
from módulo_propriedades_frações_SAR import calcular_propriedades_saturados, calcular_propriedades_aromáticos, \
    calcular_propriedades_resinas
from módulo_distribuição_massa_molar import gerar_distribuição_massa_molar
from módulo_propriedades_agregados import calcular_propriedades_agregados
from módulo_equilíbrio_líquido_líquido import calcular_composições_ELL, calcular_yield_asfaltenos
from módulo_análise_cinética import calcular_yield_tempo_infinito, calcular_yields_temporais
from módulo_gráficos import plotar_yield_curves, plotar_distribuição_massa_molar, plotar_yield_curves_cinéticas


# ======================================================================================================================
# PARTE 1 - LEITURA DE INFORMAÇÕES BÁSICAS

# 1.1 - Dados de entrada do código
diretório_deste_módulo = os.path.dirname(__file__)
diretório_do_txt = os.path.join(diretório_deste_módulo, 'variáveis_entrada_código.txt')
(n_agregados, MWmin, MWmax, alfa, MWavg, tipo_cálculo_MM_agregados, método_integração_FDP_Gamma, 
 correlação_densidade_saturados, correlação_delta_saturados,
 correlação_densidade_aromáticos, correlação_delta_aromáticos,
 correlação_densidade_resinas, correlação_delta_resinas,
 correlação_densidade_agregados, correlação_delta_agregados, 
 Alinha_delta_agregados, c_delta_agregados, d_delta_agregados,
 tipo_cálculo_equilíbrio, tipo_cálculo_cinética, tipo_regressão_equilíbrio,
 algoritmo_otimização,
 nome_planilha) = ler_variáveis_entrada_código(diretório_do_txt)

# 1.2 - Validação dos valores das variáveis 'correlação_delta_agregados' e 'tipo_regressão'
# Obs: só faz sentido que 'tipo_regressão' seja >=3 e <=5 se correlação_delta_agregados = 'Barrera'
if 3 <= tipo_regressão_equilíbrio <= 5 and correlação_delta_agregados != 'Barrera':
    mensagem = "\nATENCAO: Corrija o arquivo 'variaveis_entrada_codigo.txt'"
    mensagem += f"\nPONTO A CORRIGIR: a variavel 'tipo_regressao' = {tipo_regressão_equilíbrio} " \
                f"exige que 'correlacao_delta_agregados == 'Barrera'."
    raise ValueError(mensagem)

# 1.3 - Informações experimentais do sistema
diretório_do_xlsx = os.path.join(diretório_deste_módulo, 'dados_experimentais.xlsx')
SARA, T, solvente, ws_simplificados, yields_exp = ler_dados_experimentais(diretório_do_xlsx, nome_planilha)
SARA = normalizar_composição(SARA)  # normalização da composição SARA

# ======================================================================================================================
# PARTE 2 - PROPRIEDADES DO SOLVENTE, SATURADOS, AROMÁTICOS E RESINAS

# 2.1 - Inicialização dos arrays de massas molares, densidades, parâmetros de solubilidade e volumes molares
# de todos os componentes do sistema
# Obs: Estrutura do array: [Solvente, S, A, R, Asf0, Asf1, ...]
MMs, rhos, deltas, Vs = [np.zeros(4 + n_agregados) for _ in range(4)]

# 2.2 - Propriedades do solvente
MMs[0], rhos[0], deltas[0], Vs[0] = calcular_propriedades_solvente(T, solvente)

# 2.3 - Propriedades dos saturados, aromáticos e resinas
MMs[1], rhos[1], deltas[1], Vs[1] = calcular_propriedades_saturados(
    T, correlação_densidade_saturados, correlação_delta_saturados)
MMs[2], rhos[2], deltas[2], Vs[2] = calcular_propriedades_aromáticos(
    T, correlação_densidade_aromáticos, correlação_delta_aromáticos)
MMs[3], rhos[3], deltas[3], Vs[3] = calcular_propriedades_resinas(
    T, correlação_densidade_resinas, correlação_delta_resinas)

# ======================================================================================================================
# PARTE 3 - CRIAÇÃO DA FUNÇÃO OBJETIVO PARA REGRESSÃO DOS PARÂMETROS
# Este bloco será pulado caso tipo_cálculo_programa == 'predicao'

if tipo_cálculo_equilíbrio == 'regressao':

    def F_obj(parâmetros, *args):

        global MWavg, alfa, c_delta_agregados, Alinha_delta_agregados, d_delta_agregados

        # 3.1 - Desempacotando os parâmetros a serem estimados
        match tipo_regressão_equilíbrio:
            case 1: MWavg = parâmetros
            # OBS: Como alfa, c_delta_agregados, Alinha_delta_agregados, d_delta_agregados não serão estimadas,
            #      usam-se os valores lidos na 'PARTE 1' deste código
            case 2: MWavg, alfa = parâmetros
            # OBS: Como c_delta_agregados, Alinha_delta_agregados, d_delta_agregados não serão estimadas,
            #      usam-se os valores lidos na 'PARTE 1' deste código
            case 3: MWavg, alfa, c_delta_agregados = parâmetros
            # OBS: Alinha_delta_agregados, d_delta_agregados não serão estimadas,
            #      usam-se os valores lidos na 'PARTE 1' deste código
            case 4: MWavg, alfa, c_delta_agregados, Alinha_delta_agregados = parâmetros
            # OBS: d_delta_agregados não será estimado, usa-se os valores lidos na 'PARTE 1' deste código
            case 5: MWavg, alfa, c_delta_agregados, Alinha_delta_agregados, d_delta_agregados = parâmetros
            case _: MWavg, alfa = parâmetros
            # OBS: Em caso de erro, usa-se tipo_regressão == 2 como padrão.

        # 3.2 - Desempacotando os *args (outros argumentos da função 'F_obj' a serem passados pra função 'minimize')
        T, SARA, ws_simplificados, yields_exp = args[0]
        MMs, rhos, deltas, Vs = args[1]
        n_agregados, MWmin, MWmax, tipo_cálculo_MM_agregados, método_integração_FDP_Gamma = args[2]

        # 3.3 - Propriedades dos agregados de asfaltenos
        MMsagregados, wsagregados, xsagregados = gerar_distribuição_massa_molar(
            alfa, MWavg, n_agregados, MWmin, MWmax, tipo_cálculo_MM_agregados, método_integração_FDP_Gamma)
        wsagregados = normalizar_composição(wsagregados) 
        xsagregados = normalizar_composição(xsagregados)
        rhosagregados, deltasagregados, Vsagregados = calcular_propriedades_agregados(
            T, MMsagregados, correlação_densidade_agregados, correlação_delta_agregados,
            Alinha_delta_agregados, c_delta_agregados, d_delta_agregados)
        
        # 3.4 - Alocação das propriedades dos agregados de asfaltenos
        # nos arrays que contêm as propriedades de todos os componentes do sistema
        MMs[4:4 + n_agregados] = MMsagregados[0:n_agregados]*1e-3  # kg/mol
        rhos[4:4 + n_agregados] = rhosagregados[0:n_agregados] 
        deltas[4:4 + n_agregados] = deltasagregados[0:n_agregados]
        Vs[4:4 + n_agregados] = Vsagregados[0:n_agregados] 
        
        # 3.5 - Composição global do sistema em termos de [Solvente, S, A, R, Asf0, Asf1, ...] (base mássica e molar)
        ws_completo, xs_completo = fracionar_composição_global(ws_simplificados, SARA, wsagregados, MMs)
        ws_completo = np.apply_along_axis(func1d=normalizar_composição, axis=1, arr=ws_completo)
        xs_completo = np.apply_along_axis(func1d=normalizar_composição, axis=1, arr=xs_completo)

        # 3.6 - Cálculo de equilíbrio líquido-líquido
        n_dados_exp = yields_exp.shape[0]
        yields_calc = np.zeros(n_dados_exp)
        for i_comp in range(n_dados_exp):
            betarr, xsL, xsH, n_it = calcular_composições_ELL(T, xs_completo[i_comp], deltas, Vs, xsagregados, MMs)
            yields_calc[i_comp] = calcular_yield_asfaltenos(betarr, xsL, xsH, MMs)

        # 3.7 - Expressão matemática a ser minimizada
        yields_diferenças = np.abs(yields_calc - yields_exp)  # diferenças entre os yields calculados e experimentais

        return (1/n_dados_exp)*yields_diferenças.sum()
    
# ======================================================================================================================
# PARTE 4 - MINIMIZAÇÃO DA FUNÇÃO OBJETIVO PARA REGRESSÃO DOS PARÂMETROS
# Este bloco é pulado caso tipo_cálculo_programa == 'predicao'

    # 4.1 - Chutes iniciais dos parâmetros a serem estimados
    match tipo_regressão_equilíbrio:
        case 1: chute_inicial = np.array([MWavg])
        case 2: chute_inicial = np.array([MWavg, alfa])
        case 3: chute_inicial = np.array([MWavg, alfa, c_delta_agregados])
        case 4: chute_inicial = np.array([MWavg, alfa, c_delta_agregados, Alinha_delta_agregados])
        case 5: chute_inicial = np.array([MWavg, alfa, c_delta_agregados, Alinha_delta_agregados, d_delta_agregados])
        case _: chute_inicial = np.array([MWavg, alfa])
        # OBS: em caso de erro, usa-se tipo_regressão == 2 como padrão.

    # 4.2 - Atribuição de valores para os *args (outros argumentos da função 'F_obj'
    # a serem passados pra função 'minimize')
    dados_experimentais = (T, SARA, ws_simplificados, yields_exp) 
    propriedades_componentes = (MMs, rhos, deltas, Vs)
    variáveis_distribuição_massa_molar = (
        n_agregados, MWmin, MWmax, tipo_cálculo_MM_agregados, método_integração_FDP_Gamma
    )
    argumentos_otimização = (dados_experimentais, propriedades_componentes, variáveis_distribuição_massa_molar)

    # 4.3 - Otimização
    # 4.3.1 - Configuração do algoritmo de otimização
    if algoritmo_otimização == 1:
        # 4.3.1.1 - Nelder-Mead
        sol = scp.optimize.minimize(F_obj, chute_inicial, method="Nelder-Mead", args=argumentos_otimização)

    elif algoritmo_otimização == 2:
        # 4.3.1.2 - Brute-force
        sol = 0
        pass  # Obs: ainda falta ser implementado

    elif algoritmo_otimização == 3 or algoritmo_otimização == 4:
        # Configuração dos limites nos valores dos parâmetros a serem estimados
        match tipo_regressão:
            case 1:
                limites_parâmetros = [(1.2 * MWmin, 1e4)]
                # MWavg
            case 2:
                limites_parâmetros = [(1.2 * MWmin, 1e4), (1.15, 80)]
                # MWavg, alfa
            case 3:
                limites_parâmetros = [(1.2 * MWmin, 1e4), (1.15, 80), (0.634, 0.672)]
                # MWavg, alfa, c_delta_agregados
                # Obs: limites de c_delta_agregados com base na pg. 87 da tese de Diana Maria Barrera (2012)
            case 4:
                limites_parâmetros = [(1.2 * MWmin, 1e4), (1.15, 80), (0.634, 0.672), (0, 0.03)]
                # MWavg, alfa, c_delta_agregados, Alinha_delta_agregados
                # Obs: limites de Alinha_delta_agregados, c_delta_agregados com base nas pg. 85-87
                #      da tese de Diana Maria Barrera (2012)
            case 5:
                limites_parâmetros = [(1.2 * MWmin, 1e4), (1.15, 80), (0.634, 0.672), (0, 0.03), (0.0494, 0.0496)]
                # MWavg, alfa, c_delta_agregados, Alinha_delta_agregados, d_delta_agregados
                # Obs: limites de Alinha_delta_agregados, c_delta_agregados, d_delta_agregados com base nas pg. 85-87
                #      da tese de Diana Maria Barrera (2012)
            case _:
                limites_parâmetros = [(1.2 * MWmin, 1e4), (1.15, 80), (0.634, 0.672)]
            # MWavg, alfa, c_delta_agregados
            # Obs: limites de c_delta_agregados com base na pg. 87 da tese de Diana Maria Barrera (2012)
            # OBS: em caso de erro, usar tipo_regressão == 3 como padrão.

        if algoritmo_otimização == 3:
            # 4.3.1.3 - L-BFGS-B
            sol = scp.optimize.minimize(F_obj, chute_inicial, method="L-BFGS-B", bounds=limites_parâmetros,
                                        args=argumentos_otimização)
        else:
            # 4.3.1.4 - Powell
            sol = scp.optimize.minimize(F_obj, chute_inicial, method="Powell", bounds=limites_parâmetros,
                                        args=argumentos_otimização)

    else:  # Caso Erro
        print("Problema na escolha da variável algoritmo_otimização.")
        sol = 0

    # 4.4 - Alocação dos parâmetros estimados
    match tipo_regressão_equilíbrio:
        case 1: MWavg = sol.x[0]
        case 2: MWavg, alfa = sol.x
        case 3: MWavg, alfa, c_delta_agregados = sol.x
        case 4: MWavg, alfa, c_delta_agregados, Alinha_delta_agregados = sol.x
        case 5: MWavg, alfa, c_delta_agregados, Alinha_delta_agregados, d_delta_agregados = sol.x
        case _: print("Problema na escolha da variavél tipo_regressão.")  # Caso Erro

# ======================================================================================================================
# PARTE 5 - PREDIÇÃO DA CURVA DE SOLUBILIDADE

# 5.1 - Propriedades dos agregados de asfaltenos
# 5.1.1 - Massas molares, frações mássicas e frações molares
MMsagregados, wsagregados, xsagregados = gerar_distribuição_massa_molar(
    alfa, MWavg, n_agregados, MWmin, MWmax, tipo_cálculo_MM_agregados, método_integração_FDP_Gamma)
wsagregados = normalizar_composição(wsagregados)  # normalização das frações mássicas
xsagregados = normalizar_composição(xsagregados)  # normalização das frações molares

# 5.1.2 - Densidades, parâmetros de solubilidades e volumes molares
rhosagregados, deltasagregados, Vsagregados = calcular_propriedades_agregados(
    T, MMsagregados, correlação_densidade_agregados, correlação_delta_agregados,
    Alinha_delta_agregados, c_delta_agregados, d_delta_agregados)

# 5.1.3 - Alocação das propriedades dos agregados de asfaltenos nos arrays das propriedades dos componentes do sistema
MMs[4:4 + n_agregados] = MMsagregados[0:n_agregados]*1e-3  # kg/mol
rhos[4:4 + n_agregados] = rhosagregados[0:n_agregados] 
deltas[4:4 + n_agregados] = deltasagregados[0:n_agregados]
Vs[4:4 + n_agregados] = Vsagregados[0:n_agregados] 

# 5.2 - COMPOSIÇÃO GLOBAL DO SISTEMA EM TERMOS DE [Solvente, S, A, R, Asf0, Asf1, ...] (base mássica e base molar)
ws_completo, xs_completo = fracionar_composição_global(ws_simplificados, SARA, wsagregados, MMs)
ws_completo = np.apply_along_axis(func1d=normalizar_composição, axis=1, arr=ws_completo)
# normalização das frações mássicas de cada linha
xs_completo = np.apply_along_axis(func1d=normalizar_composição, axis=1, arr=xs_completo)
# normalização das frações molares de cada linha

# 5.3 - CÁLCULO DE EQUILÍBRIO LÍQUIDO-LÍQUIDO
# 5.3.1 - Nº de dados experimentais
n_dados_exp = yields_exp.shape[0]

# 5.3.2 - Inicialização dos arrays que armazenarão os resultados dos cálculos de ELL
betas = np.zeros(n_dados_exp)  # betas de Rachford-Rice
xsL, xsH = [np.zeros((n_dados_exp, 4 + n_agregados)) for _ in range(2)]
# composição da fase leve, composição da fase pesada
somaxsL, somaxsH = [np.zeros(n_dados_exp) for _ in range(2)]
# soma da composição da fase leve, soma da composição da fase pesada
n_it = np.zeros(n_dados_exp)  # nº de iterações

# 5.3.3 - Inicialização do array que armazenará os yields calculados
yields_calc = np.zeros(n_dados_exp)

# 5.3.4 - Cálculos das composições de ELL e yields de asfaltenos p/ cada i-ésimo dado experimental
for i in range(n_dados_exp):
    betas[i], xsL[i, :], xsH[i, :], n_it[i] = calcular_composições_ELL(T, xs_completo[i], deltas, Vs, xsagregados, MMs)
    somaxsL[i], somaxsH[i] = np.round(xsL[i, :].sum(), decimals=8), np.round(xsH[i, :].sum(), decimals=8)
    yields_calc[i] = calcular_yield_asfaltenos(betas[i], xsL[i, :], xsH[i, :], MMs)

# ======================================================================================================================
# PARTE 6 - REGRESSÃO DOS DADOS CINÉTICOS

if tipo_cálculo_cinética != "nao":

    # 6.1 - Parâmetros do Modelo Cinético
    # 6.1.1 - Relacionados ao Equilíbrio de Fases
    w_solv = np.asarray(ws_simplificados[:, 0], dtype=float)
    yields_eq = (yields_exp if np.any(yields_exp != 0) else yields_calc)
    yield_max = yields_eq[-1]
    w_onset = 0
    for i_onset in range(len(yields_eq)):
        if yields_eq[i_onset] > 0.005:
            w_onset = w_solv[i_onset] if i_onset == 0 else w_solv[i_onset - 1]
            break

    # 6.1.2 - Parâmetros para Otimização
    tempos = [2, 4, 6, 8, 16, 24]
    parâmetros_cinéticos_1 = np.array([683.75, 0.030])  # Relacionados ao Equilíbrio de Fases
    parâmetros_cinéticos_2 = np.array([5, -2])  # Relacionados ao Tempo

    # 6.1.3 - Dados Experimentais Temporais (MATRIZ ZERO PARA TESTES)
    yields_temp_exp = np.zeros((len(tempos), len(yields_eq)))

    # 6.3 - Otimização dos Parâmetros Cinéticos de Tempo Infinito
    if tipo_cálculo_cinética in ("regressao1", "regressao2"):
        # 6.3.1 - Função Objetivo
        def F_obj_1(parâmetros_opt, yields_eq, w_solv, w_onset, yield_max):
            yields_t_inf = calcular_yield_tempo_infinito(w_solv, w_onset, yield_max, parâmetros_opt)
            return np.mean(np.abs(yields_t_inf - yields_eq))

        # 6.3.2 - Otimização dos Parâmetros
        argumentos_F_obj_1 = (yields_otimização, w_solv, w_onset, yield_max)
        sol_1 = scp.optimize.minimize(F_obj_1, parâmetros_cinéticos_1, method="Nelder-Mead", args=argumentos_F_obj_1)
        parâmetros_cinéticos_1 = sol_1.x

    # 6.4 - Otimização dos Parâmetros Cinéticos Temporais
    if tipo_cálculo_cinética == "regressao2":
        # 6.4.1 - Função Objetivo
        def F_obj_2(parâmetros, tempos, yields_temp_exp, yield_max, parâmetros_t_inf, w_solv, w_onset):
            yields_ts = calcular_yields_temporais(tempos, w_solv, w_onset, yield_max, parâmetros_t_inf, parâmetros)
            return np.mean(np.abs(yields_ts - yields_temp_exp))

        # 6.4.2 - Otimização dos Parâmetros
        argumentos_F_obj_2 = (tempos, yields_temp_exp, yield_max, parâmetros_cinéticos_1, w_solv, w_onset)
        sol_2 = scp.optimize.minimize(F_obj_2, parâmetros_cinéticos_2, method='Nelder-Mead', args=argumentos_F_obj_2)
        parâmetros_cinéticos_2 = sol_2.x

    # 6.5 - Predição do Modelo Cinético
    yields_temp_calc = calcular_yields_temporais(
        tempos, w_solv, w_onset, yield_max, parâmetros_cinéticos_1, parâmetros_cinéticos_2
    )

    # 6.5.1 - Criação do Gráfico de Curvas Cinéticas em diferentes tempos
    plotar_yield_curves_cinéticas(w_solv, tempos, yields_exp, yields_temp_calc)

# ======================================================================================================================
# PARTE 7 - EXIBIÇÃO DOS RESULTADOS
 
# 7.1 - Se há dados experimentais de yields para o sistema em questão... (Ex: Yanes_P1 e Yanes_P2)
if any(yield_exp != 0 for yield_exp in yields_exp):

    # 7.1.1 - Desvios absolutos (DAs) e médio dos desvios absolutos (DMA)
    DAs = np.abs(yields_exp - yields_calc)  # desvios absolutos fracionais
    DMA = np.mean(DAs)  # média dos desvios absolutos fracionais

    # 7.1.2 - Criação de listas com os resultados formatados
    yields_exp_formatado = [f"{100*yield_exp:.2f}%" for yield_exp in yields_exp]
    yields_calc_formatado = [f"{100*yield_calc:.2f}%" for yield_calc in yields_calc]
    betas_formatado = [f"{beta:.4e}" for beta in betas]
    DAs_formatado = [f"{100*DA:.2f}%" for DA in DAs]
    DMA_formatado = f"{100*DMA:.4f}%"

# 7.2 - Se não há dados experimentais de yields para o sistema em questão...
# (Ex: Yanes_P3, Tharanivasan_Lloydminster1 e Tharanivasan_Lloydminster2)
else:

    # 7.2.1 - Criação de listas com os resultados formatados
    yields_exp_formatado = ["nao disponivel" for yield_calc in yields_calc]
    yields_calc_formatado = [f"{100*yield_calc:.2f}%" for yield_calc in yields_calc]
    betas_formatado = [f"{beta:.4e}" for beta in betas]
    DAs_formatado = ["nao disponivel" for yield_calc in yields_calc]
    DMA_formatado = "nao disponivel"

# 7.3 - Criação e impressão de Dataframe com os resultados
df_resultados = pd.DataFrame(
    {"  Fracao Solvente  ": ws_simplificados[:, 0],
     "  yield (exp.)  ": yields_exp_formatado,
     "  yield (calc.)  ": yields_calc_formatado,
     "  DA (%)  ": DAs_formatado,
     "  Beta  ": betas_formatado,
     "  somaxsL  ": somaxsL,
     "  somaxsH  ": somaxsH,
     "  qte. iteracoes  ": list(map(int, n_it))}
     )
print(f"\n| DESVIO MEDIO ABSOLUTO NOS YIELDS (%): {DMA_formatado}")
if tipo_cálculo_equilíbrio == 'regressao':
    print(f"PARAMETROS ESTIMADOS: {sol.x}")
    print(f"{tabulate(df_resultados, headers = df_resultados.columns, tablefmt = 'pretty', showindex = False)}")

# 7.4 - Criação dos gráficos: yield curves e distribuição de massa molar
informações_auxiliares = [DMA_formatado, tipo_cálculo_equilíbrio, tipo_regressão_equilíbrio,
                          algoritmo_otimização, nome_planilha]
plotar_yield_curves(ws_simplificados[:, 0], yields_exp, yields_calc, informações_auxiliares)
plotar_distribuição_massa_molar(MMsagregados, xsagregados, alfa, MWavg, informações_auxiliares)
