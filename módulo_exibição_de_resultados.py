# Importação de bibliotecas do python
import os
import numpy as np
import pandas as pd
import scipy as scp
import matplotlib.pyplot as plt
from tabulate import tabulate
from matplotlib.ticker import MultipleLocator

# Importação de módulos internos
from módulo_cálculos_erros import absolute_deviations, absolute_relative_deviations, average_absolute_deviation, \
    average_absolute_relative_deviation


# Função
def exibir_resultados_equilíbrio(nome_planilha, config, sistema, propriedades, params, yields_exp, SARA):

    # Escolha do referencial das yield curves para geração dos gráficos
    valores_x, x_label = referencial_para_yield_curves(config.cálculo.x_yield_curve, propriedades,
                                                       sistema.fase_global, sistema.fase_leve)

    # Criação e impressão de Dataframe com os resultados
    erro_geral = criar_tabela_dataframe(valores_x, x_label, yields_exp, sistema.yields_calc(propriedades.MMs),
                                        sistema.betas, params, config)

    # 6.3 - Criação dos gráficos:
    # 6.3.1 - Gráficos de yield curve (equilíbrio) e distribuição de massa molar
    if config.cálculo.plotar_gráficos:
        informações_auxiliares = [erro_geral, x_label, params.variáveis_regressão, config.cálculo, nome_planilha]
        plotar_yield_curves(valores_x, yields_exp, sistema.yields_calc(propriedades.MMs), SARA[-1], informações_auxiliares)
        plotar_distribuição_massa_molar(propriedades.asfaltenos, params, informações_auxiliares)


# Função
def exibir_resultados_cinética(config, nome_planilha, tempos, ws_precipitante, taus, params,
                               yields_eq_calc, yields_temp_exp, yields_temp_calc):

    print("taus:", np.round(taus, decimals=2))
    print("Parâmetros estimados:", {variável: f"{getattr(params, variável):.2f}"
                                    for variável in params.variáveis_regressão})

    # 5.5 - Criação do Gráfico de Curvas Cinéticas em diferentes tempos
    if config.cálculo.plotar_gráficos:
        informações_auxiliares = [config.cálculo.tipo_cálculo_cinética, params.variáveis_regressão, nome_planilha]
        plotar_yield_curves_cinéticas(
            ws_precipitante, tempos, yields_eq_calc, yields_temp_exp, yields_temp_calc, informações_auxiliares)


# Subfunção
def referencial_para_yield_curves(x_yield_curve, propriedades, fase_global, fase_leve):
    if x_yield_curve == 'solubilidade':
        phis_L = converter_fração_molar_para_fração_volumétrica(fase_leve.xs_completo, propriedades.Vs)
        deltas_L = (1/1000) * ((phis_L * propriedades.deltas[None, :] ** 2).sum(axis=1)) ** 0.5  # MPa**0.5
        return deltas_L, "Parâmetro de Solubilidade (MPa$^{0.5}$)"
    else:
        atributos = {'molar': ('xs', 'Molar'),
                     'massa': ('ws', 'Mássica'),
                     'volume:': ('phis', 'Volumétrica')}
        atributo, nome = atributos.get(x_yield_curve, ("ws", "Mássica"))
        return getattr(fase_global, atributo)[:, 0], f"Fração {nome} {propriedades.precipitante.nome} (%)"


# Subfunção
def criar_tabela_dataframe(valores_x, x_label, yields_exp, yields_calc, betas, params, config):
    # 6.1.1 - Se há dados experimentais de yields para o sistema em questão
    no_experimental_data = True if all(yield_exp == 0 for yield_exp in yields_exp) else False

    # 6.1.2 - Criação de listas com os resultados formatados
    ADs = None if no_experimental_data else absolute_deviations(yields_exp, yields_calc)
    AAD = None if no_experimental_data else average_absolute_deviation(yields_exp, yields_calc)

    ADs_formatado = ["não disponível" for yield_calc in yields_calc] if no_experimental_data \
        else [f"{100 * AD:.2f}%" for AD in ADs]
    AAD_formatado = "não disponível" if no_experimental_data else f"{100 * AAD:.4f}%"

    valores_x_formatado = [f"{100 * valor_x:.2f}%" for valor_x in valores_x] if x_label != "Parâmetro de Solubilidade" \
        else [f"{valor_x:.2f} MPa^0.5" for valor_x in valores_x]
    yields_exp_formatado = ["nao disponivel" for yield_exp in yields_exp] if no_experimental_data \
        else [f"{100 * yield_exp:.2f}%" for yield_exp in yields_exp]
    yields_calc_formatado = [f"{100 * yield_calc:.2f}%" for yield_calc in yields_calc]
    betas_formatado = [f"{beta:.4e}" for beta in betas]

    # 6.2 - Criação e impressão de Dataframe com os resultados
    df_resultados = pd.DataFrame(
        {f"  {x_label}  ": valores_x_formatado,
         "  Yield (exp.)  ": yields_exp_formatado,
         "  Yield (calc.)  ": yields_calc_formatado,
         "  AD (%)  ": ADs_formatado,
         "  Beta  ": betas_formatado,
         # "  qte. iteracoes  ": list(map(int, n_it))
         })

    if config.cálculo.tipo_cálculo_equilíbrio:
        print("Parâmetros estimados:", {variável: f"{getattr(params, variável):.2f}"
                                        for variável in params.variáveis_regressão})
    print()
    print(f"Desvio Médio Absoluto | AAD (%): {AAD_formatado}")
    print(f"{tabulate(df_resultados, headers=df_resultados.columns, tablefmt='pretty', showindex=False)}")

    return AAD_formatado


# ==================================================================================================================== #
# GERAÇÃO DE GRÁFICOS
# Função
def plotar_yield_curves(x_valores, y_valores_exp, y_valores_calc, y_valores_max, informações_auxiliares):
    """ Cria um gráfico contendo as curvas de solubilidade experimental e calculada.
    
    Inputs:
        ws_solvente (array)           : frações mássicas de solvente
        yields_exp (array)            : yields fracionais de asfaltenos (experimentais)
        yields_calc (array)           : yields fracionais de asfaltenos (calculados)
        yields_max (array)            : fração de asfaltenos pela caracterização SARA
        informações_auxiliares (list) : lista dos elementos [DMA_formatado, tipo_cálculo_programa, nome_planilha]
                                        a lista acima contém informações úteis para o nome do arquivo do gráfico a ser
                                        salvo na pasta 'Resultados'

    Outputs:
        Mostra o gráfico e o salva na pasta Resultados
    """

    # Desempacotando a lista 'informações_auxiliares'
    erro_geral, x_label, variáveis_regressão, cálculo, nome_planilha = informações_auxiliares
    ymax = 10 * ((max(y_valores_max, max(y_valores_exp), max(y_valores_exp)) // 0.10) + 1)

    # Ajuste para a variável do eixo x
    x_valores = 100 * x_valores
    xmin, xmax = 0, 100
    if x_label == "Parâmetro de Solubilidade (MPa**0.5)":
        x_valores = (1/100) * x_valores
        xmin, xmax = 15, 25

    # Título
    plt.title(f"YIELD CURVE ({cálculo.x_yield_curve}) - DMA(%): {erro_geral}", fontsize=16, fontweight="bold")

    # Série de dados experimentais
    # Obs: se todos os yields experimentais são nulos, a curva experimental é plotada na cor branca (desaparece)
    if all(y_valor_exp == 0 for y_valor_exp in y_valores_exp):
        plt.plot(x_valores, 100 * y_valores_exp, "o", mfc="white", mec="white", markersize=10)
    else:
        plt.plot(x_valores, 100 * y_valores_exp, "o", mfc="orange", mec="black", markersize=10)
    
    # Série de dados calculada
    plt.plot(x_valores, 100 * y_valores_calc, "o", mfc="blue", mec="black", markersize=10)
    plt.axhline(y=100*y_valores_max, linestyle='--', color='gray', label='Máximo')

    # Legenda
    plt.legend(["experimental", "calculado"], fontsize=12, loc="upper left")
    
    # Títulos dos eixos, valores min e max de cada eixo, fontes das marcas de escala, marcas de escala secundárias
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel("yield de asfalteno (%)", fontsize=14)
    plt.axis(xmin=xmin, xmax=xmax, ymin=0, ymax=ymax)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.gca().xaxis.set_minor_locator(MultipleLocator(5))  # Marcas de escala secundárias no eixo x
    plt.gca().yaxis.set_minor_locator(MultipleLocator(0.5))  # Marcas de escala secundárias no eixo y

    # Linhas de grade
    plt.grid(color="k", linestyle="-", linewidth=0.1)

    # Nome do arquivo do gráfico a ser salvo
    if cálculo.tipo_cálculo_equilíbrio == 'predição':
        nome_arquivo_gráfico = f"{nome_planilha}_YIELDCURVE_({cálculo.x_yield_curve}).png"
    else:
        nome_arquivo_gráfico = f"{nome_planilha}_YIELDCURVE_({cálculo.x_yield_curve})_{variáveis_regressão}_.png"
    
    # Salvando o gráfico
    diretório_da_pasta_deste_modulo = os.path.dirname(os.path.abspath(__file__))
    diretório_png = os.path.join(diretório_da_pasta_deste_modulo, "Resultados", "Resultados_Equilíbrio",
                                 cálculo.tipo_cálculo_equilíbrio, nome_arquivo_gráfico)
    plt.savefig(diretório_png, dpi=300, bbox_inches="tight")

    # Fechando o arquivo após salvá-lo
    plt.close()

    pass


# Função
def plotar_distribuição_massa_molar(asfaltenos, params, informações_auxiliares):
    """ Cria um gráfico contendo as distribuição de massa molar.
    
    Inputs:
        MMsagregados (array)          : massas molares dos agregados de asfaltenos (g/mol) 
        xsagregados (array)           : frações molares dos agregados de asfaltenos
        alfa (float)                  : parâmetro de forma da função densidade de probabilidade da
                                        distribuição Gamma (FDP_Gamma)
        MWavg (float)                 : massa molar média dos agregados de asfaltenos (g/mol)
        informações_auxiliares (list) : lista dos elementos [DMA_formatado, tipo_cálculo_programa, nome_planilha]
                                        a lista acima contém informações úteis para o nome do arquivo do gráfico
                                        a ser salvo na pasta 'Resultados'

    Outputs:
        Mostra o gráfico e o salva na pasta Resultados
    """

    # Desempacotando a lista 'informações_auxiliares'
    erro_geral, x_label, variáveis_regressão, cálculo, nome_planilha = informações_auxiliares

    # Título
    plt.title(f"DIST. MASSA MOLAR - DMA(%): {erro_geral}", fontsize=16, fontweight="bold")

    # Série de dados
    plt.plot(asfaltenos.MM, asfaltenos.x, "o-", markersize=9, mfc="white", mec="black", color="black")

    # Legenda
    plt.legend([f"alfa = {params.alfa:.4f}\nMM avg = {params.MW_avg:.2f} g/mol"], fontsize=12)

    # Títulos dos eixos, fontes das marcas de escala
    plt.xlabel("Massa molar (g/mol)", fontsize=14)
    plt.ylabel("Fração molar", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Linhas de grade
    plt.grid(color="k", linestyle="-", linewidth=0.1)

    # Nome do arquivo do gráfico a ser salvo
    if cálculo.tipo_cálculo_equilíbrio == 'predição':
        nome_arquivo_gráfico = f"{nome_planilha}_DISTMASSAMOLAR.png"
    else:
        nome_arquivo_gráfico = f"{nome_planilha}_DISTMASSAMOLAR_tipo_regressao_{variáveis_regressão}.png"
    
    # Salvando o gráfico
    diretório_da_pasta_deste_modulo = os.path.dirname(os.path.abspath(__file__))
    diretório_png = os.path.join(diretório_da_pasta_deste_modulo, "Resultados", "Distribuição Massa Molar",
                                 cálculo.tipo_cálculo_equilíbrio, nome_arquivo_gráfico)
    plt.savefig(diretório_png, dpi=300, bbox_inches="tight")

    # Fechando o arquivo após salvá-lo
    plt.close()

    pass


def plotar_yield_curves_cinéticas(x_valores, tempos, y_valores_eq, y_valores_exp, y_valores_calc,
                                  informações_auxiliares):
    """ Cria um gráfico contendo as curvas de solubilidade calculadas em diferentes tempos,
        a curva de equilíbrio (modelo) e a experimental.

    Inputs:
        ws_solvente (array)           : frações mássicas de solvente
        yields_exp (array)            : yields fracionais de asfaltenos (experimentais)
        yields_calc (array)           : yields fracionais de asfaltenos (experimentais)
        informações_auxiliares (list) : lista dos elementos [DMA_formatado, tipo_cálculo_programa, nome_planilha]
                                        a lista acima contém informações úteis para o nome do arquivo do gráfico a ser
                                        salvo na pasta 'Resultados'

    Outputs:
        Mostra o gráfico e o salva na pasta Resultados
    """

    # Desempacotando a lista 'informações_auxiliares'
    tipo_cálculo, variáveis_regressão, nome_planilha = informações_auxiliares

    # Título
    plt.title("YIELD CURVES - TIME ANALYSIS")

    # Série de dados experimentais
    # Obs: se todos os yields experimentais são nulos, a curva experimental é plotada na cor branca (desaparece)
    # if all(yield_exp == 0 for yield_exp in yields_exp):
    #     plt.plot(100 * ws_solvente, 100 * yields_exp, "o", mfc="white", mec="white", markersize=10)
    # if any(yield_exp != 0 for yield_exp in yields_exp):
    #     plt.plot(100 * ws_solvente, 100 * yields_exp, "o", mfc="blue", mec="black", markersize=10)

    # Série de dados experimental e calculada em diferentes tempos
    cmap = plt.get_cmap("tab10")
    for t in range(len(tempos)):
        color = cmap(t)
        plt.plot(100 * x_valores, 100 * y_valores_exp[t, :], "o", mfc=color, mec="black", label=f"{tempos[t]}h (exp)")
        plt.plot(100 * x_valores, 100 * y_valores_calc[t, :], c=color, label=f"{tempos[t]}h (calc)", ls='-', lw='2')

    # Série de dados de equilíbrio
    plt.plot(100 * x_valores, 100 * y_valores_eq, c="red", ls='--', lw='2', label="equilíbrio")

    # Títulos dos eixos, valores min e max de cada eixo, fontes das marcas de escala, marcas de escala secundárias
    plt.xlabel("fração de solvente, wt%", fontsize=14)
    plt.ylabel("yield de asfalteno, wt%", fontsize=14)
    plt.legend(fontsize=12, loc="upper left")
    plt.axis(xmin=0, xmax=100, ymin=0)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.gca().xaxis.set_minor_locator(MultipleLocator(5))  # Marcas de escala secundárias no eixo x
    plt.gca().yaxis.set_minor_locator(MultipleLocator(0.5))  # Marcas de escala secundárias no eixo y

    # Linhas de grade
    plt.grid(color="k", ls="-", lw=0.1)

    # Nome do arquivo do gráfico a ser salvo
    if tipo_cálculo == 'predição':
        nome_arquivo_gráfico = f"{nome_planilha}_YIELDCURVE_kinectics.png"
    else:
        nome_arquivo_gráfico = f"{nome_planilha}_YIELDCURVE_kinectics_{variáveis_regressão}.png"

    # Salvando o gráfico
    diretório_da_pasta_deste_modulo = os.path.dirname(os.path.abspath(__file__))
    diretório_png = os.path.join(diretório_da_pasta_deste_modulo, "Resultados", "Resultados_Cinética",
                                 tipo_cálculo, nome_arquivo_gráfico)
    plt.savefig(diretório_png, dpi=300, bbox_inches="tight")

    # Fechando o arquivo após salvá-lo
    plt.close()

    pass
