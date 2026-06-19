# Importação de bibliotecas do python
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os


# Função
def plotar_yield_curves(eixo_x, yields_exp, yields_calc, yield_max, informações_auxiliares):
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
    DMA_formatado, regressao, tipo_regressão, algoritmo_otimização, variável, nome_planilha = informações_auxiliares
    ymax = 10 * ((max(yield_max, max(yields_exp), max(yields_calc)) // 0.10) + 1)

    # Ajuste para a variável do eixo x
    eixo_x = 100 * eixo_x
    xmin, xmax = 0, 100
    match variável:
        case "molar":
            x_label = "fração molar de alcano (%)"
        case "mass":
            x_label = "fração mássica de alcano (%)"
        case "volume":
            x_label = "fração volumétrica de alcano (%)"
        case "solubility":
            x_label = "parâmetro de solubilidade (MPa**0.5)"
            eixo_x = (1/100) * eixo_x
            xmin, xmax = 15, 21
        case _:
            x_label = "fração molar de alcano (%)"  # Em caso de erro, usar fração mássica.

    # Título
    plt.title(f"YIELD CURVE ({variável}) - DMA(%): {DMA_formatado}", fontsize=16, fontweight="bold")

    # Série de dados experimentais
    # Obs: se todos os yields experimentais são nulos, a curva experimental é plotada na cor branca (desaparece)
    if all(yield_exp == 0 for yield_exp in yields_exp):
        plt.plot(eixo_x, 100*yields_exp, "o", mfc="white", mec="white", markersize=10)
    if any(yield_exp != 0 for yield_exp in yields_exp):
        plt.plot(eixo_x, 100*yields_exp, "o", mfc="orange", mec="black", markersize=10)
    
    # Série de dados calculada
    plt.plot(eixo_x, 100*yields_calc, "o", mfc="blue", mec="black", markersize=10)
    plt.axhline(y=100*yield_max, linestyle='--', color='gray', label='Máximo')

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
    if not regressao:
        nome_arquivo_gráfico = f"{nome_planilha}_YIELDCURVE_{variável}.png"
    else:
        nome_arquivo_gráfico = f"{nome_planilha}_YIELDCURVE_{variável}_tipo_reg_{tipo_regressão}_" \
                               f"algoritmo_opt_{algoritmo_otimização}.png"
    
    # Salvando o gráfico
    diretório_da_pasta_deste_modulo = os.path.dirname(os.path.abspath(__file__))
    if not regressao:
        diretório_png = os.path.join(diretório_da_pasta_deste_modulo, "Resultados",
                                     "Resultados_Equilíbrio", "Predição", nome_arquivo_gráfico)
    else:
        diretório_png = os.path.join(diretório_da_pasta_deste_modulo, "Resultados",
                                     "Resultados_Equilíbrio", "Regressão", nome_arquivo_gráfico)
    plt.savefig(diretório_png, dpi=300, bbox_inches="tight")

    # Fechando o arquivo após salvá-lo
    plt.close()

    pass


# Função
def plotar_distribuição_massa_molar(MMsagregados, xsagregados, alfa, MWavg, informações_auxiliares):
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
    DMA_formatado, tipo_cálculo, tipo_regressão, algoritmo_otimização, _, nome_planilha = informações_auxiliares

    # Título
    plt.title(f"DIST. MASSA MOLAR - DMA(%): {DMA_formatado}", fontsize=16, fontweight="bold")

    # Série de dados
    plt.plot(MMsagregados, xsagregados, "o-", markersize=9, mfc="white", mec="black", color="black")

    # Legenda
    plt.legend([f"alfa = {alfa:.4f}\nMMavg = {MWavg:.2f} g/mol"], fontsize=12)

    # Títulos dos eixos, fontes das marcas de escala
    plt.xlabel("Massa molar (g/mol)", fontsize=14)
    plt.ylabel("Fração molar", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Linhas de grade
    plt.grid(color="k", linestyle="-", linewidth=0.1)

    # Nome do arquivo do gráfico a ser salvo
    if tipo_cálculo == 'predicao':
        nome_arquivo_gráfico = f"{nome_planilha}_DISTMASSAMOLAR.png"
    else:
        nome_arquivo_gráfico = f"{nome_planilha}_DISTMASSAMOLAR_tipo_regressao_{tipo_regressão}_" \
                               f"algoritmo_otimizacao_{algoritmo_otimização}.png"
    
    # Salvando o gráfico
    diretório_da_pasta_deste_modulo = os.path.dirname(os.path.abspath(__file__))
    if tipo_cálculo == 'predicao':
        diretório_png = os.path.join(diretório_da_pasta_deste_modulo, "Resultados",
                                     "Distribuição Massa Molar", "Predição", nome_arquivo_gráfico)
    else:
        diretório_png = os.path.join(diretório_da_pasta_deste_modulo, "Resultados",
                                     "Distribuição Massa Molar", "Regressão", nome_arquivo_gráfico)
    plt.savefig(diretório_png, dpi=300, bbox_inches="tight")

    # Fechando o arquivo após salvá-lo
    plt.close()

    pass


def plotar_yield_curves_cinéticas(ws_solvente, tempos, yields_eq, yields_cinéticas_exp, yields_cinéticas_calc,
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
    tipo_cálculo, nome_planilha = informações_auxiliares

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
        plt.plot(100 * ws_solvente, 100 * yields_cinéticas_exp[t, :],
                 "o", mfc=color, mec="black", label=f"{tempos[t]}h (exp)")
        plt.plot(100 * ws_solvente, 100 * yields_cinéticas_calc[t, :],
                 c=color, label=f"{tempos[t]}h (calc)", ls='-', lw='2')

    # Série de dados de equilíbrio
    plt.plot(100 * ws_solvente, 100 * yields_cinéticas_calc[0, :], c="red", ls='--', lw='2', label="equilíbrio")

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
    if tipo_cálculo == 'predicao':
        nome_arquivo_gráfico = f"{nome_planilha}_YIELDCURVE_KINECTS.png"
    else:
        nome_arquivo_gráfico = f"{nome_planilha}_YIELDCURVE_KINECTS_regressao.png"

    # Salvando o gráfico
    diretório_da_pasta_deste_modulo = os.path.dirname(os.path.abspath(__file__))
    if tipo_cálculo == 'predicao':
        diretório_png = os.path.join(diretório_da_pasta_deste_modulo, "Resultados",
                                     "Resultados_Cinética", "Predição", nome_arquivo_gráfico)
    else:
        diretório_png = os.path.join(diretório_da_pasta_deste_modulo, "Resultados",
                                     "Resultados_Cinética", "Regressão", nome_arquivo_gráfico)
    plt.savefig(diretório_png, dpi=300, bbox_inches="tight")

    # Fechando o arquivo após salvá-lo
    plt.close()

    pass
