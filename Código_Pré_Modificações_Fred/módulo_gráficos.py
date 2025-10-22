# Importação de bibliotecas do python
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os


# Função
def plotar_yield_curves(ws_solvente, yields_exp, yields_calc, informações_auxiliares):
    """ Cria um gráfico contendo as curvas de solubilidade experimental e calculada.
    
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
    DMA_formatado, tipo_cálculo_programa, tipo_regressão, algoritmo_otimização, nome_planilha = informações_auxiliares

    # Título
    plt.title(f"YIELD CURVE - DMA(%): {DMA_formatado}", fontsize=16, fontweight="bold")

    # Série de dados experimentais
    # Obs: se todos os yields experimentais são nulos (ex: 'nome_planilha' = 'Yanes_P3'),
    #      a curva experimental é plotada na cor branca (desaparece)
    if all(yield_exp == 0 for yield_exp in yields_exp):
        plt.plot(100*ws_solvente, 100*yields_exp, "o", mfc="white", mec="white", markersize=10)
    if any(yield_exp != 0 for yield_exp in yields_exp):
        plt.plot(100*ws_solvente, 100*yields_exp, "o", mfc="blue", mec="black", markersize=10)
    
    # Série de dados calculada
    plt.plot(100*ws_solvente, 100*yields_calc, "o", mfc="red", mec="black", markersize=10)

    # Legenda
    plt.legend(["experimental", "calculado"], fontsize=12, loc="upper left")
    
    # Títulos dos eixos, valores min e max de cada eixo, fontes das marcas de escala, marcas de escala secundárias
    plt.xlabel("fração de solvente, wt%", fontsize=14)
    plt.ylabel("yield de asfalteno, wt%", fontsize=14)
    plt.axis(xmin=40, xmax=100, ymin=0)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.gca().xaxis.set_minor_locator(MultipleLocator(5))  # Marcas de escala secundárias no eixo x
    plt.gca().yaxis.set_minor_locator(MultipleLocator(0.5))  # Marcas de escala secundárias no eixo y

    # Linhas de grade
    plt.grid(color="k", linestyle="-", linewidth=0.1)

    # Nome do arquivo do gráfico a ser salvo
    if tipo_cálculo_programa == 'predicao':
        nome_arquivo_gráfico = f"{nome_planilha}_YIELDCURVE.png"
    else:
        nome_arquivo_gráfico = f"{nome_planilha}_YIELDCURVE_tipo_regressao_{tipo_regressão}_" \
                               f"algoritmo_otimizacao_{algoritmo_otimização}.png"
    
    # Salvando o gráfico
    diretório_da_pasta_deste_modulo = os.path.dirname(os.path.abspath(__file__))
    if tipo_cálculo_programa == 'predicao':
        diretório_png = os.path.join(diretório_da_pasta_deste_modulo, "Resultados", "Predição", nome_arquivo_gráfico)
    else:
        diretório_png = os.path.join(diretório_da_pasta_deste_modulo, "Resultados", "Regressão", nome_arquivo_gráfico)
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
    DMA_formatado, tipo_cálculo_programa, tipo_regressão, algoritmo_otimização, nome_planilha = informações_auxiliares

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
    if tipo_cálculo_programa == 'predicao':
        nome_arquivo_gráfico = f"{nome_planilha}_DISTMASSAMOLAR.png"
    else:
        nome_arquivo_gráfico = f"{nome_planilha}_DISTMASSAMOLAR_tipo_regressao_{tipo_regressão}_" \
                               f"algoritmo_otimizacao_{algoritmo_otimização}.png"
    
    # Salvando o gráfico
    diretório_da_pasta_deste_modulo = os.path.dirname(os.path.abspath(__file__))
    if tipo_cálculo_programa == 'predicao':
        diretório_png = os.path.join(diretório_da_pasta_deste_modulo, "Resultados", "Predição", nome_arquivo_gráfico)
    else:
        diretório_png = os.path.join(diretório_da_pasta_deste_modulo, "Resultados", "Regressão", nome_arquivo_gráfico)
    plt.savefig(diretório_png, dpi=300, bbox_inches="tight")

    # Fechando o arquivo após salvá-lo
    plt.close()

    pass
