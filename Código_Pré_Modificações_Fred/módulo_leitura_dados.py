# Importação de bibliotecas do python
import numpy as np
import pandas as pd


# Função
def ler_variáveis_entrada_código(diretório):
    """ Lê o arquivo 'variáveis_entrada_código.txt'.
    
    Inputs:
        diretório (string): diretório do arquivo
    
    Outputs:
        Uma tupla contendo os seguintes elementos:
            n_agregados (int)                        : nº de agregados de asfaltenos                   
            MWmin (float)                            : massa molar do monômero (g/mol)
            MWmax (float)                            : massa molar máxima possível para um agregado de asfalteno (g/mol)
            alfa (float)                             : parâmetro de forma da função densidade de probabilidade da
                                                       distribuição Gamma (FDP_Gamma)
            MWavg (float)                            : massa molar média dos agregados de asfaltenos (g/mol)
            tipo_cálculo_MM_agregados (string)       : tipo de cálculo para a determinação das massas molares
                                                       dos agregados de asfaltenos
            método_integração_FDP_Gamma (string)     : método numérico para as integrações numéricas envolvendo a
                                                       FDP_Gamma

            correlação_densidade_saturados (string)  : correlação para o cálculo da densidade dos saturados
            correlação_delta_saturados (string)      : correlação para o cálculo dos parâmetros de solubilidade dos
                                                       saturados

            correlação_densidade_aromáticos (string) : correlação para o cálculo da densidade dos aromáticos
            correlação_delta_aromáticos (string)     : correlação para o cálculo dos parâmetros de solubilidade dos
                                                       aromáticos

            correlação_densidade_resinas (string)    : correlação para o cálculo da densidade das resinas
            correlação_delta_resinas (string)        : correlação para o cálculo dos parâmetros de solubilidade das
                                                       resinas

            correlação_densidade_agregados (string)  : correlação para o cálculo da densidade dos
                                                       agregados de asfaltenos
            correlação_delta_agregados (string)      : correlação para o cálculo dos parâmetros de solubilidade dos
                                                       agregados de asfaltenos

            Alinha_delta_agregados (float)           : valor do parâmetro A' sugerido por Barrera para o cálculo dos
                                                       parâmetros de solubilidade dos agregados de asfaltenos
            c_delta_agregados (float)                : valor do parâmetro c sugerido por Barrera para o cálculo dos
                                                       parâmetros de solubilidade dos agregados de asfaltenos
            d_delta_agregados (float)                : valor do parâmetro d sugerido por Barrera para o cálculo dos
                                                       parâmetros de solubilidade dos agregados de asfaltenos

            tipo_cálculo_programa (string)           : tipo de cálculo a ser executado pelo programa principal
            tipo_regressão (string)                  : define quais parâmetros serão regredidos pelo programa principal
            algoritmo_otimização (int)               : algoritmo numérico de regressão dos parâmetros
            nome_planilha (string)                   : título da planilha que contém os dados experimentais a serem
                                                       preditos ou regredidos
            
    Observações:
        Maiores informações sobre as variáveis supracitadas estão no arquivo 'variáveis_entrada_código.txt'
    """

    # Abertura do arquivo e leitura de todas as linhas
    with open(diretório) as arquivo:
        linhas = arquivo.readlines()

    # Armazenamento apenas das linhas que contém os valores das variáveis a serem lidas pelo programa principal
    linhas_úteis = linhas[-23:-1]

    # Removendo os espaços em branco dos elementos de 'linhas_úteis'
    linhas_úteis_limpas = [linha.strip() for linha in linhas_úteis]

    # Alocação de variáveis  
    n_agregados = int(linhas_úteis_limpas[0])
    MWmin = float(linhas_úteis_limpas[1])
    MWmax = float(linhas_úteis_limpas[2])
    alfa = float(linhas_úteis_limpas[3])
    MWavg = float(linhas_úteis_limpas[4])
    tipo_cálculo_MM_agregados = linhas_úteis_limpas[5]
    método_integração_FDP_Gamma = linhas_úteis_limpas[6]
    correlação_densidade_saturados = linhas_úteis_limpas[7]
    correlação_delta_saturados = linhas_úteis_limpas[8]
    correlação_densidade_aromáticos = linhas_úteis_limpas[9]
    correlação_delta_aromáticos = linhas_úteis_limpas[10]
    correlação_densidade_resinas = linhas_úteis_limpas[11]
    correlação_delta_resinas = linhas_úteis_limpas[12]
    correlação_densidade_agregados = linhas_úteis_limpas[13]
    correlação_delta_agregados = linhas_úteis_limpas[14]
    Alinha_delta_agregados = float(linhas_úteis_limpas[15])
    c_delta_agregados = float(linhas_úteis_limpas[16])
    d_delta_agregados = float(linhas_úteis_limpas[17])
    tipo_cálculo_programa = linhas_úteis_limpas[18]
    tipo_regressão = int(linhas_úteis_limpas[19])
    algoritmo_otimização = int(linhas_úteis_limpas[20])
    nome_planilha = linhas_úteis_limpas[21]

    return (
        n_agregados, MWmin, MWmax, alfa, MWavg, tipo_cálculo_MM_agregados, método_integração_FDP_Gamma, 
        correlação_densidade_saturados, correlação_delta_saturados,
        correlação_densidade_aromáticos, correlação_delta_aromáticos,
        correlação_densidade_resinas, correlação_delta_resinas,
        correlação_densidade_agregados, correlação_delta_agregados, 
        Alinha_delta_agregados, c_delta_agregados, d_delta_agregados,
        tipo_cálculo_programa, tipo_regressão,
        algoritmo_otimização,
        nome_planilha
        )


# Função
def ler_dados_experimentais(diretório, nome_planilha):
    """ Lê o arquivo 'dados_experimentais.xlsx'.
    
    Inputs:
        diretório (string)     : diretório do arquivo
        nome_planilha (string) : nome da planilha que contém o sistema de interesse
    
    Outputs:
        Uma tupla contendo os seguintes elementos:
            SARA (array)             : composição SARA do petróleo (base mássica)         
            T (float)                : temperatura (K)       
            solvente (string)        : nome do solvente ("n-heptano" ou "n-pentano")                   
            ws_simplificados (array) : composição global do sistema em termos de [Solvente, Petróleo] (base mássica)            
            yields_exp (array)       : yields fracionais de asfaltenos (experimentais)
    """

    # Leitura do DataFrame
    df = pd.read_excel(diretório, nome_planilha)

    # Composição SARA
    SARA = df.iloc[0:4, 1].to_numpy()
    SARA = SARA*1e-2

    # Temperatura
    T = df.iloc[4, 1]

    # Solvente
    solvente = df.iloc[5, 1]

    # Composição do sistema
    frações_solvente = df.iloc[7:, 0].to_numpy()
    frações_petróleo = 1 - frações_solvente
    ws_simplificados = np.column_stack((frações_solvente, frações_petróleo)) 

    # Yields
    yields_exp = df.iloc[7:, 1].to_numpy()
    
    return SARA, T, solvente, ws_simplificados, yields_exp


# ******************************************************************************************************************** #
#  ATENÇÃO: O CÓDIGO A SEGUIR SERÁ EXECUTADO APENAS QUANDO ESTE MÓDULO FOR RODADO COMO SCRIPT PRINCIPAL.               #
#           O CÓDIGO A SEGUIR SERVE PARA CONFERIR SE AS FUNÇÕES DESTE MÓDULO FUNCIONAM CORRETAMENTE.                   #
# ******************************************************************************************************************** #
# INÍCIO DO TESTE
if __name__ == "__main__":

    # Função 'ler_variáveis_entrada_codigo'
    diretório = "variáveis_entrada_código.txt"
    saída_da_função = ler_variáveis_entrada_código(diretório)
    print("\n|---------------------------------------------------------------------------------------------------------"
          "---------------------------------------------------|")
    print("TESTE DA FUNCAO 'ler_variáveis_entrada_codigo'")
    print(f"n_agregados, MMin, MMmax, alfa, MMavg, tipo_calculo_MM_agregados, metodo_integracao_FDP_Gamma: "
          f"{saída_da_função[0:7]}")
    print(f"correlacao_densidade_saturados, correlacao_delta_saturados: {saída_da_função[7:9]}")
    print(f"correlacao_densidade_aromaticos, correlacao_delta_aromaticos: {saída_da_função[9:11]}")
    print(f"correlacao_densidade_resinas, correlacao_delta_resinas: {saída_da_função[11:13]}")
    print(f"correlacao_densidade_agregados, correlacao_delta_agregados: {saída_da_função[13:15]}")
    print(f"Alinha_delta_agregados, c_delta_agregados, d_delta_agregados{saída_da_função[15:18]}")
    print(f"tipo_calculo_programa, tipo_regressao: {saída_da_função[18:20]}")
    print(f"algoritmo_otimizacao: {saída_da_função[20]}")
    print(f"nome_planilha: {saída_da_função[21]}")
    print("|-----------------------------------------------------------------------------------------------------------"
          "-------------------------------------------------|")

    # Função 'ler_dados_experimentais'
    diretório = "dados_experimentais.xlsx"
    nome_planilha = 'Yanes_P1'
    saída_da_função = ler_dados_experimentais(diretório, nome_planilha)
    print("TESTE DA FUNCAO 'ler_dados_experimentais'")
    print(f"SARA: {saída_da_função[0]}")
    print(f"T: {saída_da_função[1]}")
    print(f"solvente: {saída_da_função[2]}")
    print(f"ws_simplificados: {saída_da_função[3]}")
    print(f"yields_exp: {saída_da_função[4]}")
    print("|-----------------------------------------------------------------------------------------------------------"
          "-------------------------------------------------|")
# FIM DO TESTE
# ******************************************************************************************************************** #
