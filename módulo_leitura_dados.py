# Importação de bibliotecas do python
import os
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
    linhas_úteis = linhas[-29:-1]

    # Removendo o nome da varíavel da linha
    linhas_úteis_valores = [linha.split(":", 1)[1] if ":" in linha else linha for linha in linhas_úteis]

    # Removendo os espaços em branco dos elementos de 'linhas_úteis'
    linhas_úteis_limpas = [linha.strip() for linha in linhas_úteis_valores]

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
    kt1_cinético = float(linhas_úteis_limpas[18])
    kt2_cinético = float(linhas_úteis_limpas[19])
    kw1_cinético = float(linhas_úteis_limpas[20])
    kw2_cinético = float(linhas_úteis_limpas[21])
    tipo_cálculo_equilíbrio = linhas_úteis_limpas[22]
    tipo_cálculo_cinética = linhas_úteis_limpas[23]
    x_yield_curve = linhas_úteis_limpas[24]
    tipo_regressão_equilibrio = int(linhas_úteis_limpas[25])
    algoritmo_otimização = int(linhas_úteis_limpas[26])
    nome_planilha = linhas_úteis_limpas[27]

    return (
        n_agregados, MWmin, MWmax, alfa, MWavg, tipo_cálculo_MM_agregados, método_integração_FDP_Gamma, 
        correlação_densidade_saturados, correlação_delta_saturados,
        correlação_densidade_aromáticos, correlação_delta_aromáticos,
        correlação_densidade_resinas, correlação_delta_resinas,
        correlação_densidade_agregados, correlação_delta_agregados, 
        Alinha_delta_agregados, c_delta_agregados, d_delta_agregados,
        kt1_cinético, kt2_cinético, kw1_cinético, kw2_cinético,
        tipo_cálculo_equilíbrio, tipo_cálculo_cinética, x_yield_curve,
        tipo_regressão_equilibrio, algoritmo_otimização, nome_planilha
        )


# Função
def ler_dados_experimentais(diretório, planilha):
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
    df = pd.read_excel(diretório, planilha)

    # Composição SARA
    SARA = df.iloc[3:7, 1].to_numpy()
    SARA = SARA*1e-2

    # Temperatura
    T = df.iloc[7, 1] + 273.15

    # Solvente
    solvente = df.iloc[8, 1]

    # Dados do sistema
    dados_exp = pd.DataFrame({
        'w_solvente': pd.to_numeric(df.iloc[2:, 3], errors='coerce'),
        'yield': pd.to_numeric(df.iloc[2:, 4], errors='coerce')
    })

    # Encontrar a última linha que possui algum dado
    mascara = dados_exp.notna().any(axis=1)
    ultima_linha = mascara[mascara].index[-1]

    # Cortar apenas o final vazio
    dados_exp = dados_exp.loc[:ultima_linha]

    frações_solvente = dados_exp['w_solvente'].to_numpy()
    yields_exp = dados_exp['yield'].to_numpy()

    frações_petróleo = 1 - frações_solvente
    ws_simplificados = np.column_stack((frações_solvente, frações_petróleo))
    
    return SARA, T, solvente, ws_simplificados, yields_exp


def ler_dados_cinéticos(diretório, planilha):

    # Leitura do DataFrame
    df = pd.read_excel(diretório, planilha, header=None)

    tempos = []
    loc_t = 4  # Primeira coluna com dados de yield

    # Preencher o vetor de tempos até encontrar uma célula vazia no excel dos dados
    while True:
        t = df.iloc[1, loc_t]

        if pd.isna(t):
            break

        tempos.append(t)
        loc_t += 1

    tempos = np.array(tempos)

    # Dados de yield curves em múltiplos tempos
    dados = (df.iloc[3:, 4:loc_t].apply(pd.to_numeric, errors='coerce'))

    ultima_linha = dados.notna().any(axis=1)
    dados = dados.loc[:ultima_linha[ultima_linha].index[-1]]
    yields_temporais = dados.to_numpy().T

    return tempos, yields_temporais


# ******************************************************************************************************************** #
#  ATENÇÃO: O CÓDIGO A SEGUIR SERÁ EXECUTADO APENAS QUANDO ESTE MÓDULO FOR RODADO COMO SCRIPT PRINCIPAL.               #
#           O CÓDIGO A SEGUIR SERVE PARA CONFERIR SE AS FUNÇÕES DESTE MÓDULO FUNCIONAM CORRETAMENTE.                   #
# ******************************************************************************************************************** #
# INÍCIO DO TESTE
if __name__ == "__main__":

    diretório_deste_módulo = os.path.dirname(__file__)

    # Função 'ler_variáveis_entrada_codigo'
    diretório_do_txt = os.path.join(diretório_deste_módulo, 'Dados de Entrada', 'variáveis_entrada_código.txt')
    saída_da_função = ler_variáveis_entrada_código(diretório_do_txt)
    variáveis_entrada = ["n_agregados", "MWmin", "MWmax", "alfa", "MWavg",
                         "tipo_cálculo_MM_agregados", "método_integração_FDP_Gamma",
                         "correlação_densidade_saturados", "correlação_delta_saturados",
                         "correlação_densidade_aromáticos", "correlação_delta_aromáticos",
                         "correlação_densidade_resinas", "correlação_delta_resinas",
                         "correlação_densidade_agregados", "correlação_delta_agregados",
                         "Alinha_delta_agregados", "c_delta_agregados", "d_delta_agregados",
                         "kt1_cinético", "kt2_cinético", "kw1_cinético", "kw2_cinético",
                         "tipo_cálculo_equilíbrio", "tipo_cálculo_cinética", "tipo_regressão",
                         "algoritmo_otimização", "nome_planilha"]
    print("\n|---------------------------------------------------------------------------------------------------------"
          "---------------------------------------------------|")
    print("TESTE DA FUNCAO 'ler_variáveis_entrada_codigo'")
    for i in range(len(variáveis_entrada)):
        print(f"{variáveis_entrada[i]}: {saída_da_função[i]}")
    print("|-----------------------------------------------------------------------------------------------------------"
          "-------------------------------------------------|")

    # Função 'ler_dados_experimentais'
    diretório_do_xlsx = os.path.join(diretório_deste_módulo, 'Dados de Entrada', 'dados_experimentais_codigo.xlsx')
    nome_planilha = 'Yanes_P1'
    saída_da_função = ler_dados_experimentais(diretório_do_xlsx, nome_planilha)
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
