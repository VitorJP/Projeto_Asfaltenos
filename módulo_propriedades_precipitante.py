import os
import numpy as np
import pandas as pd
from scipy.constants import R  # m3*Pa/mol*K


# Função
def calcular_propriedades_precipitante(T, precipitante, correlação_delta_precipitante='Akbarzadeh'):
    """ Calcula as propriedades do precipitante/alcano na temperatura de interesse.
    
    Inputs:
        T (float)         : temperatura (K)
        solvente (string) : nome do precipitante/alcano ("n-heptano" ou "n-pentano")
    
    Outputs:
        Uma tupla contendo os seguintes elementos:
           MM (float)    : massa molar (kg/mol)          
           rho (float)   : densidade (kg/m³) 
           delta (float) : parâmetro de solubilidade (Pa**0.5)
           V (float)     : volume molar (m³/mol) 

    Observações:
        Os parâmetros do modelo HBT para cada solvente foram extraídos do livro 'The Properties of Gases and Liquids',
        de Reid, Prausnitz e Poling (1987)
        As correlações para 'delta' foram originalmente propostas por Akbarzadeh et al. (2005),
        conforme citado por Tharanivasan (2012)
    """

    # Propriedades
    MM, Tc, wSRK, Vstar = obter_parâmetros_HBT(precipitante)
    rho = calcular_densidadehbt(T, MM, Tc, wSRK, Vstar)  # kg/m³
    V = (MM / rho) * 1e3  # cm³/mol

    # Cálculo do parâmetro de solubilidade
    match correlação_delta_precipitante:
        case 'Akbarzadeh':
            Delta_H_vap = 3492.8 + 276.54 * MM + 0.524 * (MM ** 2) if MM < 60e3 \
                else 103.65 + 368.7 * MM - 0.0603 * (MM ** 2)
            delta_25C = ((Delta_H_vap - 298.15*R) / V) ** 0.5
            delta = delta_25C - 0.0232 * (T - 298.15)  # MPa**0.5
        case 'Vargas':
            delta = 17.347 * rho + 2.904 if MM > 60e3 \
                else 2.904 + 26.302 * rho - 20.5618 * (rho ** 2) + 12.0425 * (rho ** 3)  # MPa**0.5
        case _:
            # Em caso de erro, usa-se Akbarzadeh como padrão.
            Delta_H_vap = 3492.8 + 276.54 * MM + 0.524 * (MM ** 2) if MM < 60e3 \
                else 103.65 + 368.7 * MM - 0.0603 * (MM ** 2)
            delta_25C = ((Delta_H_vap - 298.15 * R) / V) ** 0.5
            delta = delta_25C - 0.0232 * (T - 298.15)  # MPa**0.5

    # Ajuste de unidades
    MM = MM * 1e-3  # kg/mol
    delta = delta*1e3  # Pa**0.5
    V = MM / rho  # m³/mol

    return MM, rho, delta, V


# Função
def obter_parâmetros_HBT(nome_precipitante):

    # Database dos parâmetros HBT
    diretório_deste_módulo = os.path.dirname(__file__)
    caminho_excel = os.path.join(diretório_deste_módulo, 'Dados de Entrada', 'database_for_density_HBT.xlsx')
    df_HBT = pd.read_excel(caminho_excel, 'Data')

    # Verificação do precipitante indicado
    dados_precipitante = df_HBT[
        df_HBT.iloc[:, 0].astype(str).str.strip().str.lower() == nome_precipitante.strip().lower()]
    # OBS: Remover espaços e colocar todas as letras em minúsculas para minimizar erros de comparação

    if dados_precipitante.empty:
        raise ValueError(f'Precipitante ({nome_precipitante}) não encontrado.')
    else:
        dados_precipitante = dados_precipitante.iloc[0]

    # Leitura dos dados do precipitante
    MM = dados_precipitante.iloc[1]
    Tc = dados_precipitante.iloc[2]
    wSRK = dados_precipitante.iloc[3]
    Vstar = dados_precipitante.iloc[4]

    return MM, Tc, wSRK, Vstar


# Função 
def calcular_densidadehbt(T, MM, Tc, wSRK, Vstar):
    """ Calcula a densidade do solvente utilizando o modelo de Hankinson-Brobst-Thomson.
    
    Inputs:
        T (float)     : temperatura (K)
        MM (float)    : massa molar (g/mol)
        Tc (float)    : temperatura crítica (K)
        wSRK (float)  : fator acêntrico 'de SRK'
        Vstar (float) : parâmetro específico de cada substância (L/mol)

    Outputs:
        rho (float): Densidade (kg/m³) 

    Observações:
        A implementação foi baseada no equacionamento do livro 'The Properties of Gases and Liquids',
        de Reid, Prausnitz e Poling (1987)
    """
    
    # Variáveis auxiliares
    Tr = T / Tc
    aux = 1 - Tr
    MM = MM*1e-3  # kg/mol
    Vstar = Vstar*1e-3  # m³/mol

    # Cálculos
    a, b, c, d = -1.52816, 1.43907, -0.81446, 0.190454
    e, f, g, h = -0.296123, 0.386914, -0.0427258, -0.0480645
    Vr0 = 1 + a*aux**(1/3) + b*aux**(2/3) + c*aux + d*aux**(4/3)
    Vr1 = (e + f*Tr + g*Tr**2 + h*Tr**3)/(Tr - 1.00001)
    Vs = Vstar*(Vr0*(1 - wSRK*Vr1))
    rho_precipitante = MM/Vs

    return rho_precipitante


# ******************************************************************************************************************** #
#  ATENÇÃO: O CÓDIGO A SEGUIR SERÁ EXECUTADO APENAS QUANDO ESTE MÓDULO FOR RODADO COMO SCRIPT PRINCIPAL.               #
#           O CÓDIGO A SEGUIR SERVE PARA CONFERIR SE AS FUNÇÕES DESTE MÓDULO FUNCIONAM CORRETAMENTE.                   #
# ******************************************************************************************************************** #
# INÍCIO DO TESTE
# OBS: GABARITO EXTRAÍDO DA PG. 68 DO LIVRO 'THE PROPERTIES OF GASES AND LIQUIDS', DE REID, PRAUSNITZ E POLING (1987)
if __name__ == "__main__":

    # Função 'calcular_densidadehbt'
    substancia_teste = 'isobutano'
    MM_teste, Tc_teste, wSRK_teste, Vstar_teste = obter_parâmetros_HBT('isobutano')
    rho_teste = calcular_densidadehbt(310.93, MM_teste, Tc_teste, wSRK_teste, Vstar_teste)
    V_teste = 58.12*1e-3/rho_teste
    print("\n|---------------------------------------------------------------------------------------------------------"
          "-------------------------|")
    print("| TESTE DA FUNCAO 'calcular_densidadehbt'")
    print(f"| Volume molar calculado: {V_teste*1e6:.2f} cm3/mol")
    print(f"| Volume molar gabarito: {108.9} cm3/mol")
    print("|-----------------------------------------------------------------------------------------------------------"
          "-----------------------|")

    # Função 'calcular_propriedades_precipitante'
    MM_teste, rho_teste, delta_teste, V_teste = calcular_propriedades_precipitante(298.15, 'n-heptano')
    print("\n|---------------------------------------------------------------------------------------------------------"
          "-------------------------|")
    print("| TESTE DA FUNCAO 'calcular_propriedades_precipitante'")
    print(f"| delta calculado: {delta_teste * 1e-3:.2f} MPa ** 0.5")
    print(f"| delta gabarito: {15.18} MPa ** 0.5")
    print("|-----------------------------------------------------------------------------------------------------------"
          "-----------------------|")

# FIM DO TESTE
# ******************************************************************************************************************** #
