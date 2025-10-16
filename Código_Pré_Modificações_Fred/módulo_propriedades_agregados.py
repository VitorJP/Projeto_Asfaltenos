# Importação de bibliotecas do python
import numpy as np

# Função
def calcular_propriedades_agregados(T, MMsagregados, correlação_densidade_agregados, correlação_delta_agregados, 
                                    Alinha_delta_agregados, c_delta_agregados, d_delta_agregados):
    """ Calcula as densidades, parâmetros de solubilidade e volumes molares dos agregados de asfaltenos na temperatura de interesse.
    
    Inputs:
        T (float)                               : temperatura (K)
        MMsagregados (array)                    : massas molares dos agregados de asfaltenos (g/mol)
        correlação_densidade_agregados (string) : correlação para o cálculo da densidade dos agregados de asfaltenos 
        correlação_delta_agregados (string)     : correlação para o cálculo dos parâmetros de solubilidade dos agregados de asfaltenos
        Alinha_delta_agregados (float)          : valor do parâmetro A' sugerido por Barrera para o cálculo dos parâmetros de solubilidade dos agregados de asfaltenos
        c_delta_agregados (float)               : valor do parâmetro c sugerido por Barrera para o cálculo dos parâmetros de solubilidade dos agregados de asfaltenos
        d_delta_agregados (float)               : valor do parâmetro d sugerido por Barrera para o cálculo dos parâmetros de solubilidade dos agregados de asfaltenos

    Outputs:
        Uma tupla contendo os seguintes elementos:  
           rhosagregados (array)   : densidades (kg/m³) 
           deltasagregados (array) : parâmetros de solubilidade (Pa**0.5)
           Vsagregados (array)     : volumes molares (m³/mol) 
    """

    # Densidade (kg/m³)
    match correlação_densidade_agregados:
        case "Alboudwarej":
            rhosagregados = (MMsagregados/(1.493*MMsagregados**0.936))*1e3 # kg/m³
        
        case "Barrera":
            rhosagregados = 1100 + 100*(1 - np.exp(-MMsagregados/3850)) # kg/m³ 

    # Parâmetro de solubilidade (MPa**0.5)
    match correlação_delta_agregados:
        case "Barrera":
            A = 0.579 - 0.00075*T + Alinha_delta_agregados
            deltasagregados = np.sqrt(A*rhosagregados*c_delta_agregados*MMsagregados**d_delta_agregados)

        case "Tharanivasan":
            A = 0.579 - 0.00075*T
            deltasagregados = np.sqrt(A*rhosagregados) # MPa**0.5

    # Ajuste de unidades
    MMsagregados = MMsagregados*1e-3 # kg/mol
    deltasagregados = deltasagregados*1e3 # Pa**0.5

    # Volumes molares (m³/mol)
    Vsagregados = MMsagregados/rhosagregados 

    return rhosagregados, deltasagregados, Vsagregados