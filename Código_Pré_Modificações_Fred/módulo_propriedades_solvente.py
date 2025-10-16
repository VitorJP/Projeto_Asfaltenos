# Função 
def calcular_propriedades_solvente(T, solvente):
    """ Calcula as propriedades do solvente na temperatura de interesse.
    
    Inputs:
        T (float)         : temperatura (K)
        solvente (string) : nome do solvente ("n-heptano" ou "n-pentano")  
    
    Outputs:
        Uma tupla contendo os seguintes elementos:
           MM (float)    : massa molar (kg/mol)          
           rho (float)   : densidade (kg/m³) 
           delta (float) : parâmetro de solubilidade (Pa**0.5)
           V (float)     : volume molar (m³/mol) 

    Observações:
        Os parâmetros do modelo HBT para cada solvente foram extraídos do livro 'The Properties of Gases and Liquids', de Reid, Prausnitz e Poling (1987)
        As correlações para 'delta' foram originalmente propostas por Akbarzadeh et al. (2005), conforme citado por Tharanivasan (2012)
    """    
    
    # Propriedades
    match solvente:
        case "n-heptano":
            MM = 100 # g/mol
            rho = calcular_densidadehbt(T, MM, 540.26, 0.3507, 0.4304) # kg/m³
            delta = 15.2 - 0.0232*(T - 298.15) # MPa**0.5 
    
        case "n-pentano":
            MM = 72 # g/mol
            rho = calcular_densidadehbt(T, MM, 469.65, 0.2522, 0.3113) # kg/m³
            delta = 14.3 - 0.0232*(T - 298.15) # MPa**0.5 
    
    # Ajuste de unidades
    MM = MM*1e-3 # kg/mol
    delta = delta*1e3 # Pa**0.5
    V = MM/rho # m³/mol      

    return MM, rho, delta, V

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
        A implementação foi baseada no equacionamento do livro 'The Properties of Gases and Liquids', de Reid, Prausnitz e Poling (1987)
    """
    
    # Variáveis auxiliares
    Tr = T / Tc
    aux = 1 - Tr
    MM = MM*1e-3 # kg/mol
    Vstar = Vstar*1e-3 # m³/mol

    # Cálculos
    a, b = -1.52816, 1.43907
    c, d = -0.81446, 0.190454
    e, f = -0.296123, 0.386914
    g, h = -0.0427258, -0.0480645
    Vr0 = 1 + a*aux**(1/3) + b*aux**(2/3) + c*aux + d*aux**(4/3)
    Vr1 = (e + f*Tr + g*Tr**2 + h*Tr**3)/(Tr - 1.00001)
    Vs = Vstar*(Vr0*(1 - wSRK*Vr1))
    rho = MM/Vs

    return rho

# ********************************************************************************************************************************************************************************* #
#  ATENÇÃO: O CÓDIGO A SEGUIR SERÁ EXECUTADO APENAS QUANDO ESTE MÓDULO FOR RODADO COMO SCRIPT PRINCIPAL.                                                                            #
#           O CÓDIGO A SEGUIR SERVE PARA CONFERIR SE AS FUNÇÕES DESTE MÓDULO FUNCIONAM CORRETAMENTE.                                                                                #
# ********************************************************************************************************************************************************************************* #
# INÍCIO DO TESTE
# OBS: GABARITO EXTRAÍDO DA PG. 68 DO LIVRO 'THE PROPERTIES OF GASES AND LIQUIDS', DE REID, PRAUSNITZ E POLING (1987)
if __name__ == "__main__":

    # Função 'calcular_densidadehbt'
    rho = calcular_densidadehbt(310.93, 58.12, 408.14, 0.1825, 0.2568)
    V = 58.12*1e-3/rho
    print("\n|----------------------------------------------------------------------------------------------------------------------------------|")
    print("| TESTE DA FUNCAO 'calcular_densidadehbt'")
    print(f"| Volume molar calculado: {V*1e6} cm3/mol")
    print(f"| Volume molar gabarito: {108.9} cm3/mol")
    print("|----------------------------------------------------------------------------------------------------------------------------------|")
# FIM DO TESTE
# ********************************************************************************************************************************************************************************* #