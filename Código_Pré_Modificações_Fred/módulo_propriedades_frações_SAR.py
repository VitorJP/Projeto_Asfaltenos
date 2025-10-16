# Função
def calcular_propriedades_saturados(T, correlação_densidade_saturados, correlação_delta_saturados):
    """ Calcula as propriedades dos saturados na temperatura de interesse.
    
    Inputs:
        T (float)                               : temperatura (K)
        correlação_densidade_saturados (string) : correlação para o cálculo da densidade dos saturados
        correlação_delta_saturados (string)     : correlação para o cálculo dos parâmetros de solubilidade dos saturados
    
    Outputs:
        Uma tupla contendo os seguintes elementos:
           MM (float)    : massa molar (kg/mol)          
           rho (float)   : densidade (kg/m³) 
           delta (float) : parâmetro de solubilidade (Pa**0.5)
           V (float)     : volume molar (m³/mol) 
    """
    
    # Massa molar (g/mol)
    MM = 460

    # Densidade (kg/m³)
    match correlação_densidade_saturados:
        case "Caiua":
            rho = 1069.54 - 0.6379*T

        case "Akbarzadeh":
            rho = 1078.96 - 0.6379*T 

        case "Yanes":
            rho = 880.0

    # Parâmetro de solubilidade (MPa**0.5)
    match correlação_delta_saturados:
        case "Akbarzadeh":
            delta = 22.381 - 0.0222*T
            
        case "Tharanivasan":
            delta = 23.021 - 0.0222*T

        case "Yanes":
            delta = 16.4

    # Ajuste de unidades
    MM = MM*1e-3 # kg/mol
    delta = delta*1e3 # Pa**0.5

    # Volume molar (m³/mol)
    V = MM/rho 

    return MM, rho, delta, V

# Função
def calcular_propriedades_aromáticos(T, correlação_densidade_aromáticos, correlação_delta_aromáticos):
    """ Calcula as propriedades dos aromáticos na temperatura de interesse.
    
    Inputs:
        T (float)                                : temperatura (K)
        correlação_densidade_aromáticos (string) : correlação para o cálculo da densidade dos aromáticos
        correlação_delta_aromáticos (string)     : correlação para o cálculo dos parâmetros de solubilidade dos aromáticos
    
    Outputs:
        Uma tupla contendo os seguintes elementos:
           MM (float)    : massa molar (kg/mol)          
           rho (float)   : densidade (kg/m³) 
           delta (float) : parâmetro de solubilidade (Pa**0.5)
           V (float)     : volume molar (m³/mol) 
    """

    # Massa molar (g/mol)
    MM = 522

    # Densidade (kg/m³)
    match correlação_densidade_aromáticos:
        case "Caiua":
            rho = 1164.73 - 0.5942*T

        case "Akbarzadeh":
            rho = 1184.47 - 0.5942*T

        case "Yanes":
            rho = 990.0

    # Parâmetro de solubilidade (MPa**0.5)
    match correlação_delta_aromáticos:
        case "Akbarzadeh":
            delta = 26.333 - 0.0204*T

        case "Yanes":
            delta = 20.3

    # Ajuste de unidades
    MM = MM*1e-3 # kg/mol
    delta = delta*1e3 # Pa**0.5

    # Volume molar (m³/mol)
    V = MM/rho 

    return MM, rho, delta, V

# Função
def calcular_propriedades_resinas(T, correlação_densidade_resinas, correlação_delta_resinas):
    """ Calcula as propriedades das resinas na temperatura de interesse.
    
    Inputs:
        T (float)                             : temperatura (K)
        correlação_densidade_resinas (string) : correlação para o cálculo da densidade das resinas
        correlação_delta_resinas (string)     : correlação para o cálculo dos parâmetros de solubilidade das resinas
    
    Outputs:
        Uma tupla contendo os seguintes elementos:
           MM (float)    : massa molar (kg/mol)          
           rho (float)   : densidade (kg/m³) 
           delta (float) : parâmetro de solubilidade (Pa**0.5)
           V (float)     : volume molar (m³/mol) 
    """

    # Massa molar (g/mol)
    MM = 1040

    # Densidade (kg/m³)
    match correlação_densidade_resinas:
        case "Yanes":
            rho = 1044.0

    # Parâmetro de solubilidade (MPa**0.5)
    match correlação_delta_resinas:
        case "Yanes":
            delta = 19.3

    # Ajuste de unidades
    MM = MM*1e-3 # kg/mol
    delta = delta*1e3 # Pa**0.5

    # Volume molar (m³/mol)
    V = MM/rho 

    return MM, rho, delta, V