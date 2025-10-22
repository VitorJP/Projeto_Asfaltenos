# Importação de bibliotecas do python
import numpy as np
import scipy as scp


# Função
def gerar_distribuição_massa_molar(alfa, MWavg, n_agregados, MWmin, MWmax, tipo_cálculo_MM_agregados,
                                   método_integração_FDP_Gamma):
    """ Calcula as massas molares, frações mássicas e frações molares dos agregados de asfaltenos.
       
    Inputs:
        alfa (float)                         : parâmetro de forma da função densidade de probabilidade da
                                               distribuição Gamma (FDP_Gamma)
        MWavg (float)                        : massa molar média dos agregados de asfaltenos (g/mol)
        n_agregados (int)                    : nº de agregados de asfaltenos   
        MWmin (float)                        : limite inferior global das faixas de massa molar (g/mol)
        MWmax (float)                        : limite superior global das faixas de massa molar (g/mol)
        tipo_cálculo_MM_agregados (string)   : tipo de cálculo para a determinação das massas molares dos
                                               agregados de asfaltenos
        método_integração_FDP_Gamma (string) : método numérico para as integrações numéricas envolvendo a FDP_Gamma

    Outputs:
        Uma tupla contendo os seguintes elementos:
           MMsagregados (array) : massas molares dos agregados de asfaltenos (g/mol)  
           wsagregados (array)  : frações mássicas dos agregados de asfaltenos
           xsagregados (array)  : frações molares dos agregados de asfaltenos
    """

    # Função FDP_Gamma
    MWmon = MWmin
    beta = (MWavg - MWmon) / alfa

    def f(MWi):
        return (MWi - MWmon) ** (alfa - 1) / (beta ** alfa * scp.special.gamma(alfa)) * np.exp(-(MWi - MWmon) / beta)

    # Limites das faixas de massa molar
    n_pontos = n_agregados + 1
    MM_limites_faixas = np.linspace(MWmin, MWmax, n_pontos)

    # Massas molares dos agregados
    MMsagregados = np.zeros(n_agregados)
    match tipo_cálculo_MM_agregados:
        case "medio":
            MWf = lambda MWi: MWi * f(MWi)  # Criando a função MWi*f(MWi)
            for i in range(n_agregados):
                numerador = scp.integrate.quad(MWf, MM_limites_faixas[i], MM_limites_faixas[i + 1])[0]
                denominador = scp.integrate.quad(f, MM_limites_faixas[i], MM_limites_faixas[i + 1])[0]
                MMsagregados[i] = numerador / denominador  # g/mol
        case "superior":
            for i in range(n_agregados):
                MMsagregados[i] = MM_limites_faixas[i + 1]  # g/mol

    # Frações molares dos agregados
    match método_integração_FDP_Gamma:
        case "quadratura":
            xsagregados = np.zeros(n_agregados)
            denominador = scp.integrate.quad(f, MM_limites_faixas[0], MM_limites_faixas[-1])[0]
            for i in range(n_agregados):
                numerador = scp.integrate.quad(f, MM_limites_faixas[i], MM_limites_faixas[i + 1])[0]
                xsagregados[i] = numerador / denominador
        case "trapezios":
            xsagregados = np.zeros(n_agregados)
            MWs_denominador = MM_limites_faixas.copy()
            fMWs_denominador = f(MWs_denominador)
            denominador = np.trapezoid(fMWs_denominador, MWs_denominador)
            for i in range(n_agregados):
                MWs_numerador = MM_limites_faixas[i:i + 2]
                fMWs_numerador = f(MWs_numerador)
                numerador = np.trapezoid(fMWs_numerador, MWs_numerador)
                xsagregados[i] = numerador / denominador
        case _:  # Em caso de erro, usa-se 'trapezios' como padrão
            xsagregados = np.zeros(n_agregados)
            MWs_denominador = MM_limites_faixas.copy()
            fMWs_denominador = f(MWs_denominador)
            denominador = np.trapezoid(fMWs_denominador, MWs_denominador)
            for i in range(n_agregados):
                MWs_numerador = MM_limites_faixas[i:i + 2]
                fMWs_numerador = f(MWs_numerador)
                numerador = np.trapezoid(fMWs_numerador, MWs_numerador)
                xsagregados[i] = numerador / denominador

    # Frações mássicas dos agregados
    wsagregados = xsagregados * MMsagregados / ((xsagregados * MMsagregados).sum())

    return MMsagregados, wsagregados, xsagregados


# ******************************************************************************************************************** #
#  ATENÇÃO: O CÓDIGO A SEGUIR SERÁ EXECUTADO APENAS QUANDO ESTE MÓDULO FOR RODADO COMO SCRIPT PRINCIPAL.               #
#           O CÓDIGO A SEGUIR SERVE PARA CONFERIR SE AS FUNÇÕES DESTE MÓDULO FUNCIONAM CORRETAMENTE.                   #
# ******************************************************************************************************************** #
# INÍCIO DO TESTE
# OBS: DISTRIBUIÇÃO GERADA POR ESTE CÓDIGO VS. DISTRIBUIÇÃO RECEBIDA POR E-MAIL PARA O PETRÓLEO P1
if __name__ == "__main__":
    # Importação de bibliotecas
    import pandas as pd
    import os
    from tabulate import tabulate

    # Cálculos a partir da função 'gerar_distribuição_massa_molar'
    MMsagregados, wsagregados, xsagregados = gerar_distribuição_massa_molar(2.7822, 1859, 30, 700, 7200,
                                                                            "superior", "trapezios")

    # Leitura do arquivo recebido por e-mail
    diretório_deste_modulo = os.path.dirname(__file__)
    diretório_do_xlsx = os.path.join(diretório_deste_modulo,
                                     'Outras referências', 'Sobre distribuição gama', 'distribuição_P1_Yanes.xlsx')
    df1 = pd.read_excel(diretório_do_xlsx, "Plan1")
    df1 = df1.drop(index=0)  # apagando a linha 0 do dataframe
    MMsagregados_Yanes = df1.iloc[:, 1].to_numpy()
    xssagregados_Yanes = df1.iloc[:, 2].to_numpy()

    # DataFrame comparando os resultados
    arredondar = 4
    df2 = pd.DataFrame({"MMs_Gaba": np.round(MMsagregados_Yanes, arredondar),
                        "MMs_Calc": np.round(MMsagregados, arredondar),
                        "MMs_DRA(%)": np.round(100 * np.abs(MMsagregados_Yanes - MMsagregados) / MMsagregados_Yanes,
                                               arredondar),
                        "xs_Gaba": np.round(xssagregados_Yanes, arredondar),
                        "xs_Calc": np.round(xsagregados, arredondar),
                        "xs_DRA(%)": np.round(100 * np.abs(xssagregados_Yanes - xsagregados) / xssagregados_Yanes,
                                              arredondar)})

    # Exibição dos resultados
    print("\n|", 118 * "-")
    print(f"| TESTE DA FUNCAO 'gerar_distribuicao_massa_molar'")
    print(tabulate(df2, headers=df2.columns, tablefmt='github', showindex=True))
    print("|", 118 * "-")
# FIM DO TESTE
# ******************************************************************************************************************** #
