# Importação de bibliotecas do python
import numpy as np
import scipy as scp

# Importação de módulos internos
from módulo_composições import normalizar_composição


# Função
def calcular_por_Distribuição_Gamma(alfa, MWavg, n_agregados, MWmin, MWmax, tipo_cálculo_MM_agregados):
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
    def f(MWi):
        return scp.stats.gamma.pdf(MWi, a=alfa, loc=MWmin, scale=(MWavg - MWmin)/alfa)
        # return (MWi - MWmono)**(alfa - 1) / (beta ** alfa * scp.special.gamma(alfa)) * np.exp(-(MWi - MWmono) / beta)

    # Limites das faixas de massa molar
    MM_limites_faixas = np.linspace(MWmin, MWmax, n_agregados + 1)

    # Massas molares dos agregados
    MMsagregados = np.zeros(n_agregados)
    match tipo_cálculo_MM_agregados:
        case "medio":
            def MWf(MWi):
                return MWi * f(MWi)  # Criando a função MWi*f(MWi)
            for i in range(n_agregados):
                numerador = scp.integrate.quad(MWf, MM_limites_faixas[i], MM_limites_faixas[i + 1])[0]
                denominador = scp.integrate.quad(f, MM_limites_faixas[i], MM_limites_faixas[i + 1])[0]
                MMsagregados[i] = numerador / denominador if denominador != 0 else MM_limites_faixas[i + 1]  # g/mol
                # OBS: Caso a integração reduza o valor até se aproximar de zero, usar o caso "superior".
        case "superior" | _:
            for i in range(n_agregados):
                MMsagregados[i] = MM_limites_faixas[i + 1]  # g/mol

    xsagregados = (scp.stats.gamma.cdf(MM_limites_faixas[1:], a=alfa, loc=MWmin, scale=(MWavg - MWmin)/alfa)
                   - scp.stats.gamma.cdf(MM_limites_faixas[:-1], a=alfa, loc=MWmin, scale=(MWavg - MWmin)/alfa))

    # Frações mássicas e frações mássicas cumulativas dos agregados
    wsagregados = xsagregados * MMsagregados / ((xsagregados * MMsagregados).sum())
    xsagregados, wsagregados = normalizar_composição(xsagregados), normalizar_composição(wsagregados)
    wsagregados_cumulativa = np.cumsum(wsagregados)

    return MMsagregados, xsagregados, wsagregados, wsagregados_cumulativa


def calcular_por_Yen_Mullins(MW_min, n_nanoagregação, n_clusterização, x_Asf0, x_Asf1):
    z0 = x_Asf0
    z1 = x_Asf1 * (1.0 - z0)
    z2 = 1.0 - z0 - z1

    xsagregados = np.array([z0, z1, z2])
    MMsagregados = np.array([MW_min, n_nanoagregação * MW_min, n_nanoagregação * n_clusterização * MW_min])

    # Frações mássicas e frações mássicas cumulativas dos agregados
    wsagregados = xsagregados * MMsagregados / ((xsagregados * MMsagregados).sum())
    xsagregados, wsagregados = normalizar_composição(xsagregados), normalizar_composição(wsagregados)
    wsagregados_cumulativa = np.cumsum(wsagregados)

    return MMsagregados, xsagregados, wsagregados, wsagregados_cumulativa


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
    MMs_agregados, xs_agregados, ws_agregados, ws_agregados_cumulativo = calcular_distribuição_massa_molar(
        2.7822, 1859, 30, 700, 7200, "superior")

    # Leitura do arquivo recebido por e-mail
    diretório_deste_modulo = os.path.dirname(__file__)
    diretório_do_xlsx = os.path.join(diretório_deste_modulo,
                                     'Outras referências', 'Sobre distribuição gama', 'distribuição_P1_Yanes.xlsx')
    df1 = pd.read_excel(diretório_do_xlsx, "Plan1")
    df1 = df1.drop(index=0)  # apagando a linha 0 do dataframe
    MMs_agregados_Yanes = df1.iloc[:, 1].to_numpy()
    xs_agregados_Yanes = df1.iloc[:, 2].to_numpy()

    # DataFrame comparando os resultados
    arredondar = 4
    df2 = pd.DataFrame({"MMs_Gaba": np.round(MMs_agregados_Yanes, arredondar),
                        "MMs_Calc": np.round(MMs_agregados, arredondar),
                        "MMs_DRA(%)": np.round(100 * np.abs(MMs_agregados_Yanes - MMs_agregados) / MMs_agregados_Yanes,
                                               arredondar),
                        "xs_Gaba": np.round(xs_agregados_Yanes, arredondar),
                        "xs_Calc": np.round(xs_agregados, arredondar),
                        "xs_DRA(%)": np.round(100 * np.abs(xs_agregados_Yanes - xs_agregados) / xs_agregados_Yanes,
                                              arredondar)})

    # Exibição dos resultados
    print("\n|", 118 * "-")
    print(f"| TESTE DA FUNCAO 'gerar_distribuicao_massa_molar'")
    print(tabulate(df2, headers=df2.columns, tablefmt='github', showindex=True))
    print("|", 118 * "-")
# FIM DO TESTE
# ******************************************************************************************************************** #
