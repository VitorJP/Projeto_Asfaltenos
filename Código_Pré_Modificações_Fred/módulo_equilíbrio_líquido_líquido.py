# Importação de bibliotecas do python
import numpy as np
import scipy as scp
from scipy.constants import R  # m3*Pa/mol*K
from pyswarms.single import GlobalBestPSO

# Importação de outros módulos deste projeto
from módulo_composições import normalizar_composição


# Subfunção: Aplicação do Modelo Termodinâmico
def calcular_ln_coeficiente(xs, modelo_termodinâmico, parâmetros_do_modelo):
    """ Calcula o logaritmo natural dos coeficiente (fugacidade ou atividade) para cada componente em uma fase da
    mistura de acordo com um modelo termodinâmico determinado.

    Inputs:
        xs (array)                                      : composição da fase em termos de
                                                          [Solvente, S, A, R, Asf0, Asf1, ...] (base molar)
        modelo_termodinâmico (string)                   : modelo escolhido, podendo ser um modelo de coeficiente de
                                                          fugacidode ou um modelo de coeficiente de atividade
        parâmetros_do_modelo (list)                     : lista dos parâmetros necessários para os cálculos
                                                          do modelo escolhido.
                                                              Ex: temperatura (T),
                                                                  volumes molares (Vs),
                                                                  parâmetros de solubilidade (deltas),
                                                                  etc.

    Outputs:
        ln_coeficiente (array)                          : ln dos coeficientes de cada componente (em uma fase)

    Refs:
        FLORY, P.J. Thermodynamics of High Polymer Solutions. The Journal of Chemical Physics, v.10, n.1, p.51-61, 1942
        HUGGINS, M.L. Solutions of Long Chain Compounds. The Journal of Chemical Physics, v.9, n.5, p.440, 1941.
    """

    if modelo_termodinâmico == "Flory-Huggins":
        T, Vs, deltas = parâmetros_do_modelo[0], parâmetros_do_modelo[1], parâmetros_do_modelo[2]

        # Volume Médio da Fase:
        Vm = (xs * Vs).sum()

        # Parâmetro de Solubilidade Médio da Fase:
        phis_fase = (xs * Vs) / Vm
        deltam = (phis_fase * deltas).sum()

        # ln do coeficiente de fugacidade
        termo_1 = np.clip((Vs / Vm), 1e-16, None)
        termo_2 = np.log(termo_1)
        termo_3 = (Vs / (R * T)) * (deltas - deltam) ** 2
        ln_coeficiente = (1 - termo_1) + termo_2 + termo_3

    else:  # Em caso de Erro, usar Flory-Huggins como padrão.
        T, Vs, deltas = parâmetros_do_modelo[0], parâmetros_do_modelo[1], parâmetros_do_modelo[2]

        # Volume Médio da Fase:
        Vm = (xs * Vs).sum()

        # Parâmetro de Solubilidade Médio da Fase:
        phis_fase = (xs * Vs) / Vm
        deltam = (phis_fase * deltas).sum()

        # ln do coeficiente de fugacidade
        termo_1 = np.clip((Vs / Vm), 1e-16, None)
        termo_2 = np.log(termo_1)
        termo_3 = (Vs / (R * T)) * (deltas - deltam) ** 2
        ln_coeficiente = (1 - termo_1) + termo_2 + termo_3

    return ln_coeficiente


# Subfunção: Cálculo da Distância do Plano Tangente (Tangent Plane Distance - TPD)
def calcular_tpd(ws, modelo_termodinâmico, parâmetros_do_modelo, ds):
    """ Calcula a distância do plano tangente reduzida para uma dada composição candidata para a formação de uma nova
        fase no sistema.
        OBS: tpd(w) = TPD(w)/RT = somatório[w * (ln(w) + ln(phi_w) - valor_referência] ou
             tpd(w) = TPD(w)/RT = somatório[w * (ln(w) + ln(gamma_w) - valor_referência]

        Inputs:
            ws (array)                       : composição candidata para nova fase em termos de
                                               [Solvente, S, A, R, Asf0, Asf1, ...] (base molar)
            modelo_termodinâmico (string)    : modelo escolhido, podendo ser um modelo de coeficiente de
                                               fugacidode ou um modelo de coeficiente de atividade
            parâmetros_do_modelo (list)      : lista dos parâmetros necessários para os cálculos
                                               do modelo escolhido.
                                                    Ex: temperatura (T),
                                                        volumes molares (Vs),
                                                        parâmetros de solubilidade (deltas),
                                                        etc.
            ds (arrays)                      : valor de referência calculado a partir da composição global, dado por:
                                               ln(z) + ln(phi_z) ou ln(z) + ln(gamma_z)

        Outputs:
            tpd (float)              : valor da distância do plano tangente reduzida.

        Ref:
            Michelsen, Mollerup. Thermodynamics Models Fundamentals & Computaional Aspects. Chapter 9, p. 232.

        OBS:
            Quando for aprimorar o código para 3+ fases, é preciso ajustar para receber vários zs e calcular
            o menor tpd dentre todos eles.
        OBS:
            É importante garantir que o valor de referência foi calculado para o mesmo tipo de modelo.
    """
    ws = np.clip(ws, 1e-100, 1)
    ws = normalizar_composição(ws)
    ln_ws = np.clip(np.log(ws), -1e16, None)
    ln_coeficiente = calcular_ln_coeficiente(ws, modelo_termodinâmico, parâmetros_do_modelo)

    return np.sum(ws * (ln_ws + ln_coeficiente - ds))


def análise_de_estabilidade(zs, T, deltas, Vs, valor_referência_para_z):
    """ Realiza o teste de estabilidade de fases utilizando Distância do Plano Tangente de Gibbs (TPD).

        Inputs:
            T (float)           : temperatura (K)
            zs (array)          : composição global do sistema em termos de
                                  [Solvente, S, A, R, Asf0, Asf1, ...] (base molar)
            deltas(array)       : parâmetros de solubilidade (Pa**0.5)
            Vs(array)           : volumes molares (m³/mol)

        Outputs:
            estável (bool)      : estabilidade da composição global
            x_melhor (array)    : melhor chute para composição da segunda fase (pesada) baseado no mínimo tpd
                                  em termos de [Solvente, S, A, R, Asf0, Asf1, ...] (base molar)

        OBS:
            Posteriormente, talvez seja necessário expandir para mais fases (líquidas ou vapor).
        """

    # Declaração dos parâmetros da PSO
    n_componentes, n_partículas, n_iterações = len(zs), 150, 150

    # Declaração dos limites da PSO
    upper_bounds = zs.copy()
    lower_bounds = np.minimum(1e-32, upper_bounds / 1000)

    # Declaração do Otimizador para a PSO (Particle Swarm Optimization)
    otimizador_PSO = GlobalBestPSO(
        n_particles=n_partículas,
        dimensions=n_componentes,
        options={'c1': 0.75, 'c2': 2.5, 'w': 1.0},
        bounds=(lower_bounds, upper_bounds)
    )

    # Declaração dos argumentos da PSO para o Cálculo de TPD
    parâmetros_PSO = {
        "modelo termodinâmico": "Flory-Huggins",            # Escolha do Modelo
        "parâmetros do modelo": [T, Vs, deltas],            # Parâmetros necessários para o Modelo escolhido
        "valor de referência": valor_referência_para_z,     # Cálculo de  ln(z) + ln(phi_z)
    }

    # Função de Objetivo para a PSO:
    def função_objetivo_otimização(matriz_PSO, parameters):
        modelo = parameters.get("modelo termodinâmico")
        parâmetros_do_modelo = parameters.get("parâmetros do modelo")
        d_zs = parameters.get("valor de referência")

        valores = np.empty(matriz_PSO.shape[0], dtype=float)
        for i in range(matriz_PSO.shape[0]):
            ws = matriz_PSO[i].copy()
            valores[i] = calcular_tpd(ws, modelo, parâmetros_do_modelo, d_zs)
        return np.array(valores)

    tpd_min, x_tpd_min = otimizador_PSO.optimize(
        função_objetivo_otimização, iters=n_iterações, parameters=parâmetros_PSO, verbose=True
    )
    x_tpd_min = normalizar_composição(x_tpd_min)

    return tpd_min, x_tpd_min


# Função
def calcular_composições_ELL(T, zs, deltas, Vs, xsagregados):
    """ Calcula os betas de Rachford-Rice e as composições das fases leve e pesada (base molar).
    
    Inputs:
        T (float)           : temperatura (K)
        zs (array)          : composição global do sistema em termos de
                              [Solvente, S, A, R, Asf0, Asf1, ...] (base molar)
        deltas(array)       : parâmetros de solubilidade (Pa**0.5)
        Vs(array)           : volumes molares (m³/mol)
        xsagregados (array) : frações molares dos agregados de asfaltenos

    Outputs:
        Uma tupla contendo os seguintes elementos:
            betarr (float) : parâmetro beta de Rachford-Rice        
            xsL (array)    : composição da fase leve (base molar)
            xsH (array)    : composição da fase pesada (base molar) 
            n_int (int)    : nº de iterações para convergência das composições de equilíbrio

    OBS:
        Posteriormente, talvez seja necessário expandir o equilíbrio (ELLL, ELLV...)
    """

    # Leitura da composição global
    zs = np.clip(zs, 1e-100, 1)
    zs = normalizar_composição(zs)
    # zs = xs_completo.copy()  # útil p/ RachfordRice

    # Cálculo do valor de referência para a TPD da Análise de Estabilidade fora do loop (otimização)
    tpd_parcial_zs = np.log(zs) + calcular_ln_coeficiente(zs, "Flory-Huggins", [T, Vs, deltas])

    # Análise de Estabilidade por Distância do Plano Tangente (TPD) de Gibbs
    tpd_mínimo, x_melhor = análise_de_estabilidade(zs, T, deltas, Vs, tpd_parcial_zs)
    # estável = False

    if tpd_mínimo > 1e-4:
        betarr = 0.0
        xsL = zs.copy()
        xsH = np.zeros(len(zs))
        n_it = 0

    else:
        # Chute inicial: composição da fase leve
        xsL = zs.copy()  # composição global do sistema

        # Chute inicial: composição da fase pesada
        xsH = x_melhor.copy()
        # n_agregados = xsagregados.shape[0]
        # xsH = np.zeros(4 + n_agregados)
        xsH[4:] = xsagregados  # pura em asfaltenos

        # Iterações
        erro, tol = 1, 1e-12
        n_it, n_itmax = 0, 150

        while erro > tol:

            # Cálculo dos K: Modelo de Solução Regular Modificada de Flory-Huggins
            ln_Ks = calcular_ln_coeficiente(xsL, "Flory-Huggins", [T, Vs, deltas]) \
                    - calcular_ln_coeficiente(xsH, "Flory-Huggins", [T, Vs, deltas])
            Ks = np.clip(np.exp(ln_Ks), 1e-16, 1e16)
            Ks[0:3] = [0, 0, 0]  # retirando os componentes Solvente, S e A da fase pesada

            # Função de Rachford-Rice
            RachfordRice = lambda betarr: (zs * (Ks - 1) / (1 + betarr * (Ks - 1))).sum()

            # Resolução da equação de Rachford-Rice
            try:
                limite_inferior_betarr, limite_superior_betarr = 1e-8, 1 - 1e-8
                if RachfordRice(limite_inferior_betarr) * RachfordRice(limite_superior_betarr) > 0:
                    raise ValueError(f"A funcao de Rachford-Rice nao muda de sinal com betarr entre "
                                     f"[{limite_inferior_betarr}, {limite_superior_betarr}].")
                else:
                    betarr = scp.optimize.brentq(RachfordRice, limite_inferior_betarr, limite_superior_betarr)
            except Exception:
                chute = np.array([1e-4])
                betarr = scp.optimize.fsolve(RachfordRice, chute)[0]
                if betarr < 0:
                    print(f"Neste ponto, a funcao 'brentq' falhou e a 'fsolve' foi acionada para retornar "
                          f"betarr = {betarr:.4e}.")

            # Ajuste físico de betarr
            betarr = float(np.clip(betarr, 0.0, 1.0))

            # Composições pós-RachfordRice
            xsL_post = zs / (1 + betarr * (Ks - 1))
            xsL_post = normalizar_composição(xsL_post)
            xsH_post = xsL_post * Ks  # não é necessário normalizar esta fase, pois a fase leve já foi normalizada
            xsH_post = normalizar_composição(xsH_post)

            # Erro para verificação de convergência
            errosL = np.abs(xsL - xsL_post)
            errosH = np.abs(xsH - xsH_post)
            maxerroL = errosL.max()
            maxerroH = errosH.max()
            erro = max(maxerroL, maxerroH)

            # Composições pré-RachfordRice para a próxima iteração
            xsL = xsL_post.copy()
            xsH = xsH_post.copy()

            # Incremento no número de iterações
            n_it = n_it + 1
            if n_it == n_itmax:
                print(f"A composicao nao convergiu com {n_itmax} iteracoes.")
                break

    return betarr, [xsL, xsH], n_it


# Função 
def calcular_yield_asfaltenos(betarr, xsL, xsH, MMs):
    """ Calcula o yield fracional de asfalteno após o cálculo de equilíbrio.
    
    Inputs:
        betarr (float) : Parâmetro beta de Rachford-Rice
        xsL (array)    : composição molar da fase leve
        xsH (array)    : composição molar da fase pesada 
        MMs (array)    : massas molares (kg/mol)

    Outputs:
        yield_calc (float): yield fracional de asfalteno (calculado)
    """

    # Nº mols de alimentação, da fase pesada e da fase leve
    nF = 1  # mol -> base de cálculo
    nH = betarr * nF
    nL = nF - nH

    # Nº mols e massa dos componentes distribuídos nas duas fases
    nsL, nsH = xsL * nL, xsH * nH  # mol
    msL, msH = nsL * MMs, nsH * MMs  # kg

    # Massa de petróleo nas fase leve, fase pesada e alimentação
    m_petróleoL = msL[1:].sum()  # tirando o solvente
    m_petróleoH = msH.sum()  # não há solvente na fase pesada
    m_petróleo = m_petróleoL + m_petróleoH  # kg

    # Massa de asfalteno na fase pesada
    m_asfaltenosH = msH[4:].sum()

    # Yield 
    yield_calc = m_asfaltenosH / m_petróleo

    return yield_calc


# ******************************************************************************************************************** #
#  ATENÇÃO: O CÓDIGO A SEGUIR SERÁ EXECUTADO APENAS QUANDO ESTE MÓDULO FOR RODADO COMO SCRIPT PRINCIPAL.               #
#           O CÓDIGO A SEGUIR SERVE PARA CONFERIR SE AS FUNÇÕES DESTE MÓDULO FUNCIONAM CORRETAMENTE.                   #
# ******************************************************************************************************************** #
# INÍCIO DO TESTE

# FIM DO TESTE
# ******************************************************************************************************************** #
