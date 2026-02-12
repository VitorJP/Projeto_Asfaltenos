# Importação de bibliotecas do python
import numpy as np
import scipy as scp
from scipy.constants import R  # m3*Pa/mol*K
from pyswarms.single import GlobalBestPSO
from itertools import combinations, permutations

# Importação de outros módulos deste projeto
from módulo_composições import normalizar_composição


# ==================================================================================================================== #
# Subfunção: Aplicação do Modelo Termodinâmico
def calcular_ln_coeficiente(xs, dados_modelo_termodinâmico):
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

    modelo = dados_modelo_termodinâmico.get("modelo")
    parâmetros_modelo = dados_modelo_termodinâmico.get("parâmetros")

    if not np.all(xs == 0):
        if modelo == "Flory-Huggins":
            T, Vs, deltas = parâmetros_modelo[0], parâmetros_modelo[1], parâmetros_modelo[2]

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
            T, Vs, deltas = parâmetros_modelo[0], parâmetros_modelo[1], parâmetros_modelo[2]

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
    else:
        ln_coeficiente = 0

    return ln_coeficiente


# Subfunção: Cálculo da Distância do Plano Tangente (Tangent Plane Distance - TPD) de Gibbs
def calcular_tpd(ws, zs, dados_modelo_termodinâmico):
    """ Calcula a distância do plano tangente reduzida para uma dada composição candidata para a formação de uma nova
        fase no sistema.
        OBS: tpd(w) = TPD(w)/RT = somatório[w * (ln(w) + ln(phi_w) - d] ou
             tpd(w) = TPD(w)/RT = somatório[w * (ln(w) + ln(gamma_w) - d]

        Inputs:
            ws (array)                  : composição candidata para nova fase em termos de
                                          [Solvente, S, A, R, Asf0, Asf1, ...] (base molar)
            zs (array)                  : composição global do sistema em termos de
                                          [Solvente, S, A, R, Asf0, Asf1, ...] (base molar)
            dados_modelo_termodinâmico  : um dicionário contendo:
                                              modelo (string)   : modelo de coeficiente de fugacidade/atividade adotado.
                                              parâmetros (list) : lista com os parâmetros para o cálculo do modelo
                                                                  adotado (ex: [T, Vs, deltas] )

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

    ln_ws = np.empty_like(ws, dtype=float)
    mask = ws > 0
    ln_ws[mask] = np.log(ws[mask])
    ln_ws[~mask] = 0

    ln_zs = np.empty_like(zs, dtype=float)
    mask = zs > 0
    ln_zs[mask] = np.log(zs[mask])
    ln_zs[~mask] = 0

    return np.sum(ws * (ln_ws + calcular_ln_coeficiente(ws, dados_modelo_termodinâmico)
                        - ln_zs - calcular_ln_coeficiente(zs, dados_modelo_termodinâmico)))


# Subfunção: Energia Livre de Gibbs Adimensional
def calcular_Delta_G_sistema(beta_a, xs_2a, xs_2b, xs_1, dados_modelo_termodinâmico):
    """ Calcula a redução da Energia Livre de Gibbs do sistema ao com a separação em duas fases.

            Inputs:
                beta_a (float)              : proporção de fase A no sistema bifásico.
                xs_2a (array)               : composição da nova fase A em termos de
                                              [Solvente, S, A, R, Asf0, Asf1, ...] (base molar)
                xs_2b (array)               : composição da nova fase B em termos de
                                              [Solvente, S, A, R, Asf0, Asf1, ...] (base molar)
                xs_1 (array)                : composição do sistema em estado monofásico em termos de
                                              [Solvente, S, A, R, Asf0, Asf1, ...] (base molar)
                dados_modelo_termodinâmico  : um dicionário contendo:
                                                  modelo (string)   : modelo de coeficiente de fugacidade/atividade.
                                                  parâmetros (list) : lista com os parâmetros para o cálculo do modelo
                                                                      adotado (ex: [T, Vs, deltas] )

            Outputs:
                Delta_G (float)              : diferença entre a Energia Livre de Gibbs reduzida do sistema bifásico
                                               e a Energia Livre de Gibbs reduzida do sistema monofásico.

            Ref:
                Michelsen, Mollerup. Thermodynamics Models Fundamentals & Computaional Aspects. Chapter 9, p. 232.
    """

    G_fase_a = beta_a * calcular_tpd(xs_2a, xs_1, dados_modelo_termodinâmico)
    G_fase_b = (1 - beta_a) * calcular_tpd(xs_2b, xs_1, dados_modelo_termodinâmico)
    Delta_G = 0.0 if np.abs(G_fase_a + G_fase_b) < 1e-8 else G_fase_a + G_fase_b
    return Delta_G


# ==================================================================================================================== #
# Subfunção: Equação de Rachford-Rice
def calcular_beta_rachford_rice(Ks, zs):
    """ Calcula a proporção de fase pesada (beta) no sistema pelo método de Rachford-Rice.

            Inputs:
                Ks (array)      : coeficiente de equilíbrio entre as fases pesada e leve em termos de
                                  [Solvente, S, A, R, Asf0, Asf1, ...] (base molar)
                zs (array)      : composição global do sistema em termos de
                                  [Solvente, S, A, R, Asf0, Asf1, ...] (base molar)

            Outputs:
                beta (float)    : proporção de fase pesada no sistema
    """

    def equação_rachford_rice(beta_rr):
        return (zs * (Ks - 1) / (1 + beta_rr * (Ks - 1))).sum()

    tol = 1e-8
    limite_inferior_beta, limite_superior_beta = tol, 1 - tol

    if equação_rachford_rice(limite_inferior_beta) * equação_rachford_rice(limite_superior_beta) < 0:
        beta = scp.optimize.brentq(equação_rachford_rice, limite_inferior_beta, limite_superior_beta, xtol=1e-8)
        beta = np.clip(beta, limite_inferior_beta, limite_superior_beta)
        sistema_bifásico = True
    elif equação_rachford_rice(limite_inferior_beta) > 0 and equação_rachford_rice(limite_superior_beta) > 0:
        beta = 1.0
        sistema_bifásico = False
    else:
        beta = 0.0
        sistema_bifásico = False

    return sistema_bifásico, beta


# Subfunção: Equações das Composições das Fases por K e z
def calcular_composições_fases(Ks, zs, beta):
    """ Calcula a proporção de fase pesada (beta) no sistema pelo método de Rachford-Rice.

            Inputs:
                Ks (array)          : coeficiente de equilíbrio entre as fases pesada e lelve em termos de
                                      [Solvente, S, A, R, Asf0, Asf1, ...] (base molar)
                zs (array)          : composição global do sistema em termos de
                                      [Solvente, S, A, R, Asf0, Asf1, ...] (base molar)
                beta (float)        : proporção de fase pesada no sistema

            Outputs:
                x_leve (array)      : composição da fase leve em termos de
                                      [Solvente, S, A, R, Asf0, Asf1, ...] (base molar)
                x_pesada (array)    : composição da fase pesada em termos de
                                      [Solvente, S, A, R, Asf0, Asf1, ...] (base molar)
    """

    beta = np.clip(beta, 1e-8, 1 - 1e-8)

    x_leve = np.clip(zs / (1 + beta * (Ks - 1)), 0, 1)
    x_leve[x_leve < 1e-8] = 0.0
    x_leve = normalizar_composição(x_leve)

    x_pesada = np.clip(Ks * x_leve, 0, 1)
    x_pesada[x_pesada < 1e-8] = 0.0
    x_pesada = normalizar_composição(x_pesada)

    return x_leve, x_pesada


# Função: Cálculo Flash por Aplicação de Rachford-Rice
def calcular_flash(ws, zs, dados_modelo_termodinâmico):
    """ Calcula as frações das fases leve e pesada e a proporção de fase pesasa (beta) por iterações sucessivas
        a partir de um chute inicial de fase pesada.

            Inputs:
                ws (array)                  : composição da fase leve em termos de
                                              [Solvente, S, A, R, Asf0, Asf1, ...] (base molar)
                zs (array)                  : composição global do sistema em termos de
                                              [Solvente, S, A, R, Asf0, Asf1, ...] (base molar)
                dados_modelo_termodinâmico  : um dicionário contendo:
                                                modelo (string)     : modelo de coeficiente de fugacidade/atividade.
                                                parâmetros (list)   : lista com os parâmetros para o cálculo do modelo
                                                                      adotado (ex: [T, Vs, deltas] )

            Outputs:
                Uma tupla contendo os seguintes elementos:
                    betarr (float) : parâmetro beta de Rachford-Rice
                    xsL (array)    : composição da fase leve (base molar)
                    xsH (array)    : composição da fase pesada (base molar)
                    n_int (int)    : nº de iterações para convergência das composições de equilíbrio
    """

    Ks = np.empty_like(ws, dtype=float)
    mask = zs >= 1e-8
    Ks[mask] = ws[mask] / zs[mask]
    Ks[~mask] = 1.0
    Ks[Ks < 1e-6] = 0.0

    tol, erro = 1e-8, 1
    n_it, n_it_max = 0, 50

    while erro > tol:
        s, beta = calcular_beta_rachford_rice(Ks, zs)
        xs_L, xs_H = calcular_composições_fases(Ks, zs, beta)

        ln_Ks_post = calcular_ln_coeficiente(xs_L, dados_modelo_termodinâmico) \
                     - calcular_ln_coeficiente(xs_H, dados_modelo_termodinâmico)
        ln_Ks_post = np.clip(ln_Ks_post, None, np.log(1e6))
        Ks_post = np.where(zs == 0, 1.0, np.exp(ln_Ks_post))
        Ks_post[Ks < 1e-6] = 0.0

        # Critério de Convergência
        erro = np.sum(np.abs(Ks_post - Ks) / (1 + Ks))
        Ks = Ks_post

        n_it += 1
        if n_it >= n_it_max:
            # print(f"O método de Rachford-Rice não convergiu em {n_it} iterações. O erro final foi {erro}.")
            break

    duas_fases, beta = calcular_beta_rachford_rice(Ks, zs)
    if duas_fases:
        xs_L, xs_H = calcular_composições_fases(Ks, zs, beta)
    else:
        xs_L, xs_H = zs, zs

    return duas_fases, beta, xs_L, xs_H, n_it


# ==================================================================================================================== #
# Subfunção
def criar_conjunto_composições_candidatas(zs, xs_agregados):
    """ Calcula o beta da proporção entre as fases e as composições das fases leve e pesada (base molar).

            Inputs:
                zs (array)          : composição global do sistema em termos de
                                      [Solvente, S, A, R, Asf0, Asf1, ...] (base molar)
                xsagregados (array) : frações molares dos agregados de asfaltenos

            Outputs:
                conjunto_composições candidatas:    matriz contendo múltiplas composições candidatas para a
                                                    formação de uma segunda fase no sistema.
    """

    n_componentes = len(zs)
    conjunto_composições_candidatas = []

    composição_zero = np.zeros(n_componentes)
    puro_asfaltenos = np.zeros(n_componentes)
    puro_asfaltenos[4:] = xs_agregados
    conjunto_composições_candidatas.extend([puro_asfaltenos, composição_zero])

    def vértices(n):  # N partículas
        lista_composições = []
        for z in range(n):
            if zs[z] != 0:
                composição = np.zeros(n)
                composição[z] = 1
                lista_composições.append(composição)
        return lista_composições

    conjunto_composições_candidatas.extend(vértices(n_componentes))

    def pares_binários(n):  # 3*N*(N-1)/2 partículas
        lista_composições = []
        for i, j in combinations(range(n), 2):
            if zs[i] != 0 and zs[j] != 0:
                # (50% , 50%)
                composição = np.zeros(n)
                composição[i], composição[j] = 0.5, 0.5
                lista_composições.append(composição)
                # (25%, 75%)
                composição = np.zeros(n)
                composição[i], composição[j] = 0.25, 0.75
                lista_composições.append(composição)
                # (75%, 25%)
                composição = np.zeros(n)
                composição[i], composição[j] = 0.75, 0.25
                lista_composições.append(composição)
        return lista_composições

    conjunto_composições_candidatas.extend(pares_binários(n_componentes))

    def pertubação_em_zs(n, escalas=(0.01, 0.05, 0.2), seed=None):  # 3*N partículas
        rng = np.random.default_rng(seed)
        lista_partículas = []
        for escala in escalas:
            for _ in range(n):
                perturbação = rng.normal(loc=0.0, scale=escala, size=n)
                partícula = zs * (1.0 + perturbação)
                lista_partículas.append(partícula)
        return lista_partículas

    conjunto_composições_candidatas.extend(pertubação_em_zs(n_componentes))

    def distribuição_dirichlet(n, alfas=(0.1, 1.0, 5.0), seed=None):  # 9*N partículas
        rng = np.random.default_rng(seed)
        lista_composições = []
        tamanho_lista = 3 * n
        for a in alfas:
            alfa = np.ones(n) * a
            composições_dirichlet = rng.dirichlet(alfa, size=tamanho_lista)
            for composição in composições_dirichlet:
                composição = normalizar_composição(composição)
                lista_composições.append(composição)
        return lista_composições

    conjunto_composições_candidatas.extend(distribuição_dirichlet(n_componentes))
    conjunto_composições_candidatas = np.array(conjunto_composições_candidatas)
    n_candidatas = conjunto_composições_candidatas.shape[0]
    print('n candidatas:', n_candidatas)

    return conjunto_composições_candidatas


# Função
def análise_de_estabilidade(ws_candidatas, zs, dados_modelo_termodinâmico):
    """ Calcula o beta da proporção entre as fases e as composições das fases leve e pesada (base molar).

                Inputs:
                    ws_candidatas (array-2d)    : conjunto de composições candidatas para a fase pesada em termos de
                                                  [Solvente, S, A, R, Asf0, Asf1, ...] (base molar)
                    zs (array)                  : composição global do sistema em termos de
                                                  [Solvente, S, A, R, Asf0, Asf1, ...] (base molar)
                    dados_modelo_termodinâmico  : um dicionário contendo:
                                                    modelo (string)     : modelo de coeficiente de fugacidade/atividade.
                                                    parâmetros (list)   : lista com os parâmetros para o cálculo do
                                                                          modelo adotado (ex: [T, Vs, deltas] )


                Outputs:
                    instabilidade_detectada (bool)  : retorna 'True' se houver uma composição que promova a
                                                      separação de fases.
    """

    instabilidade_detectada = False
    for ws in ws_candidatas:
        tpd_ws = calcular_tpd(ws, zs, dados_modelo_termodinâmico)
        if tpd_ws < 0:
            instabilidade_detectada = True
            break
    return instabilidade_detectada


# Função
def identificar_composição_com_G_mínimo(ws_candidatas, zs, dados_modelo_termodinâmico):
    """ Calcula o beta da proporção entre as fases e as composições das fases leve e pesada (base molar).

                Inputs:
                    ws_candidatas (array-2d)    : conjunto de composições candidatas para a fase pesada em termos de
                                                  [Solvente, S, A, R, Asf0, Asf1, ...] (base molar)
                    zs (array)                  : composição global do sistema em termos de
                                                  [Solvente, S, A, R, Asf0, Asf1, ...] (base molar)
                    dados_modelo_termodinâmico  : um dicionário contendo:
                                                    modelo (string)     : modelo de coeficiente de fugacidade/atividade.
                                                    parâmetros (list)   : lista com os parâmetros para o cálculo do
                                                                          modelo adotado (ex: [T, Vs, deltas] )

                Outputs:
                    duas_fases (bool)   : indica se foi determinada uma composição que promova separação de fases.
                    beta (float)        : parâmetro beta de Rachford-Rice
                    xs_L (array)        : composição da fase leve (base molar)
                    xs_H (array)        : composição da fase pesada (base molar)
                    n_int (int)         : nº de iterações para convergência das composições de equilíbrio
    """

    resultados = []
    for ws in ws_candidatas:

        # Cálculo Flash
        duas_fases, beta, xs_L, xs_H, n_it = calcular_flash(ws, zs, dados_modelo_termodinâmico)

        # Cálculo da Energia Livre de Gibbs
        Delta_G = calcular_Delta_G_sistema(beta, xs_H, xs_L, zs, dados_modelo_termodinâmico)

        resultados.append((duas_fases, beta, xs_L, xs_H, n_it, Delta_G))

    resultados = np.array(resultados, dtype=object)

    # Separação dos dados em colunas
    duas_fases = resultados[:, 0].astype(bool)
    beta = resultados[:, 1].astype(float)
    xs_L = resultados[:, 2]
    xs_H = resultados[:, 3]
    n_it = resultados[:, 4].astype(int)
    Delta_G = resultados[:, 5].astype(float)

    # Filtra os chutes que dão 2 fases e acha o menor G
    casos_válidos = Delta_G[duas_fases]
    i_rel = np.argmin(casos_válidos)  # índice relativo dentro dos válidos
    i_abs = np.where(duas_fases)[0][i_rel]  # índice absoluto no array completo

    return duas_fases[i_abs], beta[i_abs], xs_L[i_abs], xs_H[i_abs], n_it[i_abs]


# Função
def calcular_composições_ELL(T, zs, deltas, Vs, xsagregados, MMs):
    """ Calcula o beta da proporção entre as fases e as composições das fases leve e pesada (base molar).

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
    zs[zs < 1e-8] = 0
    zs = normalizar_composição(zs)

    # Declaração do Dicionário do Modelo Termodinâmico
    dados_modelo_termodinamico = {
        "modelo": "Flory-Huggins",  # Escolha do Modelo
        "parâmetros": [T, Vs, deltas],  # Parâmetros necessários para o Modelo escolhido
    }

    chute_inicial_asfaltenico, printar = True, False

    # Método de indiciação do chute inicial
    if chute_inicial_asfaltenico:
        ws_inicial = np.zeros(len(zs))
        ws_inicial[4:] = xsagregados
        duas_fases, beta_opt, xs_L_opt, xs_H_opt, n_iter = calcular_flash(ws_inicial, zs, dados_modelo_termodinamico)
        G_opt = calcular_Delta_G_sistema(beta_opt, xs_H_opt, xs_L_opt, zs, dados_modelo_termodinamico)

    # Método do conjunto de chutes iniciais
    else:
        ws_candidatas = criar_conjunto_composições_candidatas(zs, xsagregados)
        sistema_é_instável = análise_de_estabilidade(ws_candidatas, zs, dados_modelo_termodinamico)
        if sistema_é_instável:
            duas_fases, beta_opt, xs_L_opt, xs_H_opt, n_iter = identificar_composição_com_G_mínimo(
                ws_candidatas, zs, dados_modelo_termodinamico
            )
        else:
            duas_fases, beta_opt, xs_L_opt, xs_H_opt, n_iter = False, 0.0, zs, zs, 0
        G_opt = calcular_Delta_G_sistema(beta_opt, xs_H_opt, xs_L_opt, zs, dados_modelo_termodinamico)

    n_beta = xs_H_opt * beta_opt
    m_beta = n_beta * MMs
    m_beta_total = m_beta[1:].sum()
    n_z = zs
    m_z = n_z * MMs
    m_z_total = m_z[1:].sum()

    m_beta_rel = m_beta_total / m_z_total

    if printar:
        print("duas fases?", duas_fases)
        print("beta: ", beta_opt)
        print("m_fase_beta(%): ", m_beta_rel)
        print("xs_fase_beta: ", xs_H_opt)
        print("xs_fase_0: ", xs_L_opt)
        print("zs: ", zs)
        print("G x10³: ", 1000*G_opt)
        print()

    return beta_opt, xs_L_opt, xs_H_opt, n_iter


# ==================================================================================================================== #
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
    nF = 1  # mol (base de cálculo)
    nH = betarr * nF
    nL = nF - nH

    # Nº mols e massa dos componentes distribuídos nas duas fases
    nsL, nsH = xsL * nL, xsH * nH  # mol
    msL, msH = nsL * MMs, nsH * MMs  # kg

    # Massa de petróleo nas fase leve, fase pesada e alimentação
    m_petróleo_L = msL[1:].sum()  # tirando o solvente
    m_petróleo_H = msH[1:].sum()  # tirando o solvente (teoricamente, não há solvente na fase pesada)
    m_petróleo = m_petróleo_L + m_petróleo_H  # kg

    # Massa de asfalteno na fase pesada
    m_asfaltenos_H = msH[1:].sum()

    # Yield
    yield_calc = m_asfaltenos_H / m_petróleo

    return yield_calc

# ******************************************************************************************************************** #
#  ATENÇÃO: O CÓDIGO A SEGUIR SERÁ EXECUTADO APENAS QUANDO ESTE MÓDULO FOR RODADO COMO SCRIPT PRINCIPAL.               #
#           O CÓDIGO A SEGUIR SERVE PARA CONFERIR SE AS FUNÇÕES DESTE MÓDULO FUNCIONAM CORRETAMENTE.                   #
# ******************************************************************************************************************** #
# INÍCIO DO TESTE

# FIM DO TESTE
# ******************************************************************************************************************** #
