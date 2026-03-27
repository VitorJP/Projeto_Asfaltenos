# Importação de bibliotecas do python
import numpy as np
import scipy as scp
from scipy.constants import R  # m3*Pa/mol*K
from itertools import combinations, permutations
import chemicals
from fluids.numerics import UnconvergedError

# Importação de outros módulos deste projeto
from módulo_composições import normalizar_composição, simplificar_composição_SARA


# ==================================================================================================================== #
# Subfunção
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

    if np.all(xs == 0):
        return np.zeros_like(xs)

    match modelo:
        case "Flory-Huggins":
            T, Vs, deltas = parâmetros_modelo[0], parâmetros_modelo[1], parâmetros_modelo[2]

            # Volume Médio da Fase:
            Vm = np.sum(xs * Vs, axis=-1, keepdims=True)

            # Parâmetro de Solubilidade Médio da Fase:
            phis = (xs * Vs) / Vm
            deltam = np.sum(phis * deltas, axis=-1, keepdims=True)

            # ln do coeficiente de fugacidade
            termo_1 = np.clip((Vs / Vm), 1e-16, None)
            termo_2 = np.log(termo_1)
            termo_3 = (Vs / (R * T)) * (deltas - deltam) ** 2

            ln_coeficiente = (1 - termo_1) + termo_2 + termo_3

        case _:
            raise ValueError("Modelo não implementado.")

    return ln_coeficiente


# Subfunção
def calcular_tpd(xs_fase_candidata, xs_fase_preliminar, dados_modelo_termodinâmico):
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

    ln_xs_candidata = np.zeros_like(xs_fase_candidata, dtype=float)
    mask = xs_fase_candidata > 0
    ln_xs_candidata[mask] = np.log(xs_fase_candidata[mask])

    ln_xs_preliminar = np.zeros_like(xs_fase_preliminar, dtype=float)
    mask = xs_fase_preliminar > 0
    ln_xs_preliminar[mask] = np.log(xs_fase_preliminar[mask])

    return np.sum(xs_fase_candidata *
                  (ln_xs_candidata + calcular_ln_coeficiente(xs_fase_candidata, dados_modelo_termodinâmico) -
                   ln_xs_preliminar - calcular_ln_coeficiente(xs_fase_preliminar, dados_modelo_termodinâmico))
                  )


# Subfunção
def calcular_Delta_G_mistura(betas, xs_multi, xs_mono, dados_modelo_termodinâmico):
    """ Calcula a Energia Livre de Gibbs de Mistura (Adimensional) do sistema multicomponente multifásico.

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

    if xs_multi.shape[0] == 1:
        return 0.0
    else:
        # Energia Livre de Gibbs: Caso Sistema Multifásico
        mask_multi = xs_multi > 0
        ln_xs_multi = np.zeros_like(xs_multi, dtype=float)
        ln_xs_multi[mask_multi] = np.log(xs_multi[mask_multi])
        ln_gamma_multi = calcular_ln_coeficiente(xs_multi, dados_modelo_termodinâmico)
        G_multi = np.sum(np.sum(betas[:, None] * xs_multi * (ln_xs_multi + ln_gamma_multi)))

        # Energia Livre de Gibbs: Caso Sistema Monofásico
        mask_mono = xs_mono > 0
        ln_xs_mono = np.zeros_like(xs_mono, dtype=float)
        ln_xs_mono[mask_mono] = np.log(xs_mono[mask_mono])
        ln_gamma_mono = calcular_ln_coeficiente(xs_mono, dados_modelo_termodinâmico)
        G_mono = np.sum(xs_mono * (ln_xs_mono + ln_gamma_mono))

        return G_multi - G_mono


# ==================================================================================================================== #
# Subfunção
def criar_conjunto_composições_candidatas(xs_global, xs_agregados, conjunto_completo=True):
    n_componentes = len(xs_global)
    conjunto_composições_candidatas = []

    def caso_asfaltênico_total(xs_asf):
        puro_asfaltenos = np.zeros(n_componentes)
        puro_asfaltenos[4:] = xs_asf
        return [puro_asfaltenos]

    def vértices(n):  # N casos
        lista_composições = []
        for i in range(n):
            if xs_global[i] != 0:
                composição = np.zeros(n)
                composição[i] = 1
                lista_composições.append(composição)
        return lista_composições

    def pares_binários(n):  # 3*N*(N-1)/2 casos
        lista_composições = []
        for i, j in combinations(range(n), 2):
            if xs_global[i] != 0 and xs_global[j] != 0:
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

    def pertubação_em_z(n, escalas=(0.01, 0.05, 0.2), seed=None):  # 3*N casos
        rng = np.random.default_rng(seed)
        lista_composições = []
        for escala in escalas:
            for _ in range(n):
                perturbação = rng.normal(loc=0.0, scale=escala, size=n)
                composição = xs_global * (1.0 + perturbação)
                lista_composições.append(composição)
        return lista_composições

    def distribuição_dirichlet(n, alfas=(0.1, 1.0, 5.0), seed=None):  # 9*N casos
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

    # conjunto_composições_candidatas.extend(caso_asfaltênico_total(xs_agregados))

    if conjunto_completo:
        conjunto_composições_candidatas.extend(vértices(n_componentes))
        conjunto_composições_candidatas.extend(pares_binários(n_componentes))
        # conjunto_composições_candidatas.extend(pertubação_em_z(n_componentes))
        conjunto_composições_candidatas.extend(distribuição_dirichlet(n_componentes))

    return np.array(conjunto_composições_candidatas)


# Subfunção
def selecionar_composições_instáveis(conjunto_xs_candidatas, xs_fase_preliminar, dados_modelo_termodinâmico):
    conjunto_xs_instáveis, conjunto_tpd = [], []
    for xs_fase_candidata in conjunto_xs_candidatas:
        tpd = calcular_tpd(xs_fase_candidata, xs_fase_preliminar, dados_modelo_termodinâmico)
        if tpd < 0.0:
            conjunto_xs_instáveis.append(xs_fase_candidata)
            conjunto_tpd.append(tpd)
    conjunto_xs_instáveis, conjunto_tpd = np.array(conjunto_xs_instáveis), np.array(conjunto_tpd)

    return conjunto_xs_instáveis


# Subfunção
def preparar_chute_para_flash(n_fase_particionada, xs_fase_nova, xs_fases_preliminares, betas_preliminares):
    xs_chute = np.vstack([xs_fase_nova, xs_fases_preliminares])
    betas_chute = [betas_preliminares[n_fase_particionada] * 0.01]

    for i in range(len(xs_fases_preliminares)):
        if i == n_fase_particionada:
            betas_chute.append(0.99 * betas_preliminares[i])
        else:
            betas_chute.append(betas_preliminares[i])

    return np.array(xs_chute), normalizar_composição(np.array(betas_chute))


# ==================================================================================================================== #
# Subfunção
def identificar_fase_asfaltênica(xs_fases, index_asfaltenos=4):
    if xs_fases.shape[0] > 1:
        frações_asfaltênicas = xs_fases[:, index_asfaltenos:].sum(axis=1)
        index_fase_asfaltênica = np.argmax(frações_asfaltênicas)
        print("\nFase asfaltênica é: ", index_fase_asfaltênica+1)
        return index_fase_asfaltênica
    else:
        print("\nNão há fase asfaltênica.")
        return None


# Subfunção
def imprimir_dados_das_fases(xs_global, T, MMs, xs_fases, betas, Delta_G_mix):
    n_fases = xs_fases.shape[0]
    xs_global_simples = normalizar_composição(simplificar_composição_SARA(xs_global))
    massa_componentes_global = xs_global * MMs
    massa_total = massa_componentes_global.sum()
    massa_total_oleo = massa_componentes_global[1:].sum()

    print("\n", 120*"-", "\n")
    print("DADOS GERAIS")
    print("Temperatura: ", T-273.15, "°C")
    print("Composição global: ", np.round(100*xs_global_simples, decimals=2))
    print("Número de Fases: ", n_fases)
    print("Soma dos Betas: ", betas.sum())
    print("Delta G de mistura: ", np.round(Delta_G_mix, decimals=6), "J/mol")
    print("\n", 120*"-", "\n")

    if n_fases > 1:
        for k in range(n_fases):
            xs_fase_simples = normalizar_composição(simplificar_composição_SARA(xs_fases[k]))
            ns_fase = xs_fases[k] * betas[k]
            massa_componentes_fase = ns_fase * MMs
            massa_fase = massa_componentes_fase.sum()
            massa_fase_oleo = massa_componentes_fase[1:].sum()
            massa_fase_rel = massa_fase / massa_total
            massa_fase_oleo_rel = massa_fase_oleo / massa_total_oleo

            print("FASE ", k+1)
            print("Beta: ", np.round(betas[k], decimals=8))
            print("Composição: ", np.round(100*xs_fase_simples, decimals=2))
            print("Massa Total: ", np.round(100*massa_fase_rel, decimals=2), "%")
            print("Massa sem Alcano: ", np.round(100*massa_fase_oleo_rel, decimals=2), "%")
            print("\n", 100 * "-", "\n")


# ==================================================================================================================== #
# Função
def calcular_flash(betas_chute, xs_chute, zs, dados_modelo_termodinâmico, tol=1e-8, it_max=50, método_Yarranton=True):

    betas, xs = betas_chute.copy(), xs_chute.copy()
    n_fases, n_comp = xs.shape

    Ks = np.ones((n_fases - 1, n_comp), dtype=float)
    mask = zs >= tol
    Ks[:, mask] = xs[:-1, mask] / xs[-1, mask]
    Ks[Ks < tol] = 0.0

    for it in range(it_max):

        try:
            novos_betas, novos_xs = chemicals.rachford_rice.Rachford_Rice_solutionN(
                zs.tolist(), Ks.tolist(), betas[:-1].tolist()
            )
        except (ValueError, ZeroDivisionError, UnconvergedError):
            novos_betas, novos_xs = np.array(betas), np.array(xs)
            novos_betas = normalizar_composição(np.clip(novos_betas, 0.0, 1.0))
        else:
            novos_betas, novos_xs = np.array(novos_betas), np.array(novos_xs)
            novos_betas = normalizar_composição(np.clip(novos_betas, 0.0, 1.0))

        ln_novos_Ks = calcular_ln_coeficiente(novos_xs[-1], dados_modelo_termodinâmico) \
                      - calcular_ln_coeficiente(novos_xs[:-1], dados_modelo_termodinâmico)
        novos_Ks = np.where(zs == 0, 1.0, np.exp(ln_novos_Ks))
        if método_Yarranton:
            novos_Ks[Ks < tol] = 0.0
        else:
            novos_Ks[novos_Ks < tol] = 0.0

        # Critério de Convergência
        erro = np.sum(np.abs(novos_Ks - Ks) / (1 + Ks))
        Ks, xs, betas = novos_Ks, novos_xs, novos_betas

        if erro <= tol:
            break
        if it == it_max-1:
            # print(f"O método de Rachford-Rice não convergiu em {it+1} iterações. O erro final foi {erro}.")
            pass

    if all(tol < beta < 1.0 - tol for beta in betas):
        return True, xs, betas
    else:
        betas = normalizar_composição(np.clip(betas[1:], 0.0, 1.0))
        return False, xs[1:], betas


# Função
def calcular_equilíbrio_multifásico(T, zs, deltas, Vs, MMs, xsagregados, max_fases=20):

    # Leitura e normalização das composições
    zs = normalizar_composição(np.where(zs < 1e-8, 0, zs))
    xsagregados = normalizar_composição(np.where(xsagregados < 1e-8, 0, xsagregados))
    xs_total_asfaltenos = np.zeros_like(zs)
    xs_total_asfaltenos[4:] = xsagregados

    # Declaração do Dicionário do Modelo Termodinâmico
    dados_modelo_termodinâmico = {
        "modelo": "Flory-Huggins",  # Escolha do Modelo
        "parâmetros": [T, Vs, deltas],  # Parâmetros necessários para o Modelo escolhido
    }

    xs_fases, betas, Delta_G_mix = np.array([zs.copy()]), np.array([1.0]), 0.0

    # for f in range(len(xs_fases)):
    #     xs_chute_asfaltênico, betas = preparar_chute_para_flash(f, xs_total_asfaltenos, xs_fases, betas)
    #     s, xs_fases, betas = calcular_flash(
    #         betas, xs_chute_asfaltênico, zs, dados_modelo_termodinâmico, método_Yarranton=False
    #     )
    #     if s:
    #         print("Chute asfaltênico particiona.")
    #     Delta_G_mix = calcular_Delta_G_mistura(betas, xs_fases, zs, dados_modelo_termodinâmico)

    particionou = True
    conj_candidatos = criar_conjunto_composições_candidatas(zs, xsagregados, conjunto_completo=True)
    print("Número de composições candidatas totais: ", len(conj_candidatos))

    while particionou:

        for f in range(len(xs_fases)):

            print("\nNúmero de fases: ", xs_fases.shape[0])
            if xs_fases.shape[0] >= max_fases:
                particionou = False
                break
            conj_instáveis = selecionar_composições_instáveis(conj_candidatos, xs_fases[f], dados_modelo_termodinâmico)
            print("Número de composições candidatas instáveis: ", len(conj_instáveis))

            if len(conj_instáveis) == 0:
                particionou = False
            else:
                resultados_flash = []
                for fase_chute in conj_instáveis:
                    xs_chute, betas_chute = preparar_chute_para_flash(f, fase_chute, xs_fases, betas)

                    flash_separa, xs_flash, betas_flash = calcular_flash(
                        betas_chute, xs_chute, zs, dados_modelo_termodinâmico, método_Yarranton=False
                    )
                    if not flash_separa:
                        continue

                    Delta_G_mix = calcular_Delta_G_mistura(betas_flash, xs_flash, zs, dados_modelo_termodinâmico)
                    if Delta_G_mix < 0.0:
                        resultados_flash.append((Delta_G_mix, xs_flash, betas_flash))

                print("Número de composições candidatas pós-flash: ", len(resultados_flash))
                if not resultados_flash:
                    particionou = False
                    continue
                else:
                    # Etapa 4: Delta_G_mínimo
                    particionou = True
                    Delta_G_mix, xs_fases, betas = min(resultados_flash, key=lambda x: x[0])

    i_fase_asfaltênica = identificar_fase_asfaltênica(xs_fases)
    imprimir_dados_das_fases(zs, T, MMs, xs_fases, betas, Delta_G_mix)

    return betas, xs_fases, i_fase_asfaltênica
