# Importação de bibliotecas do python
import numpy as np
import scipy as scp
from scipy.constants import R  # m3*Pa/mol*K

# Importação de outros módulos deste projeto
from módulo_composições import normalizar_composição


# Função
def calcular_composições_ELL(T, xs_completo, deltas, Vs, xsagregados):
    """ Calcula os betas de Rachford-Rice e as composições das fases leve e pesada (base molar).
    
    Inputs:
        T (float)           : temperatura (K)
        xs_completo (array) : composição global do sistema em termos de
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
    """

    # Leitura da composição global
    zs = xs_completo.copy()  # útil p/ RachfordRice

    # Chute inicial: composição da fase leve
    xsL = zs.copy()  # composição global do sistema

    # Chute inicial: composição da fase pesada
    n_agregados = xsagregados.shape[0]
    xsH = np.zeros(4 + n_agregados)
    xsH[4:] = xsagregados  # pura em asfaltenos

    # Iterações
    erro = 1
    tol = 1e-12
    n_it, n_itmax = 0, 150
    while erro > tol:
        
        # VmL e VmH
        VmL = (xsL*Vs).sum()
        VmH = (xsH*Vs).sum()

        # deltamL e deltamH
        phisL = (xsL*Vs)/((xsL*Vs).sum())
        phisH = (xsH*Vs)/((xsH*Vs).sum())
        deltamL = (phisL*deltas).sum()
        deltamH = (phisH*deltas).sum()

        Ks = np.exp(Vs/VmH - Vs/VmL + np.log(Vs/VmL) - np.log(Vs/VmH) + (Vs/(R*T))*(deltas - deltamL)**2
                    - (Vs/(R*T))*(deltas - deltamH)**2)
        Ks[0:3] = [0, 0, 0]  # retirando os componentes Solvente, S e A da fase pesada

        # Função de Rachford-Rice
        RachfordRice = lambda betarr: (zs*(Ks - 1)/(1 + betarr*(Ks - 1))).sum()
        
        # Resolução da equação de Rachford-Rice
        try:
            limite_inferior_betarr, limite_superior_betarr = 1e-8, 1 - 1e-8
            if RachfordRice(limite_inferior_betarr)*RachfordRice(limite_superior_betarr) > 0: 
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
        xsL_post = zs/(1 + betarr*(Ks - 1))
        xsL_post = normalizar_composição(xsL_post)
        xsH_post = xsL_post*Ks  # não é necessário normalizar esta composição, pois a da fase leve já foi normalizada
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

    return betarr, xsL, xsH, n_it


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
    nH = betarr*nF
    nL = nF - nH

    # Nº mols e massa dos componentes distribuídos nas duas fases
    nsL, nsH = xsL*nL, xsH*nH  # mol
    msL, msH = nsL*MMs, nsH*MMs  # kg

    # Massa de petróleo nas fase leve, fase pesada e alimentação
    m_petróleoL = msL[1:].sum()  # tirando o solvente
    m_petróleoH = msH.sum()  # não há solvente na fase pesada
    m_petróleo = m_petróleoL + m_petróleoH  # kg

    # Massa de asfalteno na fase pesada
    m_asfaltenosH = msH[4:].sum()

    # Yield 
    yield_calc = m_asfaltenosH/m_petróleo
    
    return yield_calc
