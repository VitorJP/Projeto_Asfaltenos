# Importação de bibliotecas do python
import numpy as np


# Função 
def normalizar_composição(frações):
    """ Normaliza as frações de modo que a soma delas seja 1.0.
    
    Inputs:
        frações (array): frações mássicas ou molares
    
    Outputs:
        frações_normalizadas (array): frações mássicas ou molares normalizadas
    """

    # Cálculo
    if frações.sum() >= 1e-12:
        frações_normalizadas = frações*(1/frações.sum())
    else:
        frações_normalizadas = np.zeros_like(frações)

    return frações_normalizadas


# Função
def fracionar_composição_SARA(ws_simplificados, SARA, wsagregados, MMs):
    """ Fragmenta a composição do sistema de [Solvente, Petróleo] para [Solvente, S, A, R, Asf0, Asf1, ...] 
    
    Inputs:
        ws_simplificados (array) : composição global do sistema em termos de [Solvente, Petróleo] (base mássica)
        SARA (array)             : composição SARA do petróleo (base mássica)
        wsagregados (array)      : frações mássicas dos agregados de asfaltenos
        MMs (array)              : massas molares dos componentes do sistema (kg/mol)

    Outputs:
        Uma tupla contendo os seguintes elementos:
            ws_completo (array) : composição global do sistema em termos de
                                  [Solvente, S, A, R, Asf0, Asf1, ...] (base mássica)
            xs_completo (array) : composição global do sistema em termos de
                                  [Solvente, S, A, R, Asf0, Asf1, ...] (base molar)
    """

    # Inicialização de arrays importantes
    n_dados_exp = ws_simplificados.shape[0]
    n_agregados = wsagregados.shape[0]
    ws_semicompleto = np.zeros((n_dados_exp, 5))
    ws_completo, xs_completo = [np.zeros((n_dados_exp, 4 + n_agregados)) for _ in range(2)]

    # Composição dos sistemas em termos de [Solvente, S, A, R, A] (base mássica)
    ws_semicompleto[:, 0] = ws_simplificados[:, 0]
    ws_semicompleto[:, 1:5] = ws_simplificados[:, [1]] * SARA

    # Composição dos sistemas em termos de [Solvente, S, A, R, Asf0, Asf1, ...] (base mássica)
    ws_completo[:, 0:4] = ws_semicompleto[:, 0:4]
    ws_completo[:, 4:] = ws_semicompleto[:, [4]] * wsagregados

    # Composição dos sistemas em termos de [Solvente, S, A, R, Asf0, Asf1, ...] (base molar)
    xs_completo = ws_completo/MMs  
    xs_completo = xs_completo/xs_completo.sum(axis=1, keepdims=True)

    return ws_completo, xs_completo


def simplificar_composição_SARA(frações_completo):
    """
    Simplifica a composição do sistema de [Solvente, S, A, R, Asf0, Asf1, ...] para [Solvente, S, A, R, Asf_total]

        Inputs:
            frações_completo (array)      : composição em termos de [Solvente, S, A, R, Asf0, Asf1, ...]

        Outputs:
            frações_simplificado (array)  : composição em termos de [Solvente, S, A, R, Asf_total]
    """

    frações_simplificado = np.zeros(5)
    frações_simplificado[:4] = frações_completo[:4]
    frações_simplificado[4] = np.sum(frações_completo[4:])
    return frações_simplificado


def converter_fração_molar_para_fração_volumétrica(xs, Vs):
    phis = xs * Vs[None, :] / (xs * Vs[None, :]).sum(axis=1, keepdims=True)
    return phis


def converter_fração_mássica_para_fração_volumétrica(ws, rhos):
    phis = (ws / rhos[None, :]) / (ws / rhos[None, :]).sum(axis=1, keepdims=True)
    return phis


# ******************************************************************************************************************** #
#  ATENÇÃO: O CÓDIGO A SEGUIR SERÁ EXECUTADO APENAS QUANDO ESTE MÓDULO FOR RODADO COMO SCRIPT PRINCIPAL.               #
#           O CÓDIGO A SEGUIR SERVE PARA CONFERIR SE AS FUNÇÕES DESTE MÓDULO FUNCIONAM CORRETAMENTE.                   #
# ******************************************************************************************************************** #
# INÍCIO DO TESTE
# OBS: GABARITO EXTRAÍDO DA PG. 105 DA DISSERTAÇÃO DE YANES (2018)
if __name__ == "__main__":

    # Função 'normalizar_composição'
    sara_crua = np.array([0.32, 0.27, 0.35, 0.0634])
    sara_normalizada = normalizar_composição(sara_crua)
    sara_gabarito = np.array([0.3172630549, 0.2721509606, 0.3471670852, 0.06341889918])
    print("\n|", 119*"-")
    print("| TESTE DA FUNCAO 'normalizar_composição'")
    print(f"| SARA calculada: {sara_normalizada}")
    print(f"| SARA gabarito: {sara_gabarito}")
    print("|", 119*"-")
# FIM DO TESTE
# ******************************************************************************************************************** #
