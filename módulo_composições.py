# Importação de bibliotecas do python
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, fields


@dataclass
class Composição:

    _xs_completo: np.ndarray | None = None
    _ws_completo: np.ndarray | None = None
    _phis_completo: np.ndarray | None = None

    @staticmethod
    def converter_mol_massa(xs, MMs):
        ws = xs * MMs[None, :]
        ws = ws / ws.sum(axis=1, keepdims=True)
        return ws

    @staticmethod
    def converter_mol_volume(xs, Vs):
        phis = xs * Vs[None, :]
        phis = phis / phis.sum(axis=1, keepdims=True)
        return phis

    @staticmethod
    def converter_massa_mol(ws, MMs):
        xs = ws / MMs[None, :]
        xs = xs / xs.sum(axis=1, keepdims=True)
        return xs

    @staticmethod
    def converter_massa_volume(ws, rhos):
        phis = ws / rhos[None, :]
        phis = phis / phis.sum(axis=1, keepdims=True)
        return phis

    @staticmethod
    def converter_volume_massa(phis, rhos):
        ws = phis * rhos[None, :]
        ws = ws / ws.sum(axis=1, keepdims=True)
        return ws

    @staticmethod
    def converter_volume_mol(phis, Vs):
        xs = phis / Vs[None, :]
        xs = xs / xs.sum(axis=1, keepdims=True)
        return xs

    @property
    def xs(self):
        return self._xs_completo

    @property
    def ws(self):
        return self._ws_completo

    @property
    def phis(self):
        return self._phis_completo

    @staticmethod
    def _formato_sara(composição):
        if composição is None:
            return None
        reduzida = np.zeros((composição.shape[0], 5))
        reduzida[:, :4] = composição[:, :4]
        reduzida[:, 4] = composição[:, 4:].sum(axis=1)
        return reduzida

    @staticmethod
    def _formato_simples(composição):
        if composição is None:
            return None
        return np.column_stack((composição[:, 0], composição[:, 1:].sum(axis=1)))

    def formato_reduzido(self, composição, formato='SARA'):
        match formato.lower():
            case "sara":
                return self._formato_sara(composição)
            case "simples":
                return self._formato_simples(composição)
            case _:
                raise ValueError("Erro de formato. Formato deve ser 'SARA' ou 'simples'.")

    def definir_composições(self, tipo, frações_base, propriedades):

        if frações_base.shape[-1] != propriedades.n_componentes:
            raise ValueError("Incompatibilidade do array de composições e de propriedadas")

        if frações_base.sum() == 0:
            self._xs_completo = frações_base
            self._ws_completo = frações_base
            self._phis_completo = frações_base
        else:
            match tipo:
                case 'molar': self._xs_completo = frações_base
                case 'massa': self._xs_completo = self.converter_massa_mol(frações_base, propriedades.MMs)
                case 'volume': self._xs_completo = self.converter_volume_mol(frações_base, propriedades.Vs)
                case _: raise ValueError(f'Tipo de fração ({tipo}) inexistente.')

            self._ws_completo = self.converter_mol_massa(self.xs, propriedades.MMs)
            self._phis_completo = self.converter_mol_volume(self.xs, propriedades.Vs)

    def validar(self):
        somas = self._xs_completo.sum(axis=1)
        if np.any(np.abs(1.0 - somas) > 1e-6):
            raise ValueError(f"Problema na composição. Frações não somam 100%.")


@dataclass
class FasesSistema:

    betas: np.ndarray | None = None

    fase_global: Composição = field(default_factory=Composição)
    fase_leve: Composição = field(default_factory=Composição)
    fase_pesada: Composição = field(default_factory=Composição)

    @classmethod
    def criar(cls, ws_exp, SARA, n_dados, propriedades):
        obj = cls()
        fração_global = cls._a_partir_de_composição_SARA(ws_exp, SARA, propriedades)
        obj.fase_global.definir_composições('massa', fração_global, propriedades)
        obj.fase_leve.definir_composições('massa', np.zeros((n_dados, propriedades.n_componentes)), propriedades)
        obj.fase_pesada.definir_composições('massa', np.zeros((n_dados, propriedades.n_componentes)), propriedades)
        obj.betas = np.zeros(n_dados)
        return obj

    @property
    def n_dados(self):
        return 0 if self.fase_global is None else self.fase_global.xs.shape[0]

    @property
    def n_componentes(self):
        return 0 if self.fase_global is None else self.fase_global.xs.shape[1]

    def validação(self):
        self.fase_global.validar()
        self.fase_leve.validar()
        self.fase_pesada.validar()

    @staticmethod
    def _a_partir_de_composição_SARA(ws_exp, SARA, propriedades):
        """ Fragmenta a composição do sistema de [Solvente, Petróleo] para [Solvente, S, A, R, Asf0, Asf1, ...]

        Inputs:
            ws_exp (array)      : composição global do sistema em termos de [Solvente, Petróleo] (base mássica)
            SARA (array)        : composição SARA do petróleo (base mássica)
            propriedades (obj)  : objeto com as propriedades termodinâmicas do sistema (MMs, rhos, deltas, MMs)

        Outputs:
            ws_completo (array) : composição global mássica do sistema em termos de
                                  [Precipitante, S, A, R, Asf0, Asf1, ...]
        """

        # Inicialização de arrays importantes
        n_dados_exp = ws_exp.shape[0]
        n_agregados = propriedades.asfaltenos.w.shape[0]
        ws_criado = np.zeros((n_dados_exp, 4 + n_agregados))

        ws_criado[:, 0] = ws_exp[:, 0]  # Precipitante
        ws_criado[:, 1:4] = ws_exp[:, [1]] * SARA[:3]  # Saturados, Aromáticos e Resinas
        ws_criado[:, 4:] = (ws_exp[:, [1]] * SARA[3] * propriedades.asfaltenos.w)  # Agregados de Asfaltenos

        return ws_criado

    def yields_calc(self, MMs):
        # Massa molar média de cada fase (sem precipitante, apenas petróleo)
        MM_petróleo_L = (self.fase_leve.xs[:, 1:] * MMs[1:]).sum(axis=1)
        MM_petróleo_H = (self.fase_pesada.xs[:, 1:] * MMs[1:]).sum(axis=1)

        # Massa de cada fase (sem precipitante, apenas petróleo)
        m_petróleo_L = (1 - self.betas) * MM_petróleo_L
        m_petróleo_H = self.betas * MM_petróleo_H

        return m_petróleo_H / (m_petróleo_L + m_petróleo_H)


# Função
def normalizar_composição(frações):
    soma = frações.sum(axis=-1, keepdims=True)
    frações_normalizadas = np.divide(frações, soma, out=np.zeros_like(frações), where=(soma > 1e-12))
    return frações_normalizadas
