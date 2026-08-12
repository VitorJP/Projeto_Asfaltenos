# Importação de bibliotecas do python
import numpy as np
import scipy as scp
from scipy.constants import R  # m3*Pa/mol*K
from itertools import combinations
from dataclasses import dataclass
from typing import ClassVar
# import chemicals

# Importação de módulos internos
from módulo_composições import normalizar_composição, FasesSistema
from módulo_propriedades import Propriedades


def gerar_conjunto_composições_candidatas(composição_global, composição_asfaltenos):

    def _asfaltenos_puro(zs, xs_asfaltenos):  # 1 candidata
        composição = np.zeros_like(zs)
        composição[4:] = xs_asfaltenos
        return [composição]

    def _vértices(zs):  # N candidatas
        n_componentes = zs.shape[0]
        lista_composições = []
        for z in range(n_componentes):
            if zs[z] != 0:
                composição = np.zeros_like(zs)
                composição[z] = 1
                lista_composições.append(composição)
        return lista_composições

    def _pares_binários(zs):  # 3*N*(N-1)/2 candidatas
        n_componentes = zs.shape[0]
        lista_composições = []
        for i, j in combinations(range(n_componentes), 2):
            if zs[i] != 0 and zs[j] != 0:
                # (50% , 50%)
                composição = np.zeros_like(zs)
                composição[i], composição[j] = 0.5, 0.5
                lista_composições.append(composição)
                # (25%, 75%)
                composição = np.zeros_like(zs)
                composição[i], composição[j] = 0.25, 0.75
                lista_composições.append(composição)
                # (75%, 25%)
                composição = np.zeros_like(zs)
                composição[i], composição[j] = 0.75, 0.25
                lista_composições.append(composição)
        return lista_composições

    def _perturbação_na_fase_global(zs, escalas=(0.01, 0.05, 0.2), seed=59):  # 3*N partículas
        n_componentes = zs.shape[0]
        rng = np.random.default_rng(seed)
        lista_composições = []
        for escala in escalas:
            for _ in range(n_componentes):
                perturbação = rng.normal(loc=0.0, scale=escala, size=n_componentes)
                composição = zs * (1.0 + perturbação)
                lista_composições.append(composição)
        return lista_composições

    def _distribuição_dirichlet(zs, alfas=(0.1, 1.0, 5.0), seed=59):  # 9*N partículas
        rng = np.random.default_rng(seed)
        n_componentes = zs.shape[0]
        lista_composições = []
        tamanho_lista = 3 * n_componentes
        for a in alfas:
            alfa = np.ones_like(zs) * a
            composições_dirichlet = rng.dirichlet(alfa, size=tamanho_lista)
            for composição in composições_dirichlet:
                composição = normalizar_composição(composição)
                lista_composições.append(composição)
        return lista_composições

    lista_candidatas = []
    lista_candidatas.extend(_asfaltenos_puro(composição_global, composição_asfaltenos))
    lista_candidatas.extend(_vértices(composição_global))
    lista_candidatas.extend(_pares_binários(composição_global))
    lista_candidatas.extend(_perturbação_na_fase_global(composição_global))
    lista_candidatas.extend(_distribuição_dirichlet(composição_global))
    return np.array(lista_candidatas)


@dataclass
class ModeloTermodinâmico:

    propriedades: Propriedades
    _modelos: ClassVar[dict[str, type]] = {}

    @classmethod
    def registrar(cls, nome):
        def decorator(subclasse):
            cls._modelos[nome] = subclasse
            return subclasse
        return decorator

    @classmethod
    def criar(cls, nome, *args, **kwargs):
        return cls._modelos[nome](*args, **kwargs)

    @staticmethod
    def _ln_seguro(xs):
        ln_seguro = np.zeros_like(xs)
        mask = xs > 0
        ln_seguro[mask] = np.log(xs[mask])
        ln_seguro[~mask] = 0.0
        return ln_seguro

    def tpd(self, fase_global, fase_nova):
        ln_fase_global = self._ln_seguro(fase_global.xs)
        ln_fase_nova = self._ln_seguro(fase_nova.xs)
        return np.sum(fase_nova.xs * (ln_fase_nova + self.ln_coeficiente(fase_nova)
                                      - ln_fase_global - self.ln_coeficiente(fase_global)))

    def delta_G_mix(self, sistema):
        G_fase_leve = (1 - sistema.betas[None, :]) * self.tpd(sistema.fase_global, sistema.fase_leve)
        G_fase_pesada = sistema.betas[None, :] * self.tpd(sistema.fase_global, sistema.fase_pesada)
        return G_fase_leve + G_fase_pesada


@ModeloTermodinâmico.registrar("Flory-Huggins")
class FloryHuggins(ModeloTermodinâmico):

    def ln_coeficiente(self, xs):
        if np.all(xs == 0):
            return np.zeros_like(xs)
        else:
            # Volume Médio da Fase:
            Vm = (xs * self.propriedades.Vs).sum()

            # Parâmetro de Solubilidade Médio da Fase:
            phis_fase = (xs * self.propriedades.Vs) / Vm
            delta_fase = (phis_fase * self.propriedades.deltas).sum()

            termo_1 = self.propriedades.Vs / Vm
            termo_2 = np.log(termo_1)
            termo_3 = (self.propriedades.Vs / (R * self.propriedades.T)) * (
                    (self.propriedades.deltas - delta_fase) ** 2)
            return (1 - termo_1) + termo_2 + termo_3


class FlashELL:

    _beta: float | np.ndarray = 0.0
    _zs: np.ndarray
    _ws: np.ndarray
    _Ks: np.ndarray

    # Configurações
    _tol: float = 1e-8
    _termodinâmica: ModeloTermodinâmico

    def __init__(self, composição_global, modelo, chute_asfaltênico=True):
        self._termodinâmica = modelo
        self._zs = composição_global
        self._ws = self._chute_inicial__ws(chute_asfaltênico)
        self._Ks = self._chute_inicial__Ks()
        self.calcular_beta_rachford_rice()

    def _chute_inicial__ws(self, chute_asfaltênico):
        if chute_asfaltênico:
            chute = np.zeros_like(self._zs)
            chute[4:] = self._termodinâmica.propriedades.asfaltenos.x
            return normalizar_composição(chute)
        else:
            raise ValueError('Método de Múltiplos Chutes não implementado.')

    def _chute_inicial__Ks(self):
        Ks_inicial = np.empty_like(self._ws)
        mask = self._zs > self._tol
        Ks_inicial[mask] = self._ws[mask] / self._zs[mask]
        Ks_inicial[~mask] = 0.0
        return Ks_inicial

    def equação_rachford_rice(self, beta_rr):
        return (self._zs * (self._Ks - 1.0) / (1.0 + beta_rr * (self._Ks - 1.0))).sum()

    @property
    def xs_L(self):
        novas_xs_L = np.clip(self._zs / (1 + self._beta * (self._Ks - 1)), 0, 1)
        novas_xs_L[novas_xs_L < self._tol] = 0.0
        return normalizar_composição(novas_xs_L)

    @ property
    def xs_H(self):
        novas_xs_H = np.clip(self._Ks * self._zs / (1 + self._beta * (self._Ks - 1)), 0, 1)
        novas_xs_H[novas_xs_H < self._tol] = 0.0
        return normalizar_composição(novas_xs_H)

    def forma_duas_fases(self):
        beta_min, beta_max = self._tol, 1.0 - self._tol
        return True if self.equação_rachford_rice(beta_min) * self.equação_rachford_rice(beta_max) < 0 else False

    def calcular_beta_rachford_rice(self):
        beta_min, beta_max = self._tol, 1.0 - self._tol
        if self.forma_duas_fases():
            novo_beta = scp.optimize.brentq(self.equação_rachford_rice, beta_min, beta_max, xtol=self._tol)
            self._beta = np.clip(novo_beta, beta_min, beta_max)
        else:
            self._beta = 1.0 if self.equação_rachford_rice(beta_min) > 0 else 0.0

    def calcular_flash(self, erro=0, n_it=1, n_it_max=100):
        for n_it in range(1, n_it_max+1):
            # Aplicação do equilíbrio termodinâmico de fases
            novos_ln_Ks = self._termodinâmica.ln_coeficiente(self.xs_L) - self._termodinâmica.ln_coeficiente(self.xs_H)
            novos_Ks = np.where(self._zs == 0, 1.0, np.exp(novos_ln_Ks))
            novos_Ks[self._Ks < self._tol] = 0.0

            # Critério de convergência
            erro = np.sum(np.abs(novos_Ks - self._Ks) / (1 + self._Ks))
            self._Ks = novos_Ks
            self.calcular_beta_rachford_rice()

            if erro <= self._tol:
                break
        else:
            print(f"O método de Rachford-Rice não convergiu em {n_it_max} iterações. O erro final foi {erro}.")

        return self._beta, self.xs_L, self.xs_H, n_it
