# Importação de bibliotecas do python
import numpy as np
import scipy as scp
import pandas as pd
from dataclasses import dataclass, field, fields
from copy import deepcopy

# Importação de módulos internos
from módulo_leitura_dados import ler_lista_datasets
from módulo_composições import FasesSistema
from módulo_ELL import FlashELL, ModeloTermodinâmico
from módulo_equilíbrio_líquido_líquido import calcular_ELL
from módulo_cálculos_cinéticos import calcular_yields_tempo_infinito, calcular_yields_temporais
from módulo_resultados import average_absolute_deviation


@dataclass
class BaseParâmetros:
    def validar(self):
        for f in fields(self):
            valor = getattr(self, f.name)
            if f.type is float:
                valor = float(valor)
                setattr(self, f.name, valor)
            if f.type is not None and not isinstance(valor, f.type):
                raise TypeError(f"[{f.name}] Tipo inválido: esperado {f.type}. Recebido {type(valor)}")

            limites = self._limites_variável(f)
            if limites is not None:
                inferior, superior = limites
                if valor < inferior or valor > superior:
                    raise ValueError(f"[{f.name}] Valor {valor} fora dos limites {limites}")

    @property
    def variáveis_regressão(self):
        return [f.name for f in fields(self) if f.metadata.get('regredir', True)]

    def _limites_variável(self, variável):
        inferior, superior = variável.metadata.get("limites")

        if callable(inferior):
            inferior = inferior(self)

        if callable(superior):
            superior = superior(self)

        return inferior, superior

    @property
    def limites_regressão(self):
        return [self._limites_variável(f) for f in fields(self) if f.metadata.get("regredir", True)]

    @property
    def valores_regressão(self):
        return np.array([getattr(self, f.name) for f in fields(self) if f.metadata.get('regredir', True)], dtype=float)

    def atualizar(self, variáveis, valores):
        fields_dict = {f.name: f for f in fields(self)}
        for variável, valor in zip(variáveis, valores):
            f = fields_dict[variável]
            if f.type is int:
                valor = int(round(valor))
            elif f.type is float:
                valor = float(valor)
            setattr(self, variável, valor)


@dataclass
class ParâmetrosEquilíbrio(BaseParâmetros):
    # Distribuição Gamma
    n_agregados: int = field(default=30,
                             metadata={'regredir': False,
                                       'limites': (1, 30)})
    MW_min: float = field(default=750,
                          metadata={'regredir': False,
                                    'limites': (600, 1800)})
    MW_max: float = field(default=27000,
                          metadata={'regredir': False,
                                    'limites': (5000, 30000)})
    MW_avg: float = field(default=2755.63974554735,
                          metadata={'regredir': False,
                                    'limites': (1800, 5000)})
    alfa: float = field(default=20.0,
                        metadata={'regredir': False,
                                  'limites': (0.1, 20.0)})

    # Correlação Barrera
    A_Barrera: float = field(default=0.0,
                             metadata={'regredir': False,
                                       'limites': (0.0, 1.0)})
    c_Barrera: float = field(default=0.647,
                             metadata={'regredir': False,
                                       'limites': (0.0, 1.0)})
    d_Barrera: float = field(default=0.0495,
                             metadata={'regredir': False,
                                       'limites': (0.0, 1.0)})

    # Correlação Ramos-Pallares
    s_Dist_rho: float = field(default=9.0,
                              metadata={'regredir': False,
                                        'limites': (3.0, 10.0)})
    n_Dist_delta: float = field(default=1.2,
                                metadata={'regredir': False,
                                          'limites': (0.0, 3.0)})
    delta_min: float = field(default=19.3,
                             metadata={'regredir': False,
                                       'limites': (lambda self: 12.0,
                                                   lambda self: self.delta_max)})
    delta_max: float = field(default=21.5,
                             metadata={'regredir': False,
                                       'limites': (lambda self: self.delta_min,
                                                   lambda self: 30.0)})

    # Método de Yen-Mullins
    MW_molecula: float = field(default=670,
                               metadata={'regredir': True,
                                         'limites': (500, 1000)})
    n_nanoagregação: float = field(default=6.0,
                                   metadata={'regredir': False,
                                             'limites': (6.0, 10.0)})
    n_clusterização: float = field(default=6.0,
                                   metadata={'regredir': False,
                                             'limites': (6.0, 10.0)})
    x_Asf0: float = field(default=0.3,
                          metadata={'regredir': True,
                                    'limites': (0.0, 1.0)})
    x_Asf1: float = field(default=0.3,
                          metadata={'regredir': True,
                                    'limites': (0.0, 1.0)})

    @staticmethod
    def regressão_equilíbrio(valores_otimização, *args):
        # Desempacotando os *args
        params_base = args[0]
        SARA, precipitante, ws_exp, yields_exp, n_dados_exp = args[1]
        propriedades = deepcopy(args[2])
        config = args[3]

        # Alocando os valores dos parâmetros possíveis de regressão
        params = deepcopy(params_base)
        params.atualizar(params.variáveis_regressão, valores_otimização)

        # Propriedades dos agregados de asfaltenos e dos componentes do sistema
        propriedades.atualizar(params, config)

        # Composição global do sistema em termos de [Solvente, S, A, R, Asf0, Asf1, ...]
        sistema = FasesSistema.criar(ws_exp, SARA, n_dados_exp, propriedades)
        modelo_termodinâmico = ModeloTermodinâmico.criar(config.cálculo.modelo_termodinâmico, propriedades)
        xs_L, xs_H = np.zeros_like(sistema.fase_leve.xs), np.zeros_like(sistema.fase_pesada.xs)

        # Cálculo de Equilíbrio Líquido-Líquido
        for i in range(n_dados_exp):
            # sistema.betas[i], xs_L[i, :], xs_H[i, :], _ = calcular_ELL(sistema.fase_global.xs[i], propriedades)
            equilíbrio = FlashELL(sistema.fase_global.xs[i], modelo_termodinâmico)
            sistema.betas[i], xs_L[i, :], xs_H[i, :], _ = equilíbrio.calcular_flash()

        sistema.fase_leve.definir_composições('molar', xs_L, propriedades)
        sistema.fase_pesada.definir_composições('molar', xs_H, propriedades)

        # Expressão matemática a ser minimizada: diferenças entre os yields calculados e experimentais
        return max(np.abs(yields_exp - sistema.yields_calc(propriedades.MMs)))

    def regredir(self, dados_experimentais, propriedades, config):
        args = (self, dados_experimentais, propriedades, config)
        sol_global = scp.optimize.differential_evolution(self.regressão_equilíbrio,
                                                         bounds=self.limites_regressão, args=args,)
        sol_final = scp.optimize.minimize(fun=self.regressão_equilíbrio, x0=sol_global.x,
                                          bounds=self.limites_regressão, args=args,
                                          method=config.cálculo.algoritmo_otimização)
        # sol_final = scp.optimize.minimize(fun=self.regressão_equilíbrio, x0=self.valores_regressão,
        #                                   bounds=self.limites_regressão, args=args,
        #                                   method=config.cálculo.algoritmo_otimização)
        self.atualizar(self.variáveis_regressão, sol_final.x)


@dataclass
class ParâmetrosCinética(BaseParâmetros):
    # Parâmetros experimentais
    onset: float = field(default=0.5,
                         metadata={'regredir': False,
                                   'limites': (0.0, 0.99)})
    yield_max: float = field(default=0.10,
                             metadata={'regredir': False,
                                       'limites': (0.0, 0.99)})

    # Parâmetros de ajuste da curva de equilíbrio com a curva de maior tempo
    kc1: float = field(default=680.0,
                       metadata={'regredir': True,
                                 'limites': (0.0, np.inf)})
    kc2: float = field(default=0.030,
                       metadata={'regredir': True,
                                 'limites': (0.0, np.inf)})

    # Parâmetros temporais
    kt1: float = field(default=5.0,
                       metadata={'regredir': True,
                                 'limites': (0.0, np.inf)})
    kt2: float = field(default=-2.0,
                       metadata={'regredir': True,
                                 'limites': (-np.inf, np.inf)})

    @staticmethod
    def regressão_cinética(valores_otimização, *args):
        # Desempacotando os *args
        params_base = args[0]
        tempos, ws_precipitante, yields_t_exp = args[1]
        yields_eq = args[2]

        params = deepcopy(params_base)
        params.atualizar(params.variáveis_regressão, valores_otimização)

        # Cálculo das equações cinéticas
        yields_eq_calc = calcular_yields_tempo_infinito(ws_precipitante, params)
        yields_t_calc, _ = calcular_yields_temporais(tempos, ws_precipitante, yields_eq_calc, params)

        # Expressão matemática a ser minimizada: diferenças entre os yields calculados e experimentais
        return average_absolute_deviation(yields_eq, yields_eq_calc) + \
               average_absolute_deviation(yields_t_exp, yields_t_calc)

    def regredir(self, dados_experimentais, yields_eq, config):
        args = (self, dados_experimentais, yields_eq)
        sol = scp.optimize.minimize(fun=self.regressão_cinética, x0=self.valores_regressão, bounds=self.limites_regressão,
                                    args=args, method=config.cálculo.algoritmo_otimização)
        self.atualizar(self.variáveis_regressão, sol.x)


@dataclass
class Parâmetros:
    equilíbrio: ParâmetrosEquilíbrio = field(
        default_factory=ParâmetrosEquilíbrio
    )

    cinética: ParâmetrosCinética = field(
        default_factory=ParâmetrosCinética
    )

    def __post_init__(self):
        self.equilíbrio.validar()
        self.cinética.validar()
