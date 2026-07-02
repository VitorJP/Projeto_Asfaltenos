# Importação de bibliotecas do python
import numpy as np
import scipy as scp
from dataclasses import dataclass, field, fields

# Importação de módulos internos
from módulo_otimização import regressão_equilíbrio, regressão_cinética
from módulo_leitura_dados import ler_lista_datasets


@dataclass
class BaseParâmetros:
    def validar(self):
        for f in fields(self):
            valor = getattr(self, f.name)
            limites = f.metadata.get('limites', None)
            if f.type is float:
                valor = float(valor)
                setattr(self, f.name, valor)
            if f.type is not None and not isinstance(valor, f.type):
                raise TypeError(f"[{f.name}] Tipo inválido: esperado {f.type}. Recebido {type(valor)}")
            if limites is not None:
                inferior, superior = limites
                if valor < inferior or valor > superior:
                    raise ValueError(f"[{f.name}] Valor {valor} fora dos limites {limites}")

    @property
    def variáveis_regressão(self):
        return [f.name for f in fields(self) if f.metadata.get('regredir', True)]

    @property
    def limites_regressão(self):
        return [f.metadata.get('limites') for f in fields(self) if f.metadata.get('regredir', True)]

    @property
    def x0(self):
        return np.array([getattr(self, f.name) for f in fields(self) if f.metadata.get('regredir', False)], dtype=float)

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
    n_agregados: int = field(default=30, metadata={'regredir': False,
                                                   'limites': (1, 30)})
    MW_min: float = field(default=750, metadata={'regredir': False,
                                                 'limites': (600, 1800)})
    MW_max: float = field(default=30000, metadata={'regredir': True,
                                                   'limites': (5000, 30000)})
    MW_avg: float = field(default=3620, metadata={'regredir': True,
                                                  'limites': (1800, 10000)})
    alfa: float = field(default=3.5, metadata={'regredir': True,
                                                'limites': (0.1, 80)})

    # Correlação Barrera
    A_Barrera: float = field(default=0.0, metadata={'regredir': False,
                                                    'limites': (0.0, 1.0)})
    c_Barrera: float = field(default=0.647, metadata={'regredir': False,
                                                      'limites': (0.0, 1.0)})
    d_Barrera: float = field(default=0.0495, metadata={'regredir': False,
                                                       'limites': (0.0, 1.0)})

    def regredir(self, dados_experimentais, propriedades, config):
        args = (self, dados_experimentais, propriedades, config)
        sol = scp.optimize.minimize(fun=regressão_equilíbrio, x0=self.x0, bounds=self.limites_regressão, args=args,
                                    method=config.cálculo.algoritmo_otimização)
        self.atualizar(self.variáveis_regressão, sol.x)


@dataclass
class ParâmetrosCinética(BaseParâmetros):
    # Parâmetros experimentais
    onset: float = field(default=0.5, metadata={'regredir': False,
                                                'limites': (0.0, 0.99)})
    yield_max: float = field(default=0.10, metadata={'regredir': False,
                                                     'limites': (0.0, 0.99)})

    # Parâmetros de ajuste da curva de equilíbrio com a curva de maior tempo
    kc1: float = field(default=680.0, metadata={'regredir': True,
                                                'limites': (0.0, np.inf)})
    kc2: float = field(default=0.030, metadata={'regredir': True,
                                                'limites': (0.0, np.inf)})

    # Parâmetros temporais
    kt1: float = field(default=5.0, metadata={'regredir': True,
                                              'limites': (0.0, np.inf)})
    kt2: float = field(default=-2.0, metadata={'regredir': True,
                                               'limites': (-np.inf, np.inf)})

    def regredir(self, dados_experimentais, yields_eq, config):
        args = (self, dados_experimentais, yields_eq)
        sol = scp.optimize.minimize(fun=regressão_cinética, x0=self.x0, bounds=self.limites_regressão, args=args,
                                    method=config.cálculo.algoritmo_otimização)
        self.atualizar(self.variáveis_regressão, sol.x)


@dataclass
class Parâmetros:
    equilíbrio: ParâmetrosEquilíbrio = field(
        default_factory=ParâmetrosEquilíbrio
    )

    cinética: ParâmetrosCinética = field(
        default_factory=ParâmetrosCinética
    )

    @classmethod
    def inicializar(cls):
        obj = cls()
        cls.equilíbrio.validar()
        cls.cinética.validar()
        return obj
