# Importação de bibliotecas do python
import numpy as np
import scipy as scp
from dataclasses import dataclass, field, fields

# Importação de módulos internos
from módulo_otimização import regressão_equilíbrio, regressão_cinética
from módulo_leitura_dados import ler_lista_datasets


@dataclass
class BaseConfigurações:
    def validar(self):
        for f in fields(self):
            opt = getattr(self, f.name)
            opções = f.metadata.get('opções')
            if opções is None:
                continue
            if opt not in opções:
                raise ValueError(f"Erro em '{f.name}': opção '{opt}' inválida. Opções válidas: \n{opções}")

    def atualizar(self, variáveis, valores):
        for variável, valor in zip(variáveis, valores):
            setattr(self, variável, valor)
        validar()


@dataclass
class CorrelaçõesConfigurações(BaseConfigurações):

    delta_precipitante: str = field(default='Akbarzadeh',
                                    metadata={'opções': ['Akbarzadeh', 'Vargas']})

    densidade_saturados: str = field(default='Akbarzadeh',
                                     metadata={'opções': ['Akbarzadeh', 'Alves', 'Yanes']})

    delta_saturados: str = field(default='Akbarzadeh',
                                 metadata={'opções': ['Akbarzadeh', 'Tharanivasan', 'Yanes']})

    densidade_aromáticos: str = field(default='Akbarzadeh',
                                      metadata={'opções': ['Alves', 'Akbarzadeh', 'Yanes']})

    delta_aromáticos: str = field(default='Akbarzadeh',
                                  metadata={'opções': ['Akbarzadeh', 'Yanes']})

    densidade_resinas: str = field(default='Akbarzadeh',
                                   metadata={'opções': ['Akbarzadeh', 'Yanes']})

    delta_resinas: str = field(default='Akbarzadeh',
                               metadata={'opções': ['Akbarzadeh', 'Yanes']})

    densidade_asfaltenos: str = field(default='Alboudwarej',
                                      metadata={'opções': ['Akbarzadeh', 'Alboudwarej', 'Barrera']})

    delta_asfaltenos: str = field(default='Tharanivasan',
                                  metadata={'opções': ['Tharanivasan', 'Barrera']})


@dataclass
class CálculosConfigurações(BaseConfigurações):
    tipo_cálculo_equilíbrio: str = field(default='regressão',
                                         metadata={'opções': ['regressão', 'predição']})

    tipo_cálculo_cinética: str = field(default='não',
                                       metadata={'opções': ['não', 'regressão', 'predição']})

    algoritmo_otimização: str = field(default='Nelder-Mead',
                                      metadata={'opções': ['Nelder-Mead', 'L-BFGS-B', 'Powell']})

    tipo_cálculo_MM_agregados: str = field(default='superior',
                                           metadata={'opções': ['superior', 'médio']})

    método_integração_FDP_Gamma: str = field(default='trapezios',
                                             metadata={'opções': ['quadratura', 'trapezios']})

    x_yield_curve: str = field(default='massa',
                               metadata={'opções': ['massa', 'molar', 'volume', 'solubilidade']})

    plotar_gráficos: bool = field(default=True,
                                  metadata={'opções': [True, False]})


@dataclass
class Configurações:
    cálculo: CálculosConfigurações = field(default_factory=CálculosConfigurações)
    correlações: CorrelaçõesConfigurações = field(default_factory=CorrelaçõesConfigurações)

    @classmethod
    def inicializar(cls):
        obj = cls()
        cls.cálculo.validar()
        cls.correlações.validar()
        return obj
