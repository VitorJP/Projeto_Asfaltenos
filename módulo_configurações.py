# Importação de bibliotecas do python
import numpy as np
import scipy as scp
from dataclasses import dataclass, field, fields

# Importação de módulos internos
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
        self.validar()


@dataclass
class CorrelaçõesConfigurações(BaseConfigurações):
    # Correlações de Densidade
    densidade_precipitante: str = field(default='HBT',
                                        metadata={'opções': ['HBT']})
    densidade_saturados: str = field(default='Ramos_Pallares',
                                     metadata={'opções': ['Akbarzadeh', 'Alves', 'Yanes', 'Ramos_Pallares']})
    densidade_aromáticos: str = field(default='Ramos_Pallares',
                                      metadata={'opções': ['Alves', 'Akbarzadeh', 'Yanes', 'Ramos_Pallares']})
    densidade_resinas: str = field(default='Ramos_Pallares',
                                   metadata={'opções': ['Akbarzadeh', 'Yanes', 'Ramos_Pallares']})
    densidade_asfaltenos: str = field(default='Akbarzadeh',
                                      metadata={'opções': ['Akbarzadeh', 'Barrera', 'Ramos_Pallares']})

    # Correlações de Parâmetro de Solubilidade
    delta_precipitante: str = field(default='Akbarzadeh',
                                    metadata={'opções': ['Akbarzadeh', 'Vargas']})
    delta_saturados: str = field(default='Akbarzadeh',
                                 metadata={'opções': ['Akbarzadeh', 'Tharanivasan', 'Yanes', 'Ramos_Pallares']})
    delta_aromáticos: str = field(default='Akbarzadeh',
                                  metadata={'opções': ['Akbarzadeh', 'Yanes', 'Ramos-Pallares']})
    delta_resinas: str = field(default='Akbarzadeh',
                               metadata={'opções': ['Akbarzadeh', 'Yanes', 'Ramos_Pallares']})
    delta_asfaltenos: str = field(default='Tharanivasan',
                                  metadata={'opções': ['Tharanivasan', 'Barrera', 'Ramos_Pallares']})

    # Correlação para o Fracionamento dos Asfaltenos
    fracionamento_asfaltenos: str = field(default='Yen_Mullins',
                                          metadata={'opções': ['Distribuição_Gamma', 'Yen_Mullins']})


@dataclass
class CálculosConfigurações(BaseConfigurações):
    # Relacionados à regressão dos parâmetros
    tipo_cálculo_equilíbrio: str = field(default='predição',
                                         metadata={'opções': ['regressão', 'predição']})
    tipo_cálculo_cinética: str = field(default='não',
                                       metadata={'opções': ['não', 'regressão', 'predição']})
    algoritmo_otimização: str = field(default='Nelder-Mead',
                                      metadata={'opções': ['Nelder-Mead', 'L-BFGS-B', 'Powell']})

    # Relacionados à integração da distibuição gamma
    tipo_cálculo_MM_agregados: str = field(default='médio',
                                           metadata={'opções': ['superior', 'médio']})

    # Relacionado ao equilíbrio líquido-líquido
    modelo_termodinâmico: str = field(default='Flory-Huggins',
                                      metadata={'opções': ['Flory-Huggins']})

    # Relacionados à exibição dos resultados
    x_yield_curve: str = field(default='massa',
                               metadata={'opções': ['massa', 'molar', 'volume', 'solubilidade']})
    plotar_gráficos: bool = field(default=False,
                                  metadata={'opções': [True, False]})


@dataclass
class Configurações:
    cálculo: CálculosConfigurações = field(default_factory=CálculosConfigurações)
    correlações: CorrelaçõesConfigurações = field(default_factory=CorrelaçõesConfigurações)

    def __post_init__(self):
        self.cálculo.validar()
        self.correlações.validar()
