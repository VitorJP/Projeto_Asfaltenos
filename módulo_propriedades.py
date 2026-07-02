# Importação de bibliotecas do python
import os
import numpy as np
import pandas as pd
from scipy.constants import R  # m3*Pa/mol*K
from dataclasses import dataclass, field, fields

# Importação de módulos internos
from módulo_distribuição_massa_molar import gerar_distribuição_massa_molar
from módulo_composições import normalizar_composição


@dataclass(frozen=True)
class PropriedadesHBT:
    MM: float
    Tc: float
    wSRK: float
    Vstar: float


@dataclass
class DataBaseHBT:
    _dados = None

    @classmethod
    def carregar_banco_de_dados(cls):
        if cls._dados is None:
            diretório = os.path.dirname(__file__)
            caminho_excel = os.path.join(diretório, "Dados de Entrada", "database_for_density_HBT.xlsx")
            df = pd.read_excel(caminho_excel, sheet_name="Data")
            cls._dados = {}

            for _, linha in df.iterrows():
                substância = linha.iloc[0].strip().lower()
                cls._dados[substância] = PropriedadesHBT(
                    MM=linha.iloc[1],
                    Tc=linha.iloc[2],
                    wSRK=linha.iloc[3],
                    Vstar=linha.iloc[4]
                )

    @classmethod
    def obter_dados(cls, nome):
        cls.carregar_banco_de_dados()
        chave = nome.strip().lower()
        if chave not in cls._dados:
            raise ValueError(f"Precipitante '{nome}' não encontrado no banco HBT.")
        return cls._dados[chave]

    @classmethod
    def listar_componentes(cls):
        cls.carregar()
        return list(cls._dados.keys())


@dataclass
class BasePropriedades:
    MM: np.ndarray | None = None
    rho: np.ndarray | None = None
    delta: np.ndarray | None = None
    V: np.ndarray | None = None


@dataclass
class ComponentePuro(BasePropriedades):
    # Definições
    nome: str | None = None
    correlação_delta: str | None = None

    # Propriedades para o modelo HBT
    Tc: float | None = None
    wSRK: float | None = None
    Vstar: float | None = None

    def carregar_parâmetros_HBT(self):
        dados = DataBaseHBT.obter_dados(self.nome)
        self.MM, self.Tc, self.wSRK, self.Vstar = dados.MM, dados.Tc, dados.wSRK, dados.Vstar

    def calcular_propriedades(self, T):
        """ Calcula as propriedades do precipitante/alcano na temperatura de interesse.

            Inputs:
                T (float)         : temperatura (K)

            Observações:
                Os parâmetros do modelo HBT para cada solvente foram extraídos do livro
                'The Properties of Gases and Liquids', de Reid, Prausnitz e Poling (1987)
                As correlações para 'delta' foram originalmente propostas por Akbarzadeh et al. (2005),
                conforme citado por Tharanivasan (2012)
            """

        # Propriedades
        self.carregar_parâmetros_HBT()
        self.densidade_HBT(T)  # kg/m³
        self.V = (self.MM / self.rho) * 1e3  # cm³/mol

        # Cálculo do parâmetro de solubilidade
        match self.correlação_delta:
            case 'Akbarzadeh':
                Delta_H_vap = 3492.8 + 276.54 * self.MM + 0.524 * (self.MM ** 2) if self.MM < 60 \
                    else 103.65 + 368.7 * self.MM - 0.0603 * (self.MM ** 2)
                delta_25C = ((Delta_H_vap - 298.15 * R) / self.V) ** 0.5
                self.delta = delta_25C - 0.0232 * (T - 298.15)  # MPa**0.5
            case 'Vargas':
                self.delta = 17.347 * (self.rho / 1000) + 2.904 if self.MM > 60 \
                    else 2.904 + 26.302 * (self.rho / 1000) - 20.5618 * ((self.rho / 1000) ** 2) + 12.0425 * (
                        (self.rho / 1000) ** 3)  # MPa**0.5
            case _:
                # Em caso de erro, usa-se Akbarzadeh como padrão.
                Delta_H_vap = 3492.8 + 276.54 * self.MM + 0.524 * (self.MM ** 2) if self.MM < 60 \
                    else 103.65 + 368.7 * self.MM - 0.0603 * (self.MM ** 2)
                delta_25C = ((Delta_H_vap - 298.15 * R) / self.V) ** 0.5
                self.delta = delta_25C - 0.0232 * (T - 298.15)  # MPa**0.5

        # Ajuste de unidades
        self.MM = self.MM * 1e-3  # kg/mol
        self.delta = self.delta * 1e3  # Pa**0.5
        self.V = self.MM / self.rho  # m³/mol

    def densidade_HBT(self, T):
        """ Calcula a densidade do solvente utilizando o modelo de Hankinson-Brobst-Thomson.

            Observações:
                A implementação foi baseada no equacionamento do livro 'The Properties of Gases and Liquids',
                de Reid, Prausnitz e Poling (1987)
        """

        # Variáveis auxiliares
        Tr = T / self.Tc
        aux = 1 - Tr

        # Cálculos
        a, b, c, d = -1.52816, 1.43907, -0.81446, 0.190454
        e, f, g, h = -0.296123, 0.386914, -0.0427258, -0.0480645
        Vr0 = 1 + a * aux ** (1 / 3) + b * aux ** (2 / 3) + c * aux + d * aux ** (4 / 3)
        Vr1 = (e + f * Tr + g * Tr ** 2 + h * Tr ** 3) / (Tr - 1.00001)

        self.V = self.Vstar * (Vr0 * (1 - self.wSRK * Vr1))
        self.rho = self.MM / self.V


@dataclass
class Saturados(BasePropriedades):
    # Correlações
    correlação_densidade: str = ''
    correlação_delta: str = ''

    def calcular_propriedades(self, T):
        """ Calcula as propriedades dos saturados na temperatura de interesse.

        Inputs:
            T (float)   : temperatura (K)
        """

        # Massa molar (g/mol)
        self.MM = 460

        # Densidade (kg/m³)
        match self.correlação_densidade:
            case "Alves":
                self.rho = 1069.54 - 0.6379 * T
            case "Akbarzadeh":
                self.rho = 1078.96 - 0.6379 * T
            case "Yanes":
                self.rho = 880.0
            case _:
                self.rho = 1078.96 - 0.6379 * T  # Em caso de erro, usa-se Akbarzadeh como padrão.

        # Parâmetro de solubilidade (MPa**0.5)
        match self.correlação_delta:
            case "Akbarzadeh":
                self.delta = 22.381 - 0.0222 * T
            case "Tharanivasan":
                self.delta = 23.021 - 0.0222 * T
            case "Yanes":
                self.delta = 16.4
            case _:
                self.delta = 22.381 - 0.0222 * T  # Em caso de erro, usa-se Akbarzadeh como padrão.

        # Ajuste de unidades
        self.MM = self.MM * 1e-3  # kg/mol
        self.delta = self.delta * 1e3  # Pa**0.5

        # Volume molar (m³/mol)
        self.V = self.MM / self.rho


@dataclass
class Aromáticos(BasePropriedades):
    # Correlações
    correlação_densidade: str = ''
    correlação_delta: str = ''

    def calcular_propriedades(self, T):
        """ Calcula as propriedades dos aromáticos na temperatura de interesse.

        Inputs:
            T (float)                                : temperatura (K)
        """

        # Massa molar (g/mol)
        self.MM = 522

        # Densidade (kg/m³)
        match self.correlação_densidade:
            case "Alves":
                self.rho = 1164.73 - 0.5942 * T
            case "Akbarzadeh":
                self.rho = 1184.47 - 0.5942 * T
            case "Yanes":
                self.rho = 990.0
            case _:
                self.rho = 1184.47 - 0.5942 * T  # Em caso de erro, usa-se Akbarzadeh como padrão.

        # Parâmetro de solubilidade (MPa**0.5)
        match self.correlação_delta:
            case "Akbarzadeh":
                self.delta = 26.333 - 0.0204 * T
            case "Yanes":
                self.delta = 20.3
            case _:
                self.delta = 26.333 - 0.0204 * T  # Em caso de erro, usa-se Akbarzadeh como padrão.

        # Ajuste de unidades
        self.MM = self.MM * 1e-3  # kg/mol
        self.delta = self.delta * 1e3  # Pa**0.5

        # Volume molar (m³/mol)
        self.V = self.MM / self.rho


@dataclass
class Resinas(BasePropriedades):
    # Correlações
    correlação_densidade: str = ''
    correlação_delta: str = ''

    def calcular_propriedades(self, T):
        """ Calcula as propriedades das resinas na temperatura de interesse.

        Inputs:
            T (float)                             : temperatura (K)
        """

        # Massa molar (g/mol)
        self.MM = 1040

        # Densidade (kg/m³)
        match self.correlação_densidade:
            case "Yanes":
                self.rho = 1044.0
            case "Akbarzadeh":
                self.rho = 670 * (self.MM ** 0.0639)
            case _:
                self.rho = 1044.0  # Em caso de erro, usa-se Yanes como padrão.

        # Parâmetro de solubilidade (MPa**0.5)
        match self.correlação_delta:
            case "Yanes":
                self.delta = 19.3
            case "Akbarzadeh":
                A = 0.579 - 0.00075 * T
                self.delta = (A * self.rho) ** 0.5
            case _:
                self.delta = 19.3  # Em caso de erro, usa-se Yanes como padrão.

        # Ajuste de unidades
        self.MM = self.MM * 1e-3  # kg/mol
        self.delta = self.delta * 1e3  # Pa**0.5

        # Volume molar (m³/mol)
        self.V = self.MM / self.rho


@dataclass
class AgregadosAsfaltenos(BasePropriedades):
    # Correlações
    correlação_densidade: str = ''
    correlação_delta: str = ''

    # Fracionamento
    w: np.ndarray | None = None
    x: np.ndarray | None = None

    @property
    def n_agregados(self):
        return 0 if self.w is None else self.w.shape[0]

    def calcular_propriedades(self, T, params, config):
        # Massa molar (g/mol), fração mássica, fração molar
        self.MM, self.w, self.x = gerar_distribuição_massa_molar(
            params.alfa, params.MW_avg, params.n_agregados, params.MW_min, params.MW_max,
            config.cálculo.tipo_cálculo_MM_agregados, config.cálculo.método_integração_FDP_Gamma
        )
        self.w, self.x = normalizar_composição(self.w), normalizar_composição(self.x)

        # Densidade (kg/m³)
        match self.correlação_densidade:
            case "Akbarzadeh":
                self.rho = 670 * (self.MM ** 0.0639)
            case "Alboudwarej":
                self.rho = (self.MM / (1.493 * self.MM ** 0.936)) * 1e3
            case "Barrera":
                self.rho = 1100 + 100 * (1 - np.exp(-self.MM / 3850))
            case _:
                self.rho = (self.MM / (1.493 * self.MM ** 0.936)) * 1e3
            # Em caso de erro, usa-se Alboudwarej como padrão.

        # Parâmetro de solubilidade (MPa**0.5)
        match self.correlação_delta:
            case "Barrera":
                A = 0.579 - 0.00075 * T + params.A_Barrera
                self.delta = np.sqrt(A * self.rho * params.c_Barrera * self.MM ** params.d_Barrera)
            case "Tharanivasan":
                A = 0.579 - 0.00075 * T
                self.delta = np.sqrt(A * self.rho)
            case _:
                A = 0.579 - 0.00075 * T
                self.delta = np.sqrt(A * self.rho)
                # Em caso de erro, usa-se Tharanivasan como padrão.

        # Ajuste de unidades
        self.MM = self.MM * 1e-3  # kg/mol
        self.delta = self.delta * 1e3  # Pa**0.5

        # Volumes molares (m³/mol)
        self.V = self.MM / self.rho


@dataclass
class Propriedades:
    precipitante: ComponentePuro = field(default_factory=ComponentePuro)
    saturados: Saturados = field(default_factory=Saturados)
    aromáticos: Aromáticos = field(default_factory=Aromáticos)
    resinas: Resinas = field(default_factory=Resinas)
    asfaltenos: AgregadosAsfaltenos = field(default_factory=AgregadosAsfaltenos)

    @property
    def n_componentes(self):
        return 0 if self.MMs is None else self.MMs.shape[0]

    @property
    def MMs(self):
        if self.asfaltenos.w is None:
            return np.array([self.precipitante.MM, self.saturados.MM, self.aromáticos.MM, self.resinas.MM])
        else:
            return np.concatenate(
                [np.array([self.precipitante.MM, self.saturados.MM, self.aromáticos.MM, self.resinas.MM]),
                 self.asfaltenos.MM])

    @property
    def rhos(self):
        if self.asfaltenos.w is None:
            return np.array([self.precipitante.rho, self.saturados.rho, self.aromáticos.rho, self.resinas.rho])
        else:
            return np.concatenate(
                [np.array([self.precipitante.rho, self.saturados.rho, self.aromáticos.rho, self.resinas.rho]),
                 self.asfaltenos.rho])

    @property
    def deltas(self):
        if self.asfaltenos.w is None:
            return np.array([self.precipitante.delta, self.saturados.delta, self.aromáticos.delta, self.resinas.delta])
        else:
            return np.concatenate(
                [np.array([self.precipitante.delta, self.saturados.delta, self.aromáticos.delta, self.resinas.delta]),
                 self.asfaltenos.delta])

    @property
    def Vs(self):
        if self.asfaltenos.w is None:
            return np.array([self.precipitante.V, self.saturados.V, self.aromáticos.V, self.resinas.V])
        else:
            return np.concatenate(
                [np.array([self.precipitante.V, self.saturados.V, self.aromáticos.V, self.resinas.V]),
                 self.asfaltenos.V])

    def calcular_propriedades(self, T, nome_precipitante, config):

        # Declaração das correlações aplicadas
        self.precipitante.nome = nome_precipitante
        self.precipitante.correlação_delta = config.correlações.delta_precipitante
        self.saturados.correlação_densidade = config.correlações.densidade_saturados
        self.saturados.correlação_delta = config.correlações.delta_saturados
        self.aromáticos.correlação_densidade = config.correlações.densidade_aromáticos
        self.aromáticos.correlação_delta = config.correlações.delta_aromáticos
        self.resinas.correlação_densidade = config.correlações.densidade_resinas
        self.resinas.correlação_delta = config.correlações.delta_resinas
        self.asfaltenos.correlação_densidade = config.correlações.densidade_asfaltenos
        self.asfaltenos.correlação_delta = config.correlações.delta_asfaltenos

        # Cálculo das propriedades dos pseudocomponentes
        self.precipitante.calcular_propriedades(T)
        self.saturados.calcular_propriedades(T)
        self.aromáticos.calcular_propriedades(T)
        self.resinas.calcular_propriedades(T)

    @classmethod
    def inicializar(cls, T, precipitante, config):
        obj = cls()
        obj.calcular_propriedades(T, precipitante, config)
        return obj

    def adicionar_asfaltenos(self, T, params_asfaltenos, config):
        self.asfaltenos.calcular_propriedades(T, params_asfaltenos, config)
