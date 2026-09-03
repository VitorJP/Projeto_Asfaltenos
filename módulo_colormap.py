# ====================================================================================================================== #
# Módulo: Colormap de Erro para Parâmetros alfa e MW_avg

# Importação das bibliotecas
import os
import numpy as np
import matplotlib.pyplot as plt
import copy
import sys

# Importação dos módulos
from módulo_configurações import Configurações
from módulo_parâmetros import Parâmetros
from módulo_propriedades import Propriedades
from módulo_composições import FasesSistema
from módulo_ELL import FlashELL, ModeloTermodinâmico
from módulo_leitura_dados import ler_dados_experimentais, ler_lista_datasets

# ==================================================================================================================== #
# PARTE 1: LEITURA DE DADOS E CONFIGURAÇÕES
# ==================================================================================================================== #

# Ler a lista de datasets
diretorio = os.path.dirname(__file__)
database = ler_lista_datasets(diretorio)

# Escolha do dataset
nome_planilha = 'Yanes_P3'  
if nome_planilha not in database:
    print(f"Dataset '{nome_planilha}' não encontrado na base de dados.")
    sys.exit()

# Leitura dos dados experimentais
diretorio_do_xlsx = os.path.join(diretorio, 'Dados de Entrada', 'database_yield_curves.xlsx')
SARA, T, precipitante, ws_exp, yields_exp, n_dados_exp = ler_dados_experimentais(diretorio_do_xlsx, nome_planilha)

# Configurações e parâmetros base (valores padrão)
config = Configurações()
params = Parâmetros()

# Alterando as configurações para usar a distribuição Gamma
config.correlações.fracionamento_asfaltenos = 'Distribuição_Gamma'

# Parâmetros regredidos (MW_molecula, x_Asf0, x_Asf1) = valores padrão. (alterável)
params.equilíbrio.MW_molecula = 670.0
params.equilíbrio.x_Asf0 = 0.3
params.equilíbrio.x_Asf1 = 0.3

# ==================================================================================================================== #
# PARTE 2: CALCULANDO O ERRO
# ==================================================================================================================== #

def calcular_erro(alpha, MW_avg):
    # Cria uma cópia dos parâmetros para não alterar o original
    params_copy = copy.deepcopy(params)
    params_copy.equilíbrio.alfa = alpha
    params_copy.equilíbrio.MW_avg = MW_avg

    # Calcula as propriedades termodinâmicas
    propriedades = Propriedades.calcular(T, precipitante, params_copy.equilíbrio, config)

    # Cria o sistema com as composições globais
    sistema = FasesSistema.criar(ws_exp, SARA, n_dados_exp, propriedades)
    modelo_termodinâmico = ModeloTermodinâmico.criar(config.cálculo.modelo_termodinâmico, propriedades)

    # Arrays para armazenar as composições das fases
    xs_L = np.zeros_like(sistema.fase_leve.xs)
    xs_H = np.zeros_like(sistema.fase_pesada.xs)
    betas = np.zeros(n_dados_exp)

    # Calcula o flash para cada ponto experimental
    for i in range(n_dados_exp):
        equilíbrio = FlashELL(sistema.fase_global.xs[i], modelo_termodinâmico)
        betas[i], xs_L[i, :], xs_H[i, :], _ = equilíbrio.calcular_flash()

    sistema.betas = betas
    sistema.fase_leve.definir_composições('molar', xs_L, propriedades)
    sistema.fase_pesada.definir_composições('molar', xs_H, propriedades)

    # Calcula os yields e o erro
    yields_calc = sistema.yields_calc(propriedades.MMs)
    erro = np.mean(np.abs(yields_exp - yields_calc)) # Pode ser modificado para outro tipo de erro.
    return erro

# ==================================================================================================================== #
# PARTE 3: DEFINIÇÃO DA GRADE E CÁLCULO DOS ERROS
# ==================================================================================================================== #

# Limites dos parâmetros
alpha_min, alpha_max = 0.1, 20.0
MW_avg_min, MW_avg_max = 1800.0, 5000.0

# Número de pontos
n_alpha = 20
n_MW = 20

alpha_range = np.linspace(alpha_min, alpha_max, n_alpha)
MW_avg_range = np.linspace(MW_avg_min, MW_avg_max, n_MW)

# Matriz de erros
errors = np.zeros((n_alpha, n_MW))

print("Iniciando o cálculo de erros...")
for i, alpha in enumerate(alpha_range):
    for j, MW_avg in enumerate(MW_avg_range):
        errors[i, j] = calcular_erro(alpha, MW_avg)
        print(f"alfa = {alpha:.2f}, MW_avg = {MW_avg:.0f}  ->  erro = {errors[i, j]:.6f}")

# Verifica se todos os erros são iguais
if np.allclose(errors, errors[0,0]):
    print("ATENÇÃO: Todos os erros são iguais.")

# ==================================================================================================================== #
# PARTE 4: PLOTAGEM DO COLORMAP
# ==================================================================================================================== #

plt.figure(figsize=(10, 8))
plt.pcolormesh(alpha_range, MW_avg_range, errors.T, 
               cmap='viridis', 
               shading='auto')

# Encontra a posição do menor erro na matriz
i_min, j_min = np.unravel_index(np.argmin(errors), errors.shape)

alpha_opt = alpha_range[i_min]
MW_opt = MW_avg_range[j_min]
erro_min = errors[i_min, j_min]

# Adiciona o marcador no gráfico
plt.scatter(alpha_opt, MW_opt, 
            color='red', marker='*', s=250, 
            edgecolor='white', linewidth=1.5,
            label=f'Mínimo: α={alpha_opt:.2f}, MW={MW_opt:.0f}')

# Adiciona uma legenda para o marcador
plt.legend(loc='upper right', fontsize=10)
cbar = plt.colorbar(label='Erro (Desvio Absoluto Médio)', pad=0.02)
cbar.ax.tick_params(labelsize=12)

plt.xlabel('Parâmetro alfa', fontsize=14)
plt.ylabel('Massa molar média MW_avg (g/mol)', fontsize=14)
plt.title(f'Mapa de erro para {nome_planilha}', fontsize=16)

if not np.allclose(errors, errors[0,0]):
    contour_levels = np.linspace(errors.min(), errors.max(), 10)
    plt.contour(alpha_range, MW_avg_range, errors.T, levels=contour_levels, colors='white', linewidths=0.5)
else:
    print("Erros constantes, contornos não serão plotados.")

plt.tight_layout()

# Salvar a figura na pasta de resultados
diretorio_saida = os.path.join(diretorio, 'Resultados', 'Mapas_Erro')
os.makedirs(diretorio_saida, exist_ok=True)
plt.savefig(os.path.join(diretorio_saida, f'{nome_planilha}_colormap_alpha_MWavg.png'), dpi=300)

plt.show()

print("O colormap foi gerado e salvo.")