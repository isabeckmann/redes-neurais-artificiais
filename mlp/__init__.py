"""Biblioteca MLP implementada do zero (sem frameworks de redes neurais).

Módulos:
    ativacoes      – funções de ativação e derivadas
    inicializacao  – Xavier/Glorot e He, com semente
    neuronio       – entidade Neurônio
    camada         – entidade Camada
    rede           – entidade Rede Neural (forward e backpropagation)
    erro           – MSE
    normalizacao   – Min-Max e Z-score
    dataset        – leitura de CSV, divisão e preparação
    treinamento    – treinamento por SGD
    persistencia   – salvar/carregar em JSON
    previsao       – execução da rede treinada sobre novos dados
"""

from .ativacoes import ativacoes_disponiveis, obter_ativacao
from .camada import Camada
from .dataset import Dataset, dividir, ler_csv, preparar, verificar_compatibilidade
from .erro import mse
from .neuronio import Neuronio
from .normalizacao import criar_normalizador, normalizacoes_disponiveis
from .persistencia import carregar_rede, salvar_rede
from .previsao import Preditor
from .rede import RedeNeural
from .treinamento import Treinador, TreinamentoDivergiu

__all__ = [
    "Camada", "Dataset", "Neuronio", "Preditor", "RedeNeural", "Treinador", "TreinamentoDivergiu",
    "ativacoes_disponiveis", "carregar_rede", "criar_normalizador", "dividir", "ler_csv", "mse",
    "normalizacoes_disponiveis", "obter_ativacao", "preparar", "salvar_rede", "verificar_compatibilidade",
]
