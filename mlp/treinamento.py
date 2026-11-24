"""Mecanismo de treinamento com SGD (requisito 1.10)."""

import math
import random
from dataclasses import dataclass

from .erro import mse


class TreinamentoDivergiu(RuntimeError):
    """O erro tornou-se infinito/NaN (geralmente taxa de aprendizado alta demais)."""


@dataclass
class RegistroEpoca:
    epoca: int
    mse_treino: float
    mse_validacao: object  # float ou None se não houver conjunto de validação


class Treinador:
    """Coordena propagação direta, cálculo do erro, retropropagação e atualização.

    Usa Stochastic Gradient Descent: os parâmetros são atualizados após cada
    exemplo de treinamento. Apenas o conjunto de treino ajusta os parâmetros; o
    de validação é só avaliado ao fim de cada época.
    """

    def __init__(self, rede, taxa_aprendizado, epocas, embaralhar=True, seed=None, ao_fim_da_epoca=None):
        if taxa_aprendizado <= 0:
            raise ValueError("A taxa de aprendizado deve ser positiva.")
        if int(epocas) < 1:
            raise ValueError("O número de épocas deve ser >= 1.")
        self.rede = rede
        self.taxa_aprendizado = float(taxa_aprendizado)
        self.epocas = int(epocas)
        self.embaralhar = embaralhar
        self.seed = seed
        self.ao_fim_da_epoca = ao_fim_da_epoca
        self.historico = []

    def treinar(self, treino, validacao=None):
        """Executa o treinamento e retorna o histórico de MSE por época."""
        if len(treino) == 0:
            raise ValueError("Conjunto de treinamento vazio.")
        rng = random.Random(self.seed)
        ordem = list(range(len(treino)))
        val_x = validacao.entradas if validacao is not None else []
        val_y = validacao.saidas if validacao is not None else []

        for epoca in range(1, self.epocas + 1):
            if self.embaralhar:
                rng.shuffle(ordem)
            try:
                for i in ordem:
                    self.rede.treinar_exemplo(treino.entradas[i], treino.saidas[i], self.taxa_aprendizado)
                mse_treino = mse(self.rede, treino.entradas, treino.saidas)
                mse_val = mse(self.rede, val_x, val_y)
            except OverflowError:
                mse_treino = math.inf
                mse_val = None

            if mse_treino is None or not math.isfinite(mse_treino):
                raise TreinamentoDivergiu(
                    f"O treinamento divergiu na época {epoca} (erro infinito/NaN). "
                    "Reduza a taxa de aprendizado ou normalize as entradas."
                )
            registro = RegistroEpoca(epoca, mse_treino, mse_val)
            self.historico.append(registro)
            if self.ao_fim_da_epoca:
                self.ao_fim_da_epoca(registro)
        return self.historico