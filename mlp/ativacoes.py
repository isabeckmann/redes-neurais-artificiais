import math

class FuncaoAtivacao:
    """Classe base das funções de ativação."""

    nome = ""

    def calcular(self, z):
        raise NotImplementedError

    def derivada(self, z, saida):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class Linear(FuncaoAtivacao):
    """f(z) = z ; f'(z) = 1"""

    nome = "linear"

    def calcular(self, z):
        return z

    def derivada(self, z, saida):
        return 1.0


class Sigmoid(FuncaoAtivacao):
    """f(z) = 1 / (1 + e^-z) ; f'(z) = f(z) * (1 - f(z))"""

    nome = "sigmoid"

    def calcular(self, z):
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        e = math.exp(z)
        return e / (1.0 + e)

    def derivada(self, z, saida):
        return saida * (1.0 - saida)


class Tanh(FuncaoAtivacao):
    """f(z) = tanh(z) ; f'(z) = 1 - tanh(z)^2"""

    nome = "tanh"

    def calcular(self, z):
        return math.tanh(z)

    def derivada(self, z, saida):
        return 1.0 - saida * saida


class ReLU(FuncaoAtivacao):
    """f(z) = max(0, z) ; f'(z) = 1 se z > 0, senão 0"""

    nome = "relu"

    def calcular(self, z):
        return z if z > 0.0 else 0.0

    def derivada(self, z, saida):
        return 1.0 if z > 0.0 else 0.0


_ATIVACOES = {cls.nome: cls() for cls in (Linear, Sigmoid, Tanh, ReLU)}


def ativacoes_disponiveis():
    """Lista os nomes das funções de ativação suportadas."""
    return list(_ATIVACOES)


def obter_ativacao(nome):
    """Retorna a função de ativação correspondente ao nome (sem diferenciar maiúsculas)."""
    chave = str(nome).strip().lower()
    if chave not in _ATIVACOES:
        raise ValueError(
            f"Função de ativação desconhecida: '{nome}'. "
            f"Disponíveis: {', '.join(ativacoes_disponiveis())}."
        )
    return _ATIVACOES[chave]
