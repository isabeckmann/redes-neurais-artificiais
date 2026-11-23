"""Entidade Camada (requisito 1.2)."""

from .ativacoes import obter_ativacao
from .inicializacao import gerar_pesos, gerar_vies, metodo_recomendado
from .neuronio import Neuronio


class Camada:
    """Agrupa neurônios que compartilham a mesma função de ativação."""

    def __init__(self, neuronios, ativacao, inicializacao=None):
        if not neuronios:
            raise ValueError("Uma camada precisa ter pelo menos um neurônio.")
        n = neuronios[0].n_entradas
        if any(neu.n_entradas != n for neu in neuronios):
            raise ValueError("Todos os neurônios da camada devem ter o mesmo número de entradas.")
        self.neuronios = neuronios
        self.ativacao = ativacao
        self.inicializacao = inicializacao

    @classmethod
    def criar(cls, n_entradas, n_neuronios, nome_ativacao, rng, inicializacao=None):
        """Cria uma camada com pesos aleatórios (Xavier/He) e vieses iniciais."""
        if n_entradas < 1 or n_neuronios < 1:
            raise ValueError("Número de entradas e de neurônios deve ser >= 1.")
        ativacao = obter_ativacao(nome_ativacao)
        metodo = inicializacao or metodo_recomendado(ativacao.nome)
        neuronios = [
            Neuronio(gerar_pesos(n_entradas, n_neuronios, metodo, rng), gerar_vies(), ativacao)
            for _ in range(n_neuronios)
        ]
        return cls(neuronios, ativacao, metodo)

    @property
    def n_entradas(self):
        return self.neuronios[0].n_entradas

    @property
    def n_neuronios(self):
        return len(self.neuronios)

    @property
    def saidas(self):
        return [n.saida for n in self.neuronios]

    def propagar(self, entradas):
        """Processa as entradas da camada anterior e devolve as saídas para a seguinte."""
        return [n.ativar(entradas) for n in self.neuronios]

    def calcular_deltas_saida(self, gradiente_erro):
        """Camada de saída: δ_k = ∂E/∂y_k · f'(z_k)."""
        for neuronio, g in zip(self.neuronios, gradiente_erro):
            neuronio.delta = g * neuronio.derivada_ativacao()

    def calcular_deltas_oculta(self, camada_seguinte):
        """Camada oculta: δ_j = (Σ_k w_kj · δ_k) · f'(z_j).

        Usa os pesos atuais da camada seguinte, que ainda não foram atualizados
        (as atualizações só ocorrem depois de todos os deltas serem calculados).
        """
        for j, neuronio in enumerate(self.neuronios):
            soma = sum(prox.pesos[j] * prox.delta for prox in camada_seguinte.neuronios)
            neuronio.delta = soma * neuronio.derivada_ativacao()

    def atualizar_parametros(self, taxa_aprendizado):
        for neuronio in self.neuronios:
            neuronio.atualizar_parametros(taxa_aprendizado)

    def __repr__(self):
        return f"Camada({self.n_neuronios} neurônios, ativacao={self.ativacao.nome})"
