"""Entidade Rede Neural (requisitos 1.3, 1.6 e 1.8)."""

from .camada import Camada
from .erro import erro_quadratico, gradiente_erro
from .inicializacao import criar_gerador


class RedeNeural:
    """Multilayer Perceptron com arquitetura composta dinamicamente.

    Atributos:
        n_entradas: quantidade de entradas da rede.
        camadas: camadas ocultas + camada de saída (a última).
        normalizacao: normalizador das entradas usado no treinamento (ou None).
        metadados: informações livres (colunas do dataset, hiperparâmetros...).
    """

    def __init__(self, n_entradas):
        if int(n_entradas) < 1:
            raise ValueError("A rede precisa ter pelo menos uma entrada.")
        self.n_entradas = int(n_entradas)
        self.camadas = []
        self.normalizacao = None
        self.metadados = {}

    # ------------------------------------------------------------------ montagem
    @classmethod
    def construir(cls, n_entradas, especificacao, seed=None, inicializacao=None):
        """Cria uma rede a partir de uma lista ``[(n_neuronios, ativacao), ...]``.

        A última tupla corresponde à camada de saída; as anteriores, às ocultas.
        """
        if not especificacao:
            raise ValueError("Informe pelo menos a camada de saída.")
        rede = cls(n_entradas)
        rng = criar_gerador(seed)
        for n_neuronios, ativacao in especificacao:
            rede.adicionar_camada(n_neuronios, ativacao, rng, inicializacao)
        rede.metadados["inicializacao"] = {"seed": seed, "metodo": inicializacao or "recomendado"}
        return rede

    def adicionar_camada(self, n_neuronios, ativacao, rng=None, inicializacao=None):
        """Acrescenta uma camada ao final; ela passa a ser a camada de saída."""
        n_ent = self.camadas[-1].n_neuronios if self.camadas else self.n_entradas
        camada = Camada.criar(n_ent, int(n_neuronios), ativacao, rng or criar_gerador(), inicializacao)
        self.camadas.append(camada)
        return camada

    def adicionar_camada_pronta(self, camada):
        """Acrescenta uma camada já construída (usado na reconstrução a partir de arquivo)."""
        esperado = self.camadas[-1].n_neuronios if self.camadas else self.n_entradas
        if camada.n_entradas != esperado:
            raise ValueError(
                f"Camada com {camada.n_entradas} entradas incompatível: esperado {esperado}."
            )
        self.camadas.append(camada)

    def definir_vies_saida(self, valores):
        """Define o viés de cada neurônio de saída (ex.: média das saídas do treino).

        Útil quando as saídas esperadas não são normalizadas e têm magnitude
        grande: a rede começa prevendo a média, evitando gradientes iniciais
        enormes que saturariam os neurônios ocultos.
        """
        if len(valores) != self.n_saidas:
            raise ValueError("Quantidade de vieses diferente do número de saídas.")
        for neuronio, v in zip(self.camada_saida.neuronios, valores):
            neuronio.vies = float(v)

    @property
    def n_saidas(self):
        return self.camadas[-1].n_neuronios if self.camadas else 0

    @property
    def camadas_ocultas(self):
        return self.camadas[:-1]

    @property
    def camada_saida(self):
        return self.camadas[-1]

    def total_parametros(self):
        return sum(c.n_neuronios * (c.n_entradas + 1) for c in self.camadas)

    def _verificar_montada(self):
        if not self.camadas:
            raise RuntimeError("A rede não possui camadas.")

    # ------------------------------------------------------- propagação direta
    def propagar(self, entradas):
        """Forward propagation: a saída de cada camada alimenta a seguinte."""
        self._verificar_montada()
        if len(entradas) != self.n_entradas:
            raise ValueError(
                f"A rede espera {self.n_entradas} entrada(s), mas recebeu {len(entradas)}."
            )
        sinal = [float(v) for v in entradas]
        for camada in self.camadas:
            sinal = camada.propagar(sinal)
        return sinal

    # ------------------------------------------------------- retropropagação
    def retropropagar(self, esperadas):
        """Backpropagation: calcula os deltas de todas as camadas (saída → entrada).

        Deve ser chamado logo após ``propagar`` com o mesmo exemplo. Nenhum peso
        é alterado aqui, de modo que os deltas das camadas ocultas são
        calculados com os mesmos pesos usados na propagação direta.
        """
        self._verificar_montada()
        if len(esperadas) != self.n_saidas:
            raise ValueError(
                f"A rede possui {self.n_saidas} saída(s), mas recebeu {len(esperadas)} valor(es) esperado(s)."
            )
        saida = self.camada_saida
        saida.calcular_deltas_saida(gradiente_erro(saida.saidas, esperadas))
        for i in range(len(self.camadas) - 2, -1, -1):
            self.camadas[i].calcular_deltas_oculta(self.camadas[i + 1])

    def atualizar_parametros(self, taxa_aprendizado):
        """Descida do gradiente em todas as camadas (após todos os deltas calculados)."""
        for camada in self.camadas:
            camada.atualizar_parametros(taxa_aprendizado)

    def treinar_exemplo(self, entradas, esperadas, taxa_aprendizado):
        """Um passo de SGD: propaga, calcula erro, retropropaga e atualiza.

        Retorna o erro do exemplo antes da atualização.
        """
        saidas = self.propagar(entradas)
        erro = erro_quadratico(saidas, esperadas)
        self.retropropagar(esperadas)
        self.atualizar_parametros(taxa_aprendizado)
        return erro

    # ------------------------------------------------------------- descrição
    def descrever(self):
        linhas = [f"Entradas: {self.n_entradas}"]
        for i, c in enumerate(self.camadas, start=1):
            tipo = "Saída " if i == len(self.camadas) else "Oculta"
            linhas.append(
                f"Camada {i} ({tipo}): {c.n_neuronios} neurônio(s), "
                f"ativação={c.ativacao.nome}, inicialização={c.inicializacao or '-'}"
            )
        linhas.append(f"Parâmetros treináveis: {self.total_parametros()}")
        if self.normalizacao is not None:
            linhas.append(f"Normalização das entradas: {self.normalizacao.nome}")
        else:
            linhas.append("Normalização das entradas: nenhuma")
        return "\n".join(linhas)

    def __repr__(self):
        arq = " → ".join([str(self.n_entradas)] + [f"{c.n_neuronios}({c.ativacao.nome})" for c in self.camadas])
        return f"RedeNeural({arq})"
