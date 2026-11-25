class Neuronio:
    """Neurônio artificial.

    Armazena pesos sinápticos e viés, calcula a soma ponderada ``z`` e aplica a
    função de ativação. Mantém os valores intermediários usados no treinamento:

    - ``entradas``: entradas recebidas na última propagação direta;
    - ``z``: soma ponderada (w·x + b);
    - ``saida``: f(z);
    - ``delta``: ∂E/∂z, calculado na retropropagação.
    """

    def __init__(self, pesos, vies, ativacao):
        self.pesos = [float(p) for p in pesos]
        self.vies = float(vies)
        self.ativacao = ativacao
        self.entradas = []
        self.z = 0.0
        self.saida = 0.0
        self.delta = 0.0

    @property
    def n_entradas(self):
        return len(self.pesos)

    def soma_ponderada(self, entradas):
        return sum(w * x for w, x in zip(self.pesos, entradas)) + self.vies

    def ativar(self, entradas):
        """Propagação direta do neurônio: guarda entradas, z e saída."""
        self.entradas = list(entradas)
        self.z = self.soma_ponderada(self.entradas)
        self.saida = self.ativacao.calcular(self.z)
        return self.saida

    def derivada_ativacao(self):
        """f'(z) avaliada no último z calculado."""
        return self.ativacao.derivada(self.z, self.saida)

    def gradientes(self):
        """Gradientes do erro: ∂E/∂w_i = δ·x_i e ∂E/∂b = δ."""
        return [self.delta * x for x in self.entradas], self.delta

    def atualizar_parametros(self, taxa_aprendizado):
        """Descida do gradiente: w ← w − η·∂E/∂w ; b ← b − η·∂E/∂b."""
        grad_pesos, grad_vies = self.gradientes()
        for i, g in enumerate(grad_pesos):
            self.pesos[i] -= taxa_aprendizado * g
        self.vies -= taxa_aprendizado * grad_vies

    def __repr__(self):
        return f"Neuronio(pesos={self.pesos}, vies={self.vies}, ativacao={self.ativacao.nome})"
