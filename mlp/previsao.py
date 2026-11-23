"""Mecanismo de previsão (requisito 1.12)."""


class Preditor:
    """Executa uma rede treinada sobre novos valores (somente propagação direta).

    Aplica automaticamente a normalização registrada na rede, se houver.
    Não altera pesos nem vieses.
    """

    def __init__(self, rede):
        self.rede = rede

    def preparar_entrada(self, valores):
        try:
            valores = [float(v) for v in valores]
        except (TypeError, ValueError):
            raise ValueError("Todos os valores de entrada devem ser numéricos.") from None
        if len(valores) != self.rede.n_entradas:
            raise ValueError(
                f"Quantidade de valores incompatível: a rede espera {self.rede.n_entradas} "
                f"entrada(s), foram informados {len(valores)}."
            )
        if self.rede.normalizacao is not None:
            valores = self.rede.normalizacao.transformar_amostra(valores)
        return valores

    def prever(self, valores):
        return self.rede.propagar(self.preparar_entrada(valores))

    def prever_lote(self, lista_valores):
        return [self.prever(v) for v in lista_valores]
