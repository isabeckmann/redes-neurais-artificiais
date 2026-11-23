"""Cálculo do erro — Erro Quadrático Médio / MSE (requisito 1.7).

Erro de um exemplo com m saídas:      E = (1/m) · Σ_k (y_k − t_k)²
Gradiente em relação a cada saída:     ∂E/∂y_k = (2/m) · (y_k − t_k)
MSE de um conjunto com N exemplos:     MSE = (1/N) · Σ_n E_n
"""


def erro_quadratico(saidas, esperadas):
    """Erro quadrático médio de um único exemplo (média sobre as saídas)."""
    if len(saidas) != len(esperadas):
        raise ValueError("Saídas e valores esperados têm tamanhos diferentes.")
    soma = 0.0
    for y, t in zip(saidas, esperadas):
        d = y - t
        soma += d * d
    return soma / len(saidas)


def gradiente_erro(saidas, esperadas):
    """∂E/∂y_k para cada saída do exemplo."""
    m = len(saidas)
    return [2.0 * (y - t) / m for y, t in zip(saidas, esperadas)]


def mse(rede, entradas, esperadas):
    """MSE da rede sobre um conjunto (treinamento, validação ou teste).

    Apenas propaga (não altera pesos). Retorna ``None`` para conjunto vazio.
    """
    if not entradas:
        return None
    total = 0.0
    for x, t in zip(entradas, esperadas):
        total += erro_quadratico(rede.propagar(x), t)
    return total / len(entradas)
