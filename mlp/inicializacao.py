import math
import random

XAVIER = "xavier"
HE = "he"
METODOS = (XAVIER, HE)


def metodo_recomendado(nome_ativacao):
    """He para ReLU; Xavier para as demais."""
    return HE if nome_ativacao == "relu" else XAVIER


def criar_gerador(seed=None):
    """Cria um gerador pseudoaleatório independente (reprodutível quando há seed)."""
    return random.Random(seed)


def gerar_pesos(fan_in, fan_out, metodo, rng):
    """Gera ``fan_in`` pesos para um neurônio de uma camada com ``fan_out`` neurônios."""
    if metodo == XAVIER:
        limite = math.sqrt(6.0 / (fan_in + fan_out))
        return [rng.uniform(-limite, limite) for _ in range(fan_in)]
    if metodo == HE:
        desvio = math.sqrt(2.0 / fan_in)
        return [rng.gauss(0.0, desvio) for _ in range(fan_in)]
    raise ValueError(f"Método de inicialização desconhecido: '{metodo}'. Use: {', '.join(METODOS)}.")


def gerar_vies():
    return 0.0
