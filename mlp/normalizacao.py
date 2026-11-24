"""Normalização dos atributos de entrada (requisito 1.9).

Os parâmetros são ajustados (``ajustar``) somente com o conjunto de treinamento
e depois aplicados (``transformar``) aos três subconjuntos e às novas entradas
da previsão. As saídas esperadas nunca são normalizadas.
"""

from typing import List


class Normalizador:
    nome = ""

    @property
    def ajustado(self):
        raise NotImplementedError

    @property
    def n_atributos(self):
        raise NotImplementedError

    def ajustar(self, amostras):
        raise NotImplementedError

    def transformar_amostra(self, amostra):
        raise NotImplementedError

    def transformar(self, amostras):
        return [self.transformar_amostra(a) for a in amostras]

    def parametros(self):
        raise NotImplementedError

    def para_dict(self):
        return {"tipo": self.nome, "parametros": self.parametros()}

    def _verificar_ajustado(self, amostra):
        if not self.ajustado:
            raise RuntimeError("Normalizador ainda não foi ajustado.")
        if len(amostra) != self.n_atributos:
            raise ValueError(
                f"Normalizador espera {self.n_atributos} atributo(s), recebeu {len(amostra)}."
            )


class MinMax(Normalizador):
    """x' = (x − min) / (max − min), resultando em [0, 1] no conjunto de treino.

    Atributos constantes (max == min) são mapeados para 0.
    """

    nome = "minmax"

    def __init__(self, minimos=None, maximos=None):
        self.minimos: List[float] = list(minimos) if minimos is not None else []
        self.maximos: List[float] = list(maximos) if maximos is not None else []

    @property
    def ajustado(self):
        return len(self.minimos) > 0

    @property
    def n_atributos(self):
        return len(self.minimos)

    def ajustar(self, amostras):
        if not amostras:
            raise ValueError("Não é possível ajustar a normalização com um conjunto vazio.")
        colunas = list(zip(*amostras))
        self.minimos = [min(c) for c in colunas]
        self.maximos = [max(c) for c in colunas]
        return self

    def transformar_amostra(self, amostra):
        self._verificar_ajustado(amostra)
        resultado = []
        for x, mn, mx in zip(amostra, self.minimos, self.maximos):
            amplitude = mx - mn
            resultado.append((x - mn) / amplitude if amplitude != 0 else 0.0)
        return resultado

    def parametros(self):
        return {"minimos": self.minimos, "maximos": self.maximos}


class ZScore(Normalizador):
    """x' = (x − média) / desvio_padrão (desvio populacional).

    Atributos com desvio 0 usam desvio 1 (resultado 0).
    """

    nome = "zscore"

    def __init__(self, medias=None, desvios=None):
        self.medias: List[float] = list(medias) if medias is not None else []
        self.desvios: List[float] = list(desvios) if desvios is not None else []

    @property
    def ajustado(self):
        return len(self.medias) > 0

    @property
    def n_atributos(self):
        return len(self.medias)

    def ajustar(self, amostras):
        if not amostras:
            raise ValueError("Não é possível ajustar a normalização com um conjunto vazio.")
        n = len(amostras)
        colunas = list(zip(*amostras))
        self.medias = [sum(c) / n for c in colunas]
        self.desvios = []
        for c, m in zip(colunas, self.medias):
            var = sum((x - m) * (x - m) for x in c) / n
            self.desvios.append(var ** 0.5)
        return self

    def transformar_amostra(self, amostra):
        self._verificar_ajustado(amostra)
        return [
            (x - m) / (d if d != 0 else 1.0)
            for x, m, d in zip(amostra, self.medias, self.desvios)
        ]

    def parametros(self):
        return {"medias": self.medias, "desvios": self.desvios}


_NORMALIZADORES = {"minmax": MinMax, "zscore": ZScore}


def normalizacoes_disponiveis():
    return list(_NORMALIZADORES)


def criar_normalizador(nome):
    """Cria um normalizador vazio pelo nome; ``None``/"nenhuma" retorna None."""
    if nome is None:
        return None
    chave = str(nome).strip().lower().replace("-", "").replace("_", "")
    if chave in ("", "nenhuma", "none"):
        return None
    if chave not in _NORMALIZADORES:
        raise ValueError(
            f"Normalização desconhecida: '{nome}'. Disponíveis: nenhuma, {', '.join(_NORMALIZADORES)}."
        )
    return _NORMALIZADORES[chave]()


def normalizador_de_dict(dados):
    """Reconstrói um normalizador a partir de ``para_dict()`` (ou None)."""
    if not dados:
        return None
    tipo = dados["tipo"]
    p = dados["parametros"]
    if tipo == "minmax":
        return MinMax(p["minimos"], p["maximos"])
    if tipo == "zscore":
        return ZScore(p["medias"], p["desvios"])
    raise ValueError(f"Normalização desconhecida no arquivo: '{tipo}'.")