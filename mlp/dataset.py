"""Leitura e preparação de datasets (requisito 1.9)."""

import csv
import random
from dataclasses import dataclass, field
from typing import Optional

from .normalizacao import Normalizador, criar_normalizador


@dataclass
class Dataset:
    """Conjunto de exemplos: ``entradas[i]`` → ``saidas[i]``."""

    colunas_entrada: list
    colunas_saida: list
    entradas: list = field(default_factory=list)
    saidas: list = field(default_factory=list)

    def __len__(self):
        return len(self.entradas)

    def subconjunto(self, indices):
        return Dataset(
            list(self.colunas_entrada),
            list(self.colunas_saida),
            [list(self.entradas[i]) for i in indices],
            [list(self.saidas[i]) for i in indices],
        )


@dataclass
class DivisaoDataset:
    treino: Dataset
    validacao: Dataset
    teste: Dataset
    normalizador: Optional[Normalizador] = None
    seed: object = None
    percentuais: tuple = ()

    def resumo(self):
        return (
            f"treino={len(self.treino)}  validação={len(self.validacao)}  teste={len(self.teste)}"
        )


def _resolver_colunas(especificacao, cabecalho, nome_param):
    """Converte nomes de colunas ou índices (base 0) em índices do cabeçalho."""
    indices = []
    for item in especificacao:
        item = str(item).strip()
        if item in cabecalho:
            indices.append(cabecalho.index(item))
        elif item.lstrip("-").isdigit() and -len(cabecalho) <= int(item) < len(cabecalho):
            indices.append(int(item) % len(cabecalho))
        else:
            raise ValueError(
                f"Coluna de {nome_param} não encontrada: '{item}'. Colunas do arquivo: {', '.join(cabecalho)}."
            )
    return indices


def ler_cabecalho(caminho, delimitador=None):
    with open(caminho, newline="", encoding="utf-8-sig") as f:
        amostra = f.read(4096)
        f.seek(0)
        delim = delimitador or _detectar_delimitador(amostra)
        return next(csv.reader(f, delimiter=delim))


def _detectar_delimitador(amostra):
    try:
        return csv.Sniffer().sniff(amostra, delimiters=",;\t").delimiter
    except csv.Error:
        return ","


def ler_csv(caminho, colunas_entrada=None, colunas_saida=None, delimitador=None):
    """Lê um CSV com cabeçalho e separa atributos de entrada e saídas esperadas.

    ``colunas_entrada``/``colunas_saida`` aceitam nomes ou índices (base 0).
    Padrão: última coluna = saída; demais = entradas.
    """
    with open(caminho, newline="", encoding="utf-8-sig") as f:
        amostra = f.read(4096)
        f.seek(0)
        delim = delimitador or _detectar_delimitador(amostra)
        leitor = csv.reader(f, delimiter=delim)
        try:
            cabecalho = [c.strip() for c in next(leitor)]
        except StopIteration:
            raise ValueError(f"Arquivo vazio: {caminho}") from None
        linhas = [l for l in leitor if any(c.strip() for c in l)]

    if colunas_saida:
        idx_saida = _resolver_colunas(colunas_saida, cabecalho, "saída")
    else:
        idx_saida = [len(cabecalho) - 1]
    if colunas_entrada:
        idx_entrada = _resolver_colunas(colunas_entrada, cabecalho, "entrada")
    else:
        idx_entrada = [i for i in range(len(cabecalho)) if i not in idx_saida]

    if not idx_entrada or not idx_saida:
        raise ValueError("É necessário pelo menos uma coluna de entrada e uma de saída.")
    if set(idx_entrada) & set(idx_saida):
        raise ValueError("Uma mesma coluna não pode ser entrada e saída ao mesmo tempo.")

    ds = Dataset([cabecalho[i] for i in idx_entrada], [cabecalho[i] for i in idx_saida])
    for n_linha, linha in enumerate(linhas, start=2):
        if len(linha) != len(cabecalho):
            raise ValueError(
                f"Linha {n_linha}: {len(linha)} coluna(s), esperado {len(cabecalho)}."
            )
        try:
            ds.entradas.append([float(linha[i]) for i in idx_entrada])
            ds.saidas.append([float(linha[i]) for i in idx_saida])
        except ValueError:
            raise ValueError(f"Linha {n_linha}: valor não numérico.") from None
    if not ds.entradas:
        raise ValueError(f"O arquivo {caminho} não possui exemplos.")
    return ds


def verificar_compatibilidade(rede, dataset):
    """Garante que o dataset tem o número de entradas e saídas da arquitetura."""
    n_ent = len(dataset.colunas_entrada)
    n_sai = len(dataset.colunas_saida)
    erros = []
    if n_ent != rede.n_entradas:
        erros.append(f"dataset tem {n_ent} coluna(s) de entrada, a rede espera {rede.n_entradas}")
    if n_sai != rede.n_saidas:
        erros.append(f"dataset tem {n_sai} coluna(s) de saída, a rede possui {rede.n_saidas} neurônio(s) de saída")
    if erros:
        raise ValueError("Dataset incompatível com a rede: " + "; ".join(erros) + ".")


def dividir(dataset, perc_treino, perc_validacao, perc_teste, seed=None):
    """Divide aleatoriamente em treino/validação/teste (percentuais somando 100)."""
    percs = (float(perc_treino), float(perc_validacao), float(perc_teste))
    if any(p < 0 for p in percs):
        raise ValueError("Os percentuais não podem ser negativos.")
    if abs(sum(percs) - 100.0) > 1e-6:
        raise ValueError(f"Os percentuais devem totalizar 100% (soma atual: {sum(percs):g}%).")

    n = len(dataset)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)

    n_treino = round(n * percs[0] / 100.0)
    n_val = round(n * percs[1] / 100.0)
    n_val = min(n_val, n - n_treino)
    if n_treino < 1:
        raise ValueError("O conjunto de treinamento ficou vazio; aumente o percentual de treino.")

    treino = dataset.subconjunto(indices[:n_treino])
    validacao = dataset.subconjunto(indices[n_treino:n_treino + n_val])
    teste = dataset.subconjunto(indices[n_treino + n_val:])
    return DivisaoDataset(treino, validacao, teste, None, seed, percs)


def preparar(dataset, percentuais=(70, 15, 15), seed=None, normalizacao=None):
    """Divide o dataset e, opcionalmente, normaliza as entradas.

    A normalização é ajustada exclusivamente no subconjunto de treinamento e
    então aplicada aos três subconjuntos. As saídas não são normalizadas.
    """
    divisao = dividir(dataset, *percentuais, seed=seed)
    normalizador = criar_normalizador(normalizacao)
    if normalizador is not None:
        normalizador.ajustar(divisao.treino.entradas)
        for parte in (divisao.treino, divisao.validacao, divisao.teste):
            parte.entradas = normalizador.transformar(parte.entradas)
    divisao.normalizador = normalizador
    return divisao