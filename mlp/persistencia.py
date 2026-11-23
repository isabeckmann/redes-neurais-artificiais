"""Persistência da rede neural em JSON (requisito 1.11).

Formato (documentado também no README):

{
  "formato": "mlp-json",
  "versao": 1,
  "n_entradas": 2,
  "n_camadas": 2,
  "camadas": [
    {
      "n_neuronios": 4,
      "ativacao": "tanh",
      "inicializacao": "xavier",
      "neuronios": [ {"pesos": [w1, w2], "vies": b}, ... ]
    },
    ...                              (a última camada é a de saída)
  ],
  "normalizacao": null | {"tipo": "minmax", "parametros": {"minimos": [...], "maximos": [...]}}
                       | {"tipo": "zscore", "parametros": {"medias": [...], "desvios": [...]}},
  "metadados": { ... informações livres: colunas, hiperparâmetros, MSEs ... }
}

Os números de ponto flutuante são gravados com ``repr`` pelo módulo json, o
que garante reconstrução exata e, portanto, saídas idênticas após carregar.
"""

import json
import os

from .ativacoes import obter_ativacao
from .camada import Camada
from .neuronio import Neuronio
from .normalizacao import normalizador_de_dict
from .rede import RedeNeural

FORMATO = "mlp-json"
VERSAO = 1


def rede_para_dict(rede):
    return {
        "formato": FORMATO,
        "versao": VERSAO,
        "n_entradas": rede.n_entradas,
        "n_camadas": len(rede.camadas),
        "camadas": [
            {
                "n_neuronios": c.n_neuronios,
                "ativacao": c.ativacao.nome,
                "inicializacao": c.inicializacao,
                "neuronios": [{"pesos": list(n.pesos), "vies": n.vies} for n in c.neuronios],
            }
            for c in rede.camadas
        ],
        "normalizacao": rede.normalizacao.para_dict() if rede.normalizacao is not None else None,
        "metadados": rede.metadados,
    }


def rede_de_dict(dados):
    if dados.get("formato") != FORMATO:
        raise ValueError("Arquivo não está no formato 'mlp-json'.")
    if dados.get("versao") != VERSAO:
        raise ValueError(f"Versão de formato não suportada: {dados.get('versao')}.")
    if not dados.get("camadas"):
        raise ValueError("O arquivo não descreve nenhuma camada.")

    rede = RedeNeural(dados["n_entradas"])
    for i, c in enumerate(dados["camadas"], start=1):
        ativacao = obter_ativacao(c["ativacao"])
        neuronios = [Neuronio(n["pesos"], n["vies"], ativacao) for n in c["neuronios"]]
        if len(neuronios) != c.get("n_neuronios", len(neuronios)):
            raise ValueError(f"Camada {i}: n_neuronios não confere com a lista de neurônios.")
        try:
            rede.adicionar_camada_pronta(Camada(neuronios, ativacao, c.get("inicializacao")))
        except ValueError as e:
            raise ValueError(f"Camada {i}: {e}") from None
    rede.normalizacao = normalizador_de_dict(dados.get("normalizacao"))
    if rede.normalizacao is not None and rede.normalizacao.n_atributos != rede.n_entradas:
        raise ValueError("Parâmetros de normalização incompatíveis com o número de entradas.")
    rede.metadados = dados.get("metadados") or {}
    return rede


def salvar_rede(rede, caminho):
    pasta = os.path.dirname(os.path.abspath(caminho))
    os.makedirs(pasta, exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(rede_para_dict(rede), f, ensure_ascii=False, indent=2)


def carregar_rede(caminho):
    with open(caminho, encoding="utf-8") as f:
        dados = json.load(f)
    return rede_de_dict(dados)
