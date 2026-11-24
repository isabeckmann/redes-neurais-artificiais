"""Interface web local (só biblioteca padrão): python main.py web  →  http://127.0.0.1:8000

Reaproveita a biblioteca ``mlp``: nada de algoritmo é reimplementado aqui.
"""

import json
import os
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional

from mlp import (
    Preditor, RedeNeural, Treinador, TreinamentoDivergiu, carregar_rede, ler_csv, mse, preparar,
    salvar_rede, verificar_compatibilidade,
)
from mlp.dataset import ler_cabecalho

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAIDA = os.path.join(RAIZ, "modelos", "rede_web.json")
ESTADO: Dict[str, Optional[RedeNeural]] = {"rede": None}


def _camadas(texto):
    return [(int(q), a.strip().lower()) for q, a in (i.split(":") for i in texto.split(",") if i.strip())]


def _csv_temp(texto):
    f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8", newline="")
    f.write(texto)
    f.close()
    return f.name


def api_cabecalho(d):
    caminho = _csv_temp(d["csv"])
    try:
        return {"colunas": [c.strip() for c in ler_cabecalho(caminho)]}
    finally:
        os.unlink(caminho)


def api_treinar(d):
    caminho = _csv_temp(d["csv"])
    try:
        ds = ler_csv(caminho, d["entradas"], d["saidas"])
    finally:
        os.unlink(caminho)
    arq = _camadas(d["ocultas"]) + _camadas(d["saida"])
    rede = RedeNeural.construir(len(ds.colunas_entrada), arq, seed=d.get("seed"))
    verificar_compatibilidade(rede, ds)
    div = preparar(ds, tuple(d["divisao"]), seed=d.get("seedDivisao"), normalizacao=d.get("normalizacao") or None)
    rede.normalizacao = div.normalizador
    medias = [sum(c) / len(c) for c in zip(*div.treino.saidas)]
    if d.get("viesMedia"):
        rede.definir_vies_saida(medias)
    try:
        hist = Treinador(rede, float(d["taxa"]), int(d["epocas"]), seed=d.get("seedTreino")).treinar(div.treino, div.validacao)
    except TreinamentoDivergiu as e:
        return {"erro": str(e)}

    def conj(parte):
        if len(parte) == 0:
            return None
        return mse(rede, parte.entradas, parte.saidas)

    def base(parte):  # MSE de quem sempre prevê a média do treino
        if len(parte) == 0:
            return None
        return sum(sum((m - t) ** 2 for m, t in zip(medias, y)) / len(y) for y in parte.saidas) / len(parte)

    res = {"treino": conj(div.treino), "validacao": conj(div.validacao), "teste": conj(div.teste)}
    rede.metadados.update({
        "colunas_entrada": ds.colunas_entrada, "colunas_saida": ds.colunas_saida,
        "divisao": {"percentuais": d["divisao"], "seed": d.get("seedDivisao")},
        "treinamento": {"algoritmo": "SGD", "taxa_aprendizado": float(d["taxa"]), "epocas": int(d["epocas"]),
                        "seed": d.get("seedTreino"), "vies_saida_media": bool(d.get("viesMedia"))},
        "resultado": {"mse_treino": res["treino"], "mse_validacao": res["validacao"], "mse_teste": res["teste"]},
    })
    salvar_rede(rede, SAIDA)
    ESTADO["rede"] = rede
    pares = [[t[0], rede.propagar(x)[0]] for x, t in zip(div.teste.entradas, div.teste.saidas)]
    return {
        "historico": [[r.epoca, r.mse_treino, r.mse_validacao] for r in hist],
        "mse": res, "baseline": {"validacao": base(div.validacao), "teste": base(div.teste)},
        "pares_teste": pares, "colunas_entrada": ds.colunas_entrada, "colunas_saida": ds.colunas_saida,
        "descricao": rede.descrever(), "divisao": div.resumo(), "arquivo": os.path.relpath(SAIDA, RAIZ),
    }


def api_prever(d):
    rede = ESTADO["rede"] or (carregar_rede(SAIDA) if os.path.exists(SAIDA) else None)
    if rede is None:
        return {"erro": "Treine uma rede primeiro."}
    ESTADO["rede"] = rede
    return {"saida": Preditor(rede).prever([float(v) for v in d["valores"]])}


ROTAS = {"/api/cabecalho": api_cabecalho, "/api/treinar": api_treinar, "/api/prever": api_prever}


class Handler(BaseHTTPRequestHandler):
    def _enviar(self, codigo, corpo, tipo):
        self.send_response(codigo)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(corpo)))
        self.end_headers()
        self.wfile.write(corpo)

    def do_GET(self):
        if self.path == "/api/datasets":
            pasta = os.path.join(RAIZ, "datasets")
            nomes = sorted(f for f in os.listdir(pasta) if f.endswith(".csv")) if os.path.isdir(pasta) else []
            dados = [{"nome": n, "csv": open(os.path.join(pasta, n), encoding="utf-8-sig").read()} for n in nomes]
            return self._enviar(200, json.dumps(dados).encode(), "application/json")
        with open(os.path.join(os.path.dirname(__file__), "web.html"), "rb") as f:
            self._enviar(200, f.read(), "text/html; charset=utf-8")

    def do_POST(self):
        func = ROTAS.get(self.path)
        if not func:
            return self._enviar(404, b"{}", "application/json")
        try:
            corpo = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            resp = func(corpo)
        except (ValueError, KeyError, RuntimeError, FileNotFoundError) as e:
            resp = {"erro": str(e)}
        self._enviar(200, json.dumps(resp).encode(), "application/json")

    def log_message(self, format, *args):
        pass


def servir(porta=8000):
    srv = ThreadingHTTPServer(("127.0.0.1", porta), Handler)
    print(f"Interface web em http://127.0.0.1:{porta}  (Ctrl+C para encerrar)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0