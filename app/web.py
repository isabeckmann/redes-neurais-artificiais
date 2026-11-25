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
from mlp.persistencia import rede_de_dict, rede_para_dict

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
    rede = RedeNeural.construir(
        len(ds.colunas_entrada), arq, seed=d.get("seed"), inicializacao=d.get("inicializacao") or None,
    )
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
    pares = [[t, rede.propagar(x)] for x, t in zip(div.teste.entradas, div.teste.saidas)]
    return {
        "historico": [[r.epoca, r.mse_treino, r.mse_validacao] for r in hist],
        "mse": res, "baseline": {"validacao": base(div.validacao), "teste": base(div.teste)},
        "pares_teste": pares, "colunas_entrada": ds.colunas_entrada, "colunas_saida": ds.colunas_saida,
        "descricao": rede.descrever(), "divisao": div.resumo(), "arquivo": os.path.relpath(SAIDA, RAIZ),
    }


def api_carregar(d):
    rede = rede_de_dict(d["rede"])
    ESTADO["rede"] = rede
    metadados = rede.metadados
    return {
        "descricao": rede.descrever(), "n_entradas": rede.n_entradas, "n_saidas": rede.n_saidas,
        "colunas_entrada": metadados.get("colunas_entrada", []),
        "colunas_saida": metadados.get("colunas_saida", []),
    }


def _rede_ativa():
    rede = ESTADO["rede"]
    if rede is None and os.path.exists(SAIDA):
        rede = carregar_rede(SAIDA)
        ESTADO["rede"] = rede
    return rede


def api_salvar_como(d):
    rede = _rede_ativa()
    if rede is None:
        return {"erro": "Carregue ou treine uma rede primeiro."}
    nome = os.path.basename(d["nome"].strip())
    if nome in ("", ".", "..") or nome != d["nome"].strip():
        return {"erro": "Informe apenas um nome de arquivo, sem diretórios."}
    if not nome.lower().endswith(".json"):
        nome += ".json"
    caminho = os.path.join(RAIZ, "modelos", nome)
    salvar_rede(rede, caminho)
    return {"arquivo": os.path.relpath(caminho, RAIZ)}


def api_prever(d):
    rede = _rede_ativa()
    if rede is None:
        return {"erro": "Treine uma rede primeiro."}
    ESTADO["rede"] = rede
    return {"saida": Preditor(rede).prever([float(v) for v in d["valores"]])}


ROTAS = {"/api/cabecalho": api_cabecalho, "/api/treinar": api_treinar, "/api/prever": api_prever}
ROTAS.update({"/api/carregar": api_carregar, "/api/salvar-como": api_salvar_como})


class Handler(BaseHTTPRequestHandler):
    def _enviar(self, codigo, corpo, tipo):
        self.send_response(codigo)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(corpo)))
        self.end_headers()
        self.wfile.write(corpo)

    def do_GET(self):
        if self.path == "/api/baixar":
            rede = _rede_ativa()
            if rede is None:
                return self._enviar(404, b'{"erro":"Nenhuma rede carregada."}', "application/json")
            corpo = json.dumps(rede_para_dict(rede), ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="rede.json"')
            self.send_header("Content-Length", str(len(corpo)))
            self.end_headers()
            return self.wfile.write(corpo)
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
        except (ValueError, KeyError, RuntimeError, FileNotFoundError, TypeError, AttributeError, IndexError) as e:
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