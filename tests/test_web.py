import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.request
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import web
from mlp import Preditor, RedeNeural, carregar_rede, salvar_rede


class TestInterfaceWeb(unittest.TestCase):
    def setUp(self):
        self.rede_anterior = web.ESTADO["rede"]
        web.ESTADO["rede"] = None

    def tearDown(self):
        web.ESTADO["rede"] = self.rede_anterior

    def test_treinar_com_multiplas_saidas_e_escolher_inicializacao(self):
        csv = (
            "x1,x2,y1,y2\n"
            "1,2,3,4\n"
            "2,3,5,7\n"
            "3,4,7,10\n"
            "4,5,9,13\n"
            "5,6,11,16\n"
            "6,7,13,19\n"
            "7,8,15,22\n"
            "8,9,17,25\n"
        )
        with tempfile.TemporaryDirectory() as pasta:
            caminho = os.path.join(pasta, "rede_web.json")
            with mock.patch.object(web, "SAIDA", caminho):
                resultado = web.api_treinar({
                    "csv": csv,
                    "entradas": ["x1", "x2"],
                    "saidas": ["y1", "y2"],
                    "ocultas": "",
                    "saida": "2:linear",
                    "inicializacao": "he",
                    "seed": 42,
                    "divisao": [50, 25, 25],
                    "seedDivisao": 42,
                    "normalizacao": "",
                    "taxa": "0.001",
                    "epocas": "2",
                    "seedTreino": 42,
                    "viesMedia": False,
                })

            self.assertNotIn("erro", resultado)
            self.assertEqual(resultado["colunas_saida"], ["y1", "y2"])
            self.assertEqual(len(resultado["pares_teste"]), 2)
            self.assertTrue(all(
                len(reais) == len(previstos) == 2
                for reais, previstos in resultado["pares_teste"]
            ))
            self.assertEqual(carregar_rede(caminho).camadas[-1].inicializacao, "he")

    def test_carregar_rede_cli_prever_e_salvar_como(self):
        with tempfile.TemporaryDirectory() as pasta:
            caminho_cli = os.path.join(pasta, "rede_cli.json")
            rede = RedeNeural.construir(
                2, [(2, "linear")], seed=7, inicializacao="xavier"
            )
            salvar_rede(rede, caminho_cli)
            with open(caminho_cli, encoding="utf-8") as arquivo:
                dados_cli = json.load(arquivo)

            with mock.patch.object(web, "RAIZ", pasta):
                carregada = web.api_carregar({"rede": dados_cli})
                self.assertEqual(carregada["n_entradas"], 2)
                self.assertEqual(carregada["n_saidas"], 2)

                valores = [1.5, -2.0]
                previsao = web.api_prever({"valores": valores})["saida"]
                self.assertEqual(previsao, Preditor(rede).prever(valores))

                salva = web.api_salvar_como({"nome": "copia.json"})
                caminho_copia = os.path.join(pasta, salva["arquivo"])
                self.assertTrue(os.path.isfile(caminho_copia))
                self.assertEqual(
                    carregar_rede(caminho_copia).propagar(valores), previsao
                )

    def test_baixar_json_da_rede_ativa(self):
        rede = RedeNeural.construir(1, [(1, "linear")], seed=2)
        web.ESTADO["rede"] = rede
        servidor = web.ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
        thread = threading.Thread(target=servidor.serve_forever)
        thread.start()
        try:
            url = "http://127.0.0.1:{}/api/baixar".format(servidor.server_port)
            with urllib.request.urlopen(url) as resposta:
                dados = json.loads(resposta.read().decode("utf-8"))
                self.assertEqual(resposta.status, 200)
                self.assertIn("rede.json", resposta.headers["Content-Disposition"])
                self.assertEqual(dados["n_entradas"], 1)
                self.assertEqual(dados["camadas"][0]["n_neuronios"], 1)
        finally:
            servidor.shutdown()
            servidor.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
