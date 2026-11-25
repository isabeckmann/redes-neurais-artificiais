"""Testes automatizados.  Executar com:  python -m unittest discover -s tests -v"""

import io
import json
import math
import os
import sys
import tempfile
import unittest
import pytest
import csv
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.cli import _aplicar_normalizador, cmd_treinar, cmd_avaliar, cmd_prever
from mlp.dataset import _resolver_colunas
from mlp.normalizacao import ZScore, MinMax
from mlp import (
    Dataset, Preditor, RedeNeural, Treinador, carregar_rede, dividir,
    ler_csv, mse, obter_ativacao, preparar, salvar_rede,
    verificar_compatibilidade,
)
from mlp.erro import erro_quadratico

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(RAIZ, "datasets", "dataset.csv")


class TestAtivacoes(unittest.TestCase):
    def test_derivadas_numericas(self):
        h = 1e-6
        for nome in ("linear", "sigmoid", "tanh", "relu"):
            f = obter_ativacao(nome)
            for z in (-2.0, -0.3, 0.4, 1.7):
                num = (f.calcular(z + h) - f.calcular(z - h)) / (2 * h)
                self.assertAlmostEqual(f.derivada(z, f.calcular(z)), num, places=5, msg=nome)

    def test_sigmoid_estavel(self):
        f = obter_ativacao("sigmoid")
        self.assertAlmostEqual(f.calcular(-1000), 0.0)
        self.assertAlmostEqual(f.calcular(1000), 1.0)


class TestRede(unittest.TestCase):
    def _rede(self, seed=1):
        return RedeNeural.construir(3, [(4, "tanh"), (3, "relu"), (2, "sigmoid")], seed=seed)

    def test_arquitetura_dinamica(self):
        rede = self._rede()
        self.assertEqual([c.n_neuronios for c in rede.camadas], [4, 3, 2])
        self.assertEqual([c.n_entradas for c in rede.camadas], [3, 4, 3])
        self.assertEqual(rede.camadas[1].inicializacao, "he")
        self.assertEqual(rede.camadas[0].inicializacao, "xavier")
        self.assertEqual(len(rede.propagar([0.1, 0.2, 0.3])), 2)

    def test_seed_reprodutivel(self):
        a, b, c = self._rede(7), self._rede(7), self._rede(8)
        pa = [n.pesos for cam in a.camadas for n in cam.neuronios]
        pb = [n.pesos for cam in b.camadas for n in cam.neuronios]
        pc = [n.pesos for cam in c.camadas for n in cam.neuronios]
        self.assertEqual(pa, pb)
        self.assertNotEqual(pa, pc)

    def test_gradientes_conferem_com_diferencas_finitas(self):
        """δ·x calculado pelo backpropagation ≈ derivada numérica do erro."""
        rede = RedeNeural.construir(2, [(3, "tanh"), (2, "sigmoid"), (2, "linear")], seed=3)
        x, t = [0.5, -1.2], [0.3, -0.7]
        rede.propagar(x)
        rede.retropropagar(t)
        analiticos = [(cam, n, i, n.delta * n.entradas[i])
                      for cam in rede.camadas for n in cam.neuronios for i in range(n.n_entradas)]
        h = 1e-6
        for _, n, i, g in analiticos:
            original = n.pesos[i]
            n.pesos[i] = original + h
            e1 = erro_quadratico(rede.propagar(x), t)
            n.pesos[i] = original - h
            e2 = erro_quadratico(rede.propagar(x), t)
            n.pesos[i] = original
            self.assertAlmostEqual(g, (e1 - e2) / (2 * h), places=6)

    def test_deltas_calculados_antes_da_atualizacao(self):
        rede = RedeNeural.construir(2, [(2, "tanh"), (1, "linear")], seed=0)
        x, t = [1.0, 2.0], [3.0]
        rede.propagar(x)
        pesos_saida = list(rede.camada_saida.neuronios[0].pesos)
        rede.retropropagar(t)
        d_saida = rede.camada_saida.neuronios[0].delta
        for j, n in enumerate(rede.camadas[0].neuronios):
            esperado = pesos_saida[j] * d_saida * n.derivada_ativacao()
            self.assertAlmostEqual(n.delta, esperado)

    def test_aprende_xor(self):
        # XOR pode cair em mínimo local com uma semente específica; basta uma de três convergir.
        ds = Dataset(["a", "b"], ["y"], [[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [0]])
        aprendeu = False
        for seed in (0, 1, 2):
            rede = RedeNeural.construir(2, [(4, "tanh"), (1, "sigmoid")], seed=seed)
            Treinador(rede, 0.5, 3000, seed=seed).treinar(ds)
            if all(abs(rede.propagar(x)[0] - t[0]) < 0.15 for x, t in zip(ds.entradas, ds.saidas)):
                aprendeu = True
                break
        self.assertTrue(aprendeu)

    def test_vies_saida(self):
        rede = RedeNeural.construir(2, [(3, "tanh"), (2, "linear")], seed=0)
        rede.definir_vies_saida([10.0, -5.0])
        self.assertEqual([n.vies for n in rede.camada_saida.neuronios], [10.0, -5.0])
        with self.assertRaises(ValueError):
            rede.definir_vies_saida([1.0])

    def test_entrada_incompativel(self):
        with self.assertRaises(ValueError):
            self._rede().propagar([1, 2])


class TestDataset(unittest.TestCase):
    def test_leitura(self):
        ds = ler_csv(CSV)
        self.assertEqual(ds.colunas_entrada, ["sementes_m2", "nitrogenio_kg_ha"])
        self.assertEqual(ds.colunas_saida, ["produtividade_kg_ha"])
        self.assertEqual(len(ds), 179)

    def test_divisao(self):
        ds = ler_csv(CSV)
        d1 = dividir(ds, 70, 15, 15, seed=42)
        d2 = dividir(ds, 70, 15, 15, seed=42)
        self.assertEqual((len(d1.treino), len(d1.validacao), len(d1.teste)), (125, 27, 27))
        self.assertEqual(d1.teste.entradas, d2.teste.entradas)
        with self.assertRaises(ValueError):
            dividir(ds, 70, 20, 20)

    def test_normalizacao_ajustada_so_no_treino(self):
        ds = ler_csv(CSV)
        div = preparar(ds, (60, 20, 20), seed=1, normalizacao="minmax")
        cols = list(zip(*div.treino.entradas))
        for c in cols:
            self.assertAlmostEqual(min(c), 0.0)
            self.assertAlmostEqual(max(c), 1.0)
        bruto = dividir(ds, 60, 20, 20, seed=1)
        self.assertEqual(div.teste.saidas, bruto.teste.saidas)  # saídas não normalizadas
        div_z = preparar(ds, (60, 20, 20), seed=1, normalizacao="zscore")
        for c in zip(*div_z.treino.entradas):
            self.assertAlmostEqual(sum(c) / len(c), 0.0, places=9)

    def test_compatibilidade(self):
        ds = ler_csv(CSV)
        verificar_compatibilidade(RedeNeural.construir(2, [(1, "linear")]), ds)
        with self.assertRaises(ValueError):
            verificar_compatibilidade(RedeNeural.construir(3, [(1, "linear")]), ds)
        with self.assertRaises(ValueError):
            verificar_compatibilidade(RedeNeural.construir(2, [(2, "linear")]), ds)

    def test_impedir_indices_duplicados(self):
        cabecalho = ["idade", "altura", "peso"]

        with pytest.raises(ValueError, match="colunas duplicadas"):
            _resolver_colunas(
                ["0", "0"],
                cabecalho,
                "entrada"
            )
    def test_impedir_colunas_duplicadas(self):
        cabecalho = ["idade", "altura", "peso"]

        with pytest.raises(ValueError, match="colunas duplicadas"):
            _resolver_colunas(
                ["idade", "idade"],
                cabecalho,
                "entrada"
            )
    def test_dividir_dataset_pequeno(self):
        dataset = Dataset(
            ["idade"],
            ["resultado"],
            [[10], [20], [30]],
            [[1], [0], [1]]
        )

        divisao = dividir(dataset, 70, 15, 15, seed=42)

        self.assertEqual(len(divisao.treino), 1)
        self.assertEqual(len(divisao.validacao), 1)
        self.assertEqual(len(divisao.teste), 1)


    def test_dividir_dataset_insuficiente(self):
        dataset = Dataset(
            ["idade"],
            ["resultado"],
            [[10], [20]],
            [[1], [0]]
        )

        with self.assertRaises(ValueError):
            dividir(dataset, 70, 15, 15)
    def test_reutilizar_normalizador_sem_reajustar(self):
        dataset = Dataset(
            ["idade"],
            ["resultado"],
            [[10], [20], [30], [40], [50]],
            [[1], [2], [3], [4], [5]]
        )

        normalizador = ZScore().ajustar(dataset.entradas)

        medias_originais = normalizador.medias.copy()
        desvios_originais = normalizador.desvios.copy()

        divisao = dividir(dataset, 70, 15, 15, seed=42)
        divisao = _aplicar_normalizador(divisao, normalizador)

        self.assertEqual(normalizador.medias, medias_originais)
        self.assertEqual(normalizador.desvios, desvios_originais)
        self.assertIs(divisao.normalizador, normalizador)

    def test_leitura_csv_separado_por_ponto_virgula(self):
        with tempfile.TemporaryDirectory() as pasta:
            caminho = os.path.join(pasta, "dados.csv")

            with open(caminho, "w", encoding="utf-8") as f:
                f.write(
                    "idade;altura;resultado\n"
                    "20;1.70;1\n"
                    "25;1.80;2\n"
                    "30;1.75;3\n"
                )

            dataset = ler_csv(caminho)

        self.assertEqual(
            dataset.colunas_entrada,
            ["idade", "altura"]
        )
        self.assertEqual(dataset.colunas_saida, ["resultado"])

        self.assertEqual(
            dataset.entradas,
            [[20.0, 1.70], [25.0, 1.80], [30.0, 1.75]]
        )

        self.assertEqual(
            dataset.saidas,
            [[1.0], [2.0], [3.0]]
        )
class TestTreinoPersistenciaPrevisao(unittest.TestCase):
    def test_fluxo_completo_no_dataset(self):
        ds = ler_csv(CSV)
        div = preparar(ds, (70, 15, 15), seed=42, normalizacao="zscore")
        rede = RedeNeural.construir(2, [(6, "tanh"), (1, "linear")], seed=42)
        rede.normalizacao = div.normalizador
        mse_inicial = mse(rede, div.treino.entradas, div.treino.saidas)
        assert mse_inicial is not None
        hist = Treinador(rede, 0.01, 200, seed=42).treinar(div.treino, div.validacao)
        self.assertEqual(len(hist), 200)
        self.assertIsNotNone(hist[-1].mse_validacao)
        self.assertLess(hist[-1].mse_treino, mse_inicial / 50)
        mse_teste = mse(rede, div.teste.entradas, div.teste.saidas)
        assert mse_teste is not None
        self.assertTrue(math.isfinite(mse_teste))

        with tempfile.TemporaryDirectory() as pasta:
            caminho = os.path.join(pasta, "rede.json")
            salvar_rede(rede, caminho)
            with open(caminho, encoding="utf-8") as f:
                dados = json.load(f)
            self.assertEqual(dados["normalizacao"]["tipo"], "zscore")
            carregada = carregar_rede(caminho)

        p1, p2 = Preditor(rede), Preditor(carregada)
        for x in ([150, 0], [250, 60], [450, 120], [300, 45]):
            self.assertEqual(p1.prever(x), p2.prever(x))  # saídas idênticas

        pesos_antes = [list(n.pesos) for c in carregada.camadas for n in c.neuronios]
        p2.prever([250, 60])
        self.assertEqual(pesos_antes, [list(n.pesos) for c in carregada.camadas for n in c.neuronios])
        with self.assertRaises(ValueError):
            p2.prever([250])

    def test_persistencia_minmax(self):
        rede = RedeNeural.construir(2, [(3, "tanh"), (1, "linear")], seed=42)

        normalizador = MinMax().ajustar([
            [10, 100],
            [20, 200],
            [30, 300],
        ])

        rede.normalizacao = normalizador

        with tempfile.TemporaryDirectory() as pasta:
            caminho = os.path.join(pasta, "rede_minmax.json")

            salvar_rede(rede, caminho)
            carregada = carregar_rede(caminho)

        self.assertIsInstance(carregada.normalizacao, MinMax)
        self.assertEqual(carregada.normalizacao.minimos, [10, 100])
        self.assertEqual(carregada.normalizacao.maximos, [30, 300])

        p1 = Preditor(rede)
        p2 = Preditor(carregada)

        self.assertEqual(p1.prever([15, 150]), p2.prever([15, 150]))

    def test_persistencia_sem_normalizacao(self):
        rede = RedeNeural.construir(2, [(3, "tanh"), (1, "linear")], seed=42)

        self.assertIsNone(rede.normalizacao)

        with tempfile.TemporaryDirectory() as pasta:
            caminho = os.path.join(pasta, "rede_sem_normalizacao.json")

            salvar_rede(rede, caminho)

            with open(caminho, encoding="utf-8") as f:
                dados = json.load(f)

            carregada = carregar_rede(caminho)

        self.assertIsNone(dados["normalizacao"])
        self.assertIsNone(carregada.normalizacao)

        p1 = Preditor(rede)
        p2 = Preditor(carregada)

        self.assertEqual(p1.prever([15, 150]), p2.prever([15, 150]))

    def test_retreino_preserva_normalizacao_original(self):
    # Prepara uma rede com normalização Z-Score.
        ds = ler_csv(CSV)
        div = preparar(ds, (70, 15, 15), seed=42, normalizacao="zscore")

        rede = RedeNeural.construir(
            2, [(6, "tanh"), (1, "linear")], seed=42
        )
        rede.normalizacao = div.normalizador

    # Simula uma rede que já foi treinada.
        rede.metadados.update({
            "colunas_entrada": ds.colunas_entrada,
            "colunas_saida": ds.colunas_saida,
            "treinamento": {"epocas": 200},
        })

        medias_originais = rede.normalizacao.medias.copy()
        desvios_originais = rede.normalizacao.desvios.copy()

        with tempfile.TemporaryDirectory() as pasta:
            caminho = os.path.join(pasta, "rede.json")
            salvar_rede(rede, caminho)

        # Retreina sem informar normalização e com outra seed de divisão.
            args = SimpleNamespace(
                rede=caminho,
                dados=CSV,
                entradas=None,
                saidas=None,
                divisao="70,15,15",
                seed_divisao=99,
                normalizacao=None,
                taxa=0.001,
                epocas=2,
                sem_embaralhar=True,
                seed_treino=42,
                vies_saida_media=False,
                log_a_cada=1,
                mostrar_teste=0,
                saida_rede=None,
                historico=None,
                grafico=None,
            )

            with redirect_stdout(io.StringIO()):
                resultado = cmd_treinar(args)

            self.assertEqual(resultado, 0)

            carregada = carregar_rede(caminho)

    # Confirma que os parâmetros originais foram preservados.
        self.assertIsInstance(carregada.normalizacao, ZScore)
        self.assertEqual(carregada.normalizacao.medias, medias_originais)
        self.assertEqual(carregada.normalizacao.desvios, desvios_originais)

    # Confirma que a nova seed foi registrada.
        self.assertEqual(
            carregada.metadados["divisao"]["seed"], 99
        )

    def test_impedir_troca_de_normalizacao_no_retreino(self):
        ds = ler_csv(CSV)
        div = preparar(ds, (70, 15, 15), seed=42, normalizacao="zscore")

        rede = RedeNeural.construir(
            2, [(6, "tanh"), (1, "linear")], seed=42
        )
        rede.normalizacao = div.normalizador
        rede.metadados.update({
            "colunas_entrada": ds.colunas_entrada,
            "colunas_saida": ds.colunas_saida,
            "treinamento": {"epocas": 200},
        })

        with tempfile.TemporaryDirectory() as pasta:
            caminho = os.path.join(pasta, "rede.json")
            salvar_rede(rede, caminho)

            args = SimpleNamespace(
                rede=caminho,
                dados=CSV,
                entradas=None,
                saidas=None,
                divisao="70,15,15",
                seed_divisao=99,
                normalizacao="minmax",
                taxa=0.001,
                epocas=2,
                sem_embaralhar=True,
                seed_treino=42,
                vies_saida_media=False,
                log_a_cada=1,
                mostrar_teste=0,
                saida_rede=None,
                historico=None,
                grafico=None,
            )

            with self.assertRaisesRegex(ValueError, "Não é possível retreinar"):
                with redirect_stdout(io.StringIO()):
                    cmd_treinar(args)

        # Confirma que o arquivo original não foi sobrescrito.
            carregada = carregar_rede(caminho)
            self.assertIsInstance(carregada.normalizacao, ZScore)

    def test_impedir_adicionar_normalizacao_no_retreino(self):
        ds = ler_csv(CSV)

        rede = RedeNeural.construir(
            2, [(6, "tanh"), (1, "linear")], seed=42
        )
        rede.metadados.update({
            "colunas_entrada": ds.colunas_entrada,
            "colunas_saida": ds.colunas_saida,
            "treinamento": {"epocas": 200},
        })

        self.assertIsNone(rede.normalizacao)

        with tempfile.TemporaryDirectory() as pasta:
            caminho = os.path.join(pasta, "rede.json")
            salvar_rede(rede, caminho)

            args = SimpleNamespace(
                rede=caminho,
                dados=CSV,
                entradas=None,
                saidas=None,
                divisao="70,15,15",
                seed_divisao=99,
                normalizacao="zscore",
                taxa=0.001,
                epocas=2,
                sem_embaralhar=True,
                seed_treino=42,
                vies_saida_media=False,
                log_a_cada=1,
                mostrar_teste=0,
                saida_rede=None,
                historico=None,
                grafico=None,
            )

            with self.assertRaisesRegex(ValueError, "Não é possível retreinar"):
                with redirect_stdout(io.StringIO()):
                    cmd_treinar(args)

            carregada = carregar_rede(caminho)
            self.assertIsNone(carregada.normalizacao)
class TestComandoAvaliar(unittest.TestCase):

    def test_avaliar_dataset_completo(self):
        ds = ler_csv(CSV)
        div = preparar(ds, (70, 15, 15), seed=42, normalizacao="zscore")

        rede = RedeNeural.construir(
            2, [(6, "tanh"), (1, "linear")], seed=42
        )
        rede.normalizacao = div.normalizador

        rede.metadados.update({
            "colunas_entrada": ds.colunas_entrada,
            "colunas_saida": ds.colunas_saida,
        })

        with tempfile.TemporaryDirectory() as pasta:
            caminho = os.path.join(pasta, "rede.json")
            salvar_rede(rede, caminho)

            args = SimpleNamespace(
                rede=caminho,
                dados=CSV,
                entradas=None,
                saidas=None,
                conjunto="todos",
            )

            saida = io.StringIO()
            with redirect_stdout(saida):
                cmd_avaliar(args)

        resultado = saida.getvalue()

        self.assertIn("Exemplos avaliados:", resultado)
        self.assertIn("MSE:", resultado)
        self.assertIn("RMSE:", resultado)

    def test_avaliar_conjunto_teste(self):
        ds = ler_csv(CSV)
        div = preparar(ds, (70, 15, 15), seed=42, normalizacao="zscore")

        rede = RedeNeural.construir(
            2, [(6, "tanh"), (1, "linear")], seed=42
        )
        rede.normalizacao = div.normalizador

        rede.metadados.update({
            "colunas_entrada": ds.colunas_entrada,
            "colunas_saida": ds.colunas_saida,
            "divisao": {
                "percentuais": [70, 15, 15],
                "seed": 42,
            },
        })

        with tempfile.TemporaryDirectory() as pasta:
            caminho = os.path.join(pasta, "rede.json")
            salvar_rede(rede, caminho)

            args = SimpleNamespace(
                rede=caminho,
                dados=CSV,
                entradas=None,
                saidas=None,
                conjunto="teste",
            )

            saida = io.StringIO()
            with redirect_stdout(saida):
                cmd_avaliar(args)

        resultado = saida.getvalue()

        self.assertIn("Subconjunto 'teste'", resultado)
        self.assertIn("Exemplos avaliados:", resultado)
        self.assertIn("MSE:", resultado)

class TestComandoPrever(unittest.TestCase):

    def test_prever_com_arquivo_csv(self):
        ds = ler_csv(CSV)
        div = preparar(ds, (70, 15, 15), seed=42, normalizacao="zscore")

        rede = RedeNeural.construir(
            2, [(6, "tanh"), (1, "linear")], seed=42
        )
        rede.normalizacao = div.normalizador
        rede.metadados.update({
            "colunas_entrada": ds.colunas_entrada,
            "colunas_saida": ds.colunas_saida,
        })

        with tempfile.TemporaryDirectory() as pasta:
            caminho_rede = os.path.join(pasta, "rede.json")
            caminho_csv = os.path.join(pasta, "entradas.csv")

            salvar_rede(rede, caminho_rede)

            # Cria um CSV contendo somente as entradas.
            with open(caminho_csv, "w", newline="", encoding="utf-8") as f:
                escritor = csv.writer(f)
                escritor.writerow(ds.colunas_entrada)
                escritor.writerow([150, 0])
                escritor.writerow([250, 60])
                escritor.writerow([450, 120])

            args = SimpleNamespace(
                rede=caminho_rede,
                valores=None,
                arquivo=caminho_csv,
            )

            saida = io.StringIO()
            with redirect_stdout(saida):
                cmd_prever(args)

        linhas = [
            linha for linha in saida.getvalue().splitlines()
            if linha.strip()
        ]

        self.assertEqual(len(linhas), 3)

        for nome in ds.colunas_entrada:
            self.assertTrue(any(nome in linha for linha in linhas))

        for linha in linhas:
            self.assertIn("→", linha)

    def test_prever_com_arquivo_separado_por_ponto_virgula(self):
        rede = RedeNeural.construir(
            2, [(6, "tanh"), (1, "linear")], seed=42
        )
        rede.metadados.update({
            "colunas_entrada": ["idade", "altura"],
            "colunas_saida": ["resultado"],
        })

        with tempfile.TemporaryDirectory() as pasta:
            caminho_rede = os.path.join(pasta, "rede.json")
            caminho_csv = os.path.join(pasta, "entradas.csv")

            salvar_rede(rede, caminho_rede)

            with open(caminho_csv, "w", encoding="utf-8") as f:
                f.write(
                    "idade;altura\n"
                    "20;1.70\n"
                    "25;1.80\n"
                    "30;1.75\n"
                )

            args = SimpleNamespace(
                rede=caminho_rede,
                valores=None,
                arquivo=caminho_csv,
            )

            saida = io.StringIO()
            with redirect_stdout(saida):
                cmd_prever(args)

        linhas = [
            linha for linha in saida.getvalue().splitlines()
            if linha.strip()
        ]

        self.assertEqual(len(linhas), 3)
        self.assertTrue(all("→" in linha for linha in linhas))

if __name__ == "__main__":
    unittest.main()
