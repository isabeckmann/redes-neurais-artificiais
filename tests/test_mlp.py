"""Testes automatizados.  Executar com:  python -m unittest discover -s tests -v"""

import json
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlp import (  # noqa: E402
    Dataset, Preditor, RedeNeural, Treinador, carregar_rede, dividir, ler_csv, mse, obter_ativacao,
    preparar, salvar_rede, verificar_compatibilidade,
)
from mlp.erro import erro_quadratico  # noqa: E402

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


class TestTreinoPersistenciaPrevisao(unittest.TestCase):
    def test_fluxo_completo_no_dataset(self):
        ds = ler_csv(CSV)
        div = preparar(ds, (70, 15, 15), seed=42, normalizacao="zscore")
        rede = RedeNeural.construir(2, [(6, "tanh"), (1, "linear")], seed=42)
        rede.normalizacao = div.normalizador
        mse_inicial = mse(rede, div.treino.entradas, div.treino.saidas)
        hist = Treinador(rede, 0.01, 200, seed=42).treinar(div.treino, div.validacao)
        self.assertEqual(len(hist), 200)
        self.assertIsNotNone(hist[-1].mse_validacao)
        self.assertLess(hist[-1].mse_treino, mse_inicial / 50)
        self.assertTrue(math.isfinite(mse(rede, div.teste.entradas, div.teste.saidas)))

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


if __name__ == "__main__":
    unittest.main()
