"""Interface de linha de comando da aplicação.

Subcomandos:
    criar    – monta uma nova arquitetura e salva a rede inicializada
    treinar  – treina uma rede salva com um dataset CSV (SGD)
    avaliar  – calcula o MSE de uma rede sobre um dataset (ou subconjunto)
    prever   – executa a rede sobre novos valores de entrada
    info     – mostra a arquitetura de uma rede salva
    menu     – modo interativo (padrão quando nenhum subcomando é informado)
"""

import argparse
import csv
import os
import sys

from mlp import (
    Preditor, RedeNeural, Treinador, TreinamentoDivergiu, ativacoes_disponiveis, carregar_rede,
    dividir, ler_csv, mse, normalizacoes_disponiveis, preparar, salvar_rede, verificar_compatibilidade,
)
from mlp.inicializacao import METODOS

from .graficos import salvar_curva_mse_svg


# ---------------------------------------------------------------- utilitários
def _lista(texto):
    """'a, b ,c' → ['a', 'b', 'c'] (lista vazia para texto vazio/None)."""
    if not texto:
        return []
    return [p.strip() for p in str(texto).split(",") if p.strip()]


def _numeros(texto):
    try:
        return [float(v) for v in _lista(texto.replace(";", ","))]
    except ValueError:
        raise ValueError(f"Valores inválidos: '{texto}'. Use números separados por vírgula.") from None


def _camadas(texto, nome):
    """'8:tanh,4:relu' → [(8, 'tanh'), (4, 'relu')]."""
    camadas = []
    for item in _lista(texto):
        try:
            qtd, ativ = item.split(":")
            qtd = int(qtd)
        except ValueError:
            raise ValueError(
                f"Formato inválido em {nome}: '{item}'. Use NEURONIOS:ATIVACAO, ex.: 8:tanh"
            ) from None
        if qtd < 1:
            raise ValueError(f"{nome}: a quantidade de neurônios deve ser >= 1 ('{item}').")
        camadas.append((qtd, ativ.strip().lower()))
    return camadas


def _fmt(v):
    return "—" if v is None else f"{v:.6g}"


# ------------------------------------------------------------------ comandos
def cmd_criar(a):
    ocultas = _camadas(a.ocultas, "--ocultas")
    saida = _camadas(a.saida, "--saida")
    if len(saida) != 1:
        raise ValueError("--saida deve descrever exatamente uma camada, ex.: 1:linear")
    inicializacao = None if a.inicializacao == "auto" else a.inicializacao
    rede = RedeNeural.construir(a.entradas, ocultas + saida, seed=a.seed, inicializacao=inicializacao)
    salvar_rede(rede, a.arquivo)
    print("Rede criada:")
    print(rede.descrever())
    print(f"\nSalva em: {a.arquivo}")


def cmd_treinar(a):
    rede = carregar_rede(a.rede)
    meta = rede.metadados
    entradas = _lista(a.entradas) or meta.get("colunas_entrada")
    saidas = _lista(a.saidas) or meta.get("colunas_saida")
    dataset = ler_csv(a.dados, entradas, saidas)
    verificar_compatibilidade(rede, dataset)

    percentuais = _numeros(a.divisao)
    if len(percentuais) != 3:
        raise ValueError("--divisao deve ter três percentuais: TREINO,VALIDACAO,TESTE")
    if "treinamento" in meta:
        print("Aviso: a rede já foi treinada; o treinamento continuará a partir dos pesos atuais.")
    divisao = preparar(dataset, percentuais, seed=a.seed_divisao, normalizacao=a.normalizacao)

    print(f"Dataset: {a.dados}  ({len(dataset)} exemplos)")
    print(f"Entradas: {', '.join(dataset.colunas_entrada)}  |  Saídas: {', '.join(dataset.colunas_saida)}")
    print(f"Divisão ({'/'.join(f'{p:g}' for p in percentuais)}%, seed={a.seed_divisao}): {divisao.resumo()}")
    print(f"Normalização: {divisao.normalizador.nome if divisao.normalizador else 'nenhuma'}")
    print(f"Arquitetura: {rede!r}")
    print(f"SGD: taxa={a.taxa:g}  épocas={a.epocas}  embaralhar={not a.sem_embaralhar}  seed={a.seed_treino}\n")
    if len(divisao.validacao) == 0:
        print("Aviso: conjunto de validação vazio; o MSE de validação não será calculado.")

    intervalo = a.log_a_cada or max(1, a.epocas // 20)

    def ao_fim_da_epoca(r):
        if r.epoca == 1 or r.epoca % intervalo == 0 or r.epoca == a.epocas:
            print(f"{r.epoca:>7} {_fmt(r.mse_treino):>16} {_fmt(r.mse_validacao):>16}")

    # A normalização passa a fazer parte da rede (usada na previsão e salva no arquivo).
    rede.normalizacao = divisao.normalizador
    if a.vies_saida_media and "treinamento" not in meta:
        n = len(divisao.treino)
        medias = [sum(col) / n for col in zip(*divisao.treino.saidas)]
        rede.definir_vies_saida(medias)
        print(f"Viés da camada de saída inicializado com a média do treino: {', '.join(f'{m:.6g}' for m in medias)}\n")
    print(f"{'Época':>7} {'MSE treino':>16} {'MSE validação':>16}")
    treinador = Treinador(
        rede, a.taxa, a.epocas, embaralhar=not a.sem_embaralhar, seed=a.seed_treino,
        ao_fim_da_epoca=ao_fim_da_epoca,
    )
    try:
        historico = treinador.treinar(divisao.treino, divisao.validacao)
    except TreinamentoDivergiu as e:
        print(f"\nERRO: {e}")
        return 1

    final = historico[-1]
    mse_teste = mse(rede, divisao.teste.entradas, divisao.teste.saidas)
    print("\nResultado final")
    print(f"  MSE treino:    {_fmt(final.mse_treino)}  (RMSE {_fmt(final.mse_treino ** 0.5)})")
    if final.mse_validacao is not None:
        print(f"  MSE validação: {_fmt(final.mse_validacao)}  (RMSE {_fmt(final.mse_validacao ** 0.5)})")
    if mse_teste is not None:
        print(f"  MSE teste:     {_fmt(mse_teste)}  (RMSE {_fmt(mse_teste ** 0.5)})")

    if a.mostrar_teste and len(divisao.teste):
        print(f"\nAmostras do conjunto de teste (esperado → previsto):")
        for x, t in list(zip(divisao.teste.entradas, divisao.teste.saidas))[: a.mostrar_teste]:
            y = rede.propagar(x)
            print("  " + ", ".join(f"{ti:.4g} → {yi:.4g}" for ti, yi in zip(t, y)))

    rede.metadados.update({
        "colunas_entrada": dataset.colunas_entrada,
        "colunas_saida": dataset.colunas_saida,
        "dataset": os.path.abspath(a.dados),
        "divisao": {"percentuais": percentuais, "seed": a.seed_divisao, "tamanhos": {
            "treino": len(divisao.treino), "validacao": len(divisao.validacao), "teste": len(divisao.teste)}},
        "treinamento": {"algoritmo": "SGD", "taxa_aprendizado": a.taxa, "epocas": a.epocas,
                        "embaralhar": not a.sem_embaralhar, "seed": a.seed_treino,
                        "vies_saida_media": a.vies_saida_media},
        "resultado": {"mse_treino": final.mse_treino, "mse_validacao": final.mse_validacao,
                      "mse_teste": mse_teste},
    })
    destino = a.saida_rede or a.rede
    salvar_rede(rede, destino)
    print(f"\nRede treinada salva em: {destino}")

    for arquivo in (a.historico, a.grafico):
        if arquivo:
            os.makedirs(os.path.dirname(os.path.abspath(arquivo)), exist_ok=True)
    if a.historico:
        with open(a.historico, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoca", "mse_treino", "mse_validacao"])
            for r in historico:
                w.writerow([r.epoca, repr(r.mse_treino), "" if r.mse_validacao is None else repr(r.mse_validacao)])
        print(f"Histórico por época salvo em: {a.historico}")
    if a.grafico:
        salvar_curva_mse_svg(historico, a.grafico)
        print(f"Gráfico da curva de MSE salvo em: {a.grafico}")
    return 0


def cmd_avaliar(a):
    rede = carregar_rede(a.rede)
    meta = rede.metadados
    entradas = _lista(a.entradas) or meta.get("colunas_entrada")
    saidas = _lista(a.saidas) or meta.get("colunas_saida")
    dataset = ler_csv(a.dados, entradas, saidas)
    verificar_compatibilidade(rede, dataset)

    if a.conjunto != "todos":
        info = meta.get("divisao")
        if not info:
            raise ValueError("A rede não registra a divisão usada no treino; use --conjunto todos.")
        div = dividir(dataset, *info["percentuais"], seed=info["seed"])
        dataset = {"treino": div.treino, "validacao": div.validacao, "teste": div.teste}[a.conjunto]
        print(f"Subconjunto '{a.conjunto}' reproduzido com seed={info['seed']} e percentuais={info['percentuais']}.")

    preditor = Preditor(rede)
    x = [preditor.preparar_entrada(v) for v in dataset.entradas]
    erro = mse(rede, x, dataset.saidas)
    print(f"Exemplos avaliados: {len(dataset)}")
    print(f"MSE:  {_fmt(erro)}")
    if erro is not None:
        print(f"RMSE: {_fmt(erro ** 0.5)}")


def cmd_prever(a):
    rede = carregar_rede(a.rede)
    preditor = Preditor(rede)
    nomes_ent = rede.metadados.get("colunas_entrada") or [f"x{i + 1}" for i in range(rede.n_entradas)]
    nomes_sai = rede.metadados.get("colunas_saida") or [f"y{i + 1}" for i in range(rede.n_saidas)]

    lotes = [_numeros(v) for v in (a.valores or [])]
    if a.arquivo:
        # Usa as colunas com os nomes registrados no treino; se não existirem, todas as colunas.
        with open(a.arquivo, newline="", encoding="utf-8-sig") as f:
            leitor = csv.reader(f)
            cab = [c.strip() for c in next(leitor)]
            if all(n in cab for n in nomes_ent):
                idx = [cab.index(n) for n in nomes_ent]
            else:
                idx = list(range(len(cab)))
            for linha in leitor:
                if any(c.strip() for c in linha):
                    lotes.append(_numeros(",".join(linha[i] for i in idx)))
    if not lotes:
        raise ValueError("Informe valores com --valores (ex.: --valores 250,60) ou --arquivo.")

    for valores in lotes:
        saida = preditor.prever(valores)
        ent = ", ".join(f"{n}={v:g}" for n, v in zip(nomes_ent, valores))
        sai = ", ".join(f"{n}={v:.6g}" for n, v in zip(nomes_sai, saida))
        print(f"{ent}  →  {sai}")


def cmd_info(a):
    rede = carregar_rede(a.rede)
    print(rede.descrever())
    meta = rede.metadados
    if meta.get("colunas_entrada"):
        print(f"Colunas de entrada: {', '.join(meta['colunas_entrada'])}")
        print(f"Colunas de saída:   {', '.join(meta['colunas_saida'])}")
    if meta.get("treinamento"):
        t = meta["treinamento"]
        print(f"Treinamento: {t['algoritmo']}, taxa={t['taxa_aprendizado']}, épocas={t['epocas']}")
    if meta.get("resultado"):
        r = meta["resultado"]
        print(f"MSE final: treino={_fmt(r['mse_treino'])}  validação={_fmt(r['mse_validacao'])}  teste={_fmt(r['mse_teste'])}")
    if a.pesos:
        for i, c in enumerate(rede.camadas, start=1):
            print(f"\nCamada {i} ({c.ativacao.nome}):")
            for j, n in enumerate(c.neuronios, start=1):
                print(f"  N{j}: vies={n.vies:.6g}  pesos=[{', '.join(f'{w:.6g}' for w in n.pesos)}]")


# -------------------------------------------------------------------- parser
def criar_parser():
    ativs = ", ".join(ativacoes_disponiveis())
    p = argparse.ArgumentParser(
        prog="python main.py",
        description="Sistema configurável de Redes Neurais Artificiais (MLP).",
    )
    sub = p.add_subparsers(dest="comando")

    c = sub.add_parser("criar", help="cria uma nova rede (arquitetura configurável)")
    c.add_argument("--entradas", type=int, required=True, help="quantidade de entradas")
    c.add_argument("--ocultas", default="", help=f"camadas ocultas NEURONIOS:ATIVACAO separadas por vírgula, ex.: 8:tanh,4:relu ({ativs})")
    c.add_argument("--saida", required=True, help="camada de saída NEURONIOS:ATIVACAO, ex.: 1:linear")
    c.add_argument("--seed", type=int, default=None, help="semente da inicialização dos pesos")
    c.add_argument("--inicializacao", choices=("auto",) + METODOS, default="auto",
                   help="auto = He para ReLU e Xavier para as demais (padrão)")
    c.add_argument("-o", "--arquivo", required=True, help="arquivo JSON de destino")
    c.set_defaults(func=cmd_criar)

    t = sub.add_parser("treinar", help="treina uma rede com um dataset CSV (SGD)")
    t.add_argument("--rede", required=True, help="arquivo JSON da rede")
    t.add_argument("--dados", required=True, help="arquivo CSV")
    t.add_argument("--entradas", help="colunas de entrada (nomes ou índices base 0, separados por vírgula)")
    t.add_argument("--saidas", help="colunas de saída (padrão: última coluna)")
    t.add_argument("--divisao", default="70,15,15", help="percentuais TREINO,VALIDACAO,TESTE (padrão 70,15,15)")
    t.add_argument("--seed-divisao", type=int, default=None, help="semente da divisão do dataset")
    t.add_argument("--normalizacao", default="nenhuma", choices=["nenhuma"] + normalizacoes_disponiveis())
    t.add_argument("--taxa", type=float, default=0.01, help="taxa de aprendizado (padrão 0.01)")
    t.add_argument("--epocas", type=int, default=500)
    t.add_argument("--seed-treino", type=int, default=None, help="semente do embaralhamento a cada época")
    t.add_argument("--sem-embaralhar", action="store_true", help="não embaralhar os exemplos a cada época")
    t.add_argument("--vies-saida-media", action="store_true",
                   help="inicia o viés da saída com a média das saídas do conjunto de treino "
                        "(recomendado quando as saídas têm magnitude grande)")
    t.add_argument("--log-a-cada", type=int, default=0, help="imprime o MSE a cada N épocas")
    t.add_argument("--mostrar-teste", type=int, default=5, help="nº de exemplos de teste exibidos ao final")
    t.add_argument("-o", "--saida-rede", help="arquivo onde salvar a rede treinada (padrão: sobrescreve --rede)")
    t.add_argument("--historico", help="CSV com o MSE de treino e validação por época")
    t.add_argument("--grafico", help="SVG com a curva de MSE por época")
    t.set_defaults(func=cmd_treinar)

    v = sub.add_parser("avaliar", help="calcula o MSE da rede sobre um dataset")
    v.add_argument("--rede", required=True)
    v.add_argument("--dados", required=True)
    v.add_argument("--entradas", help="padrão: colunas registradas no treinamento")
    v.add_argument("--saidas", help="padrão: colunas registradas no treinamento")
    v.add_argument("--conjunto", choices=("todos", "treino", "validacao", "teste"), default="todos",
                   help="reproduz a divisão do treino e avalia só o subconjunto escolhido")
    v.set_defaults(func=cmd_avaliar)

    r = sub.add_parser("prever", help="executa a rede treinada sobre novos valores")
    r.add_argument("--rede", required=True)
    r.add_argument("--valores", action="append", help="valores de entrada separados por vírgula (pode repetir)")
    r.add_argument("--arquivo", help="CSV com cabeçalho contendo as entradas a prever")
    r.set_defaults(func=cmd_prever)

    i = sub.add_parser("info", help="mostra a arquitetura e os metadados de uma rede salva")
    i.add_argument("--rede", required=True)
    i.add_argument("--pesos", action="store_true", help="lista também pesos e vieses")
    i.set_defaults(func=cmd_info)

    m = sub.add_parser("menu", help="modo interativo")
    m.set_defaults(func=None)
    return p


def main(argv=None):
    parser = criar_parser()
    args = parser.parse_args(argv)
    if args.comando in (None, "menu"):
        from .menu import executar_menu
        return executar_menu(main)
    try:
        return args.func(args) or 0
    except (ValueError, FileNotFoundError, RuntimeError, KeyError) as e:
        print(f"ERRO: {e}", file=sys.stderr)
        return 1
