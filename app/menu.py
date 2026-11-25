from mlp import ativacoes_disponiveis, normalizacoes_disponiveis
from mlp.dataset import ler_cabecalho


def _perguntar(texto, padrao=None, obrigatorio=False):
    sufixo = f" [{padrao}]" if padrao not in (None, "") else ""
    while True:
        resp = input(f"{texto}{sufixo}: ").strip()
        if resp:
            return resp
        if padrao is not None:
            return str(padrao)
        if not obrigatorio:
            return ""
        print("  Valor obrigatório.")


def _criar():
    ativs = "/".join(ativacoes_disponiveis())
    n_ent = _perguntar("Quantidade de entradas", obrigatorio=True)
    n_ocultas = int(_perguntar("Quantidade de camadas ocultas", 1))
    ocultas = []
    for i in range(1, n_ocultas + 1):
        n = _perguntar(f"  Camada oculta {i} – neurônios", 8)
        f = _perguntar(f"  Camada oculta {i} – ativação ({ativs})", "tanh")
        ocultas.append(f"{n}:{f}")
    n_sai = _perguntar("Neurônios de saída", 1)
    f_sai = _perguntar(f"Ativação da saída ({ativs})", "linear")
    seed = _perguntar("Seed da inicialização (vazio = aleatória)")
    arquivo = _perguntar("Arquivo de destino", "modelos/rede.json")
    argv = ["criar", "--entradas", n_ent, "--saida", f"{n_sai}:{f_sai}", "-o", arquivo]
    if ocultas:
        argv += ["--ocultas", ",".join(ocultas)]
    if seed:
        argv += ["--seed", seed]
    return argv


def _treinar():
    rede = _perguntar("Arquivo da rede", "modelos/rede.json")
    dados = _perguntar("Arquivo CSV", "datasets/dataset.csv")
    try:
        print(f"  Colunas do arquivo: {', '.join(ler_cabecalho(dados))}")
    except OSError:
        pass
    entradas = _perguntar("Colunas de entrada (vazio = todas menos as de saída)")
    saidas = _perguntar("Colunas de saída (vazio = última coluna)")
    divisao = _perguntar("Percentuais treino,validação,teste", "70,15,15")
    seed_div = _perguntar("Seed da divisão", 42)
    norm = _perguntar(f"Normalização (nenhuma/{'/'.join(normalizacoes_disponiveis())})", "zscore")
    taxa = _perguntar("Taxa de aprendizado", 0.0001)
    epocas = _perguntar("Épocas", 3000)
    seed_tr = _perguntar("Seed do embaralhamento", 42)
    vies = _perguntar("Iniciar viés da saída com a média do treino? (s/n)", "s").lower().startswith("s")
    saida = _perguntar("Salvar rede treinada em", rede)
    hist = _perguntar("CSV do histórico (vazio = não salvar)", "modelos/historico.csv")
    graf = _perguntar("SVG da curva de MSE (vazio = não salvar)", "modelos/curva_mse.svg")
    argv = ["treinar", "--rede", rede, "--dados", dados, "--divisao", divisao,
            "--seed-divisao", seed_div, "--normalizacao", norm, "--taxa", taxa,
            "--epocas", epocas, "--seed-treino", seed_tr, "-o", saida]
    if vies:
        argv.append("--vies-saida-media")
    if entradas:
        argv += ["--entradas", entradas]
    if saidas:
        argv += ["--saidas", saidas]
    if hist:
        argv += ["--historico", hist]
    if graf:
        argv += ["--grafico", graf]
    return argv


def _avaliar():
    rede = _perguntar("Arquivo da rede", "modelos/rede.json")
    dados = _perguntar("Arquivo CSV", "datasets/dataset.csv")
    conj = _perguntar("Conjunto (todos/treino/validacao/teste)", "teste")
    return ["avaliar", "--rede", rede, "--dados", dados, "--conjunto", conj]


def _prever():
    rede = _perguntar("Arquivo da rede", "modelos/rede.json")
    argv = ["prever", "--rede", rede]
    print("Digite os valores de entrada separados por vírgula (linha vazia encerra).")
    while True:
        v = input("  > ").strip()
        if not v:
            break
        argv += ["--valores", v]
    return argv if "--valores" in argv else None


def _info():
    rede = _perguntar("Arquivo da rede", "modelos/rede.json")
    pesos = _perguntar("Mostrar pesos? (s/n)", "n").lower().startswith("s")
    return ["info", "--rede", rede] + (["--pesos"] if pesos else [])


OPCOES = {
    "1": ("Criar nova rede neural", _criar),
    "2": ("Treinar rede", _treinar),
    "3": ("Avaliar rede (MSE)", _avaliar),
    "4": ("Prever com rede treinada", _prever),
    "5": ("Mostrar informações da rede", _info),
}


def executar_menu(executar_comando):
    print("=== Sistema Configurável de Redes Neurais (MLP) ===")
    while True:
        print()
        for k, (nome, _) in OPCOES.items():
            print(f"  {k}. {nome}")
        print("  0. Sair")
        try:
            escolha = input("Opção: ").strip()
        except EOFError:
            return 0
        if escolha == "0":
            return 0
        if escolha not in OPCOES:
            print("Opção inválida.")
            continue
        try:
            argv = OPCOES[escolha][1]()
        except (ValueError, EOFError) as e:
            print(f"ERRO: {e}")
            continue
        except KeyboardInterrupt:
            print("\nOperação cancelada.")
            continue
        if argv:
            print()
            try:
                executar_comando(argv)
            except SystemExit:
                pass  # argparse encerra em argumentos inválidos; volta ao menu
