# Sistema Configurável de Redes Neurais Artificiais (MLP)

Aplicação para construir, configurar, treinar, salvar, carregar e utilizar redes
Multilayer Perceptron. Os algoritmos (propagação direta, MSE, retropropagação e
descida do gradiente) são implementados do zero em **Python puro** — apenas a
biblioteca padrão é usada (`csv`, `json`, `math`, `random`, `argparse`,
`http.server`).

Há três formas de usar a mesma biblioteca: **linha de comando**, **menu
interativo** e **interface web local** (treino, curvas de MSE e previsão no navegador).

Requisito: Python 3.8 ou superior. Não há dependências a instalar.

## Estrutura

```
main.py                 ponto de entrada (CLI, menu interativo e servidor web)
mlp/                    biblioteca (núcleo, sem interface)
  ativacoes.py          1.4  Linear, Sigmoid, Tanh, ReLU + derivadas
  inicializacao.py      1.5  Xavier/Glorot e He, com seed
  neuronio.py           1.1  pesos, viés, z, saída e delta (∂E/∂z)
  camada.py             1.2  agrupa neurônios; deltas da saída e das ocultas
  rede.py               1.3/1.6/1.8  arquitetura dinâmica, forward e backpropagation
  erro.py               1.7  MSE e gradiente do erro
  normalizacao.py       1.9  Min-Max e Z-score
  dataset.py            1.9  leitura de CSV, compatibilidade, divisão e preparação
  treinamento.py        1.10 treinamento por SGD, MSE de treino/validação por época
  persistencia.py       1.11 salvar/carregar em JSON
  previsao.py           1.12 previsão com normalização registrada
app/                    interface
  cli.py                subcomandos criar / treinar / avaliar / prever / info / web / menu
  menu.py               menu interativo (monta e executa os mesmos subcomandos)
  graficos.py           curva de MSE em SVG
  web.py                servidor HTTP local (API JSON) que reaproveita a biblioteca mlp
  web.html              página única da interface web (HTML/CSS/JS, sem dependências)
datasets/dataset.csv    dataset fornecido (179 exemplos: sementes_m2, nitrogenio_kg_ha, produtividade_kg_ha)
modelos/                redes, históricos e gráficos gerados (exemplos incluídos)
  rede_web.json         rede salva pela interface web (sobrescrita a cada treino)
tests/test_mlp.py       testes automatizados
```

## Uso

### Menu interativo

```
python main.py
```

Pergunta os parâmetros (com valores padrão sugeridos) e executa os mesmos
subcomandos da linha de comando. Opções: 1 criar rede, 2 treinar, 3 avaliar (MSE),
4 prever, 5 mostrar informações da rede, 0 sair.

### Interface web

```
python main.py web                # http://127.0.0.1:8000
python main.py web --porta 8080   # outra porta
```

Abra o endereço no navegador. A página permite:

- escolher um CSV da pasta `datasets/` ou enviar outro arquivo, e marcar as colunas de entrada e de saída;
- definir camadas ocultas e de saída (`neurônios:ativação`), seeds, divisão treino/validação/teste, normalização, taxa, épocas e o viés de saída pela média do treino;
- treinar e ver o MSE de treino/validação/teste, a curva de MSE por época e o gráfico previsto × real do conjunto de teste (junto com a referência de quem sempre prevê a média);
- prever novos valores com a última rede treinada.

A rede treinada é salva em `modelos/rede_web.json` (sobrescrito a cada treino). O
servidor escuta apenas em `127.0.0.1` e usa só a biblioteca padrão; encerre com
`Ctrl+C`. A API usada pela página: `GET /api/datasets`, `POST /api/cabecalho`,
`POST /api/treinar` e `POST /api/prever`.

### Linha de comando

**1. Criar uma arquitetura** (nada é fixo no código: entradas, camadas ocultas,
neurônios por camada, ativação de cada camada e neurônios de saída são parâmetros):

```
python main.py criar --entradas 2 --ocultas 8:tanh --saida 1:linear --seed 42 -o modelos/rede.json
python main.py criar --entradas 2 --ocultas 16:relu,8:tanh --saida 1:linear --seed 7 -o modelos/rede2.json
```

`--inicializacao auto` (padrão) usa He em camadas ReLU e Xavier nas demais.

**2. Treinar** (SGD, atualização após cada exemplo):

```
python main.py treinar --rede modelos/rede.json --dados datasets/dataset.csv ^
    --entradas sementes_m2,nitrogenio_kg_ha --saidas produtividade_kg_ha ^
    --divisao 70,15,15 --seed-divisao 42 --normalizacao zscore ^
    --taxa 0.0001 --epocas 3000 --seed-treino 42 --vies-saida-media ^
    -o modelos/rede_treinada.json --historico modelos/historico.csv --grafico modelos/curva_mse.svg
```

(no PowerShell use `` ` `` no lugar de `^` para quebrar linhas, ou escreva tudo em uma linha)

- `--entradas`/`--saidas` aceitam nomes ou índices (base 0). Padrão: última coluna é a saída.
- A cada época são exibidos o MSE de treino e de validação; ao final, o MSE de teste.
- `--historico` grava o MSE por época em CSV e `--grafico` desenha a curva em SVG.
- `--normalizacao` aceita `nenhuma` (padrão), `minmax` ou `zscore`; `--divisao` recebe `TREINO,VALIDACAO,TESTE` (padrão `70,15,15`).
- `--sem-embaralhar` desliga o embaralhamento por época; `--log-a-cada N` controla a frequência do log; `--mostrar-teste N` exibe N exemplos de teste (esperado → previsto).
- `-o/--saida-rede` define onde salvar a rede treinada (sem ele, sobrescreve `--rede`).
- Treinar uma rede já treinada continua a partir dos pesos atuais (com aviso). Se o MSE divergir, o treino é interrompido com erro.

**3. Avaliar** (MSE sobre o arquivo inteiro ou sobre um subconjunto; a divisão é
reproduzida a partir da seed registrada na rede):

```
python main.py avaliar --rede modelos/rede_treinada.json --dados datasets/dataset.csv --conjunto teste
```

**4. Prever** (apenas propagação direta; normalização aplicada automaticamente):

```
python main.py prever --rede modelos/rede_treinada.json --valores 250,60 --valores 300,45
python main.py prever --rede modelos/rede_treinada.json --arquivo novos_dados.csv
```

Com `--arquivo`, o CSV precisa de cabeçalho; são usadas as colunas com os nomes de
entrada registrados no treino (ou, se não existirem, todas as colunas).

**5. Inspecionar** uma rede salva:

```
python main.py info --rede modelos/rede_treinada.json --pesos
```

### Testes

```
python -m unittest discover -s tests -v
```

São 14 testes. Incluem verificação do gradiente do backpropagation por diferenças finitas,
reprodutibilidade por seed, XOR, divisão/normalização do dataset e
equivalência das saídas após salvar e carregar. A interface web não tem testes automatizados.

## Decisões de implementação

**Erro.** Para um exemplo com *m* saídas: `E = (1/m)·Σ(y_k − t_k)²`, logo
`∂E/∂y_k = (2/m)·(y_k − t_k)`. O MSE de um conjunto é a média de `E` sobre os exemplos.

**Retropropagação.** Saída: `δ_k = ∂E/∂y_k · f'(z_k)`. Ocultas:
`δ_j = (Σ_k w_kj·δ_k) · f'(z_j)`. Gradientes: `∂E/∂w_ji = δ_j·x_i`, `∂E/∂b_j = δ_j`.
`RedeNeural.retropropagar` calcula **todos** os deltas e só depois
`RedeNeural.atualizar_parametros` aplica `w ← w − η·∂E/∂w`, garantindo que os
deltas das ocultas usem os mesmos pesos da propagação direta.

**Inicialização.** Xavier uniforme `U(±√(6/(fan_in+fan_out)))` para
Linear/Sigmoid/Tanh e He normal `N(0, √(2/fan_in))` para ReLU. Vieses iniciam em 0.
Cada execução usa um `random.Random(seed)` próprio.

**Divisão e normalização.** Embaralhamento com seed; percentuais devem somar
100%. A normalização é ajustada só no treino e aplicada aos três subconjuntos;
as saídas nunca são normalizadas. A normalização (tipo e parâmetros) passa a
fazer parte da rede e é salva no JSON, sendo reaplicada na previsão.

**Viés da saída com a média do treino (`--vies-saida-media`).** Neste dataset a
saída está em torno de 2500 kg/ha e não pode ser normalizada. Começando com viés
0, o erro inicial (~2500) gera gradientes enormes que saturam os neurônios tanh
ou fazem a ReLU divergir. Com essa opção, o viés de cada neurônio de saída
começa com a média das saídas do **conjunto de treino** (sem vazar informação de
validação/teste), a rede parte prevendo a média e aprende só o desvio. É opcional
e fica registrada nos metadados.

**Interface web.** `app/web.py` apenas orquestra `mlp` (ler CSV, construir, dividir, treinar,
salvar, prever); nenhum algoritmo é reimplementado. Os mesmos parâmetros da CLI estão
disponíveis, exceto `--inicializacao` (usa o método recomendado por ativação).

**Referência para o dataset.** A variância das produtividades é ≈ 61.600 (MSE de
quem sempre prevê a média) e a variância *dentro* de cada combinação
sementes × nitrogênio é ≈ 20.700 — este é o piso de erro devido ao ruído, que
nenhum modelo com essas duas entradas consegue superar. Com a configuração do
exemplo acima, o MSE de treino fica tipicamente entre 20.000 e 30.000. Como o
teste tem só 27 exemplos, o MSE de teste varia bastante conforme a seed da divisão.

## Formato do arquivo da rede (JSON)

```json
{
  "formato": "mlp-json",
  "versao": 1,
  "n_entradas": 2,
  "n_camadas": 2,
  "camadas": [
    {
      "n_neuronios": 8,
      "ativacao": "tanh",
      "inicializacao": "xavier",
      "neuronios": [ { "pesos": [0.41, -0.77], "vies": 0.03 }, "..." ]
    },
    {
      "n_neuronios": 1,
      "ativacao": "linear",
      "inicializacao": "xavier",
      "neuronios": [ { "pesos": [120.5, "..."], "vies": 2541.2 } ]
    }
  ],
  "normalizacao": {
    "tipo": "zscore",
    "parametros": { "medias": [300.1, 59.8], "desvios": [111.9, 42.4] }
  },
  "metadados": {
    "colunas_entrada": ["sementes_m2", "nitrogenio_kg_ha"],
    "colunas_saida": ["produtividade_kg_ha"],
    "divisao": { "percentuais": [70, 15, 15], "seed": 42 },
    "treinamento": { "algoritmo": "SGD", "taxa_aprendizado": 0.0001, "epocas": 3000 },
    "resultado": { "mse_treino": 0, "mse_validacao": 0, "mse_teste": 0 }
  }
}
```

| Campo | Conteúdo |
|---|---|
| `n_entradas` | quantidade de entradas |
| `camadas` | camadas ocultas em ordem; a **última** é a de saída |
| `camadas[i].ativacao` | `linear`, `sigmoid`, `tanh` ou `relu` |
| `camadas[i].neuronios[j].pesos` | um peso por entrada da camada |
| `camadas[i].neuronios[j].vies` | viés do neurônio |
| `normalizacao` | `null`, ou `minmax` (`minimos`, `maximos`) ou `zscore` (`medias`, `desvios` populacionais), calculados no treino |
| `metadados` | informativo (colunas, divisão, hiperparâmetros, MSE final); não é necessário para reconstruir a rede |

Os números são gravados com precisão total, então uma rede carregada produz
exatamente as mesmas saídas que a original.
