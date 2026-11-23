# eval-harness

Harness pequeno para avaliar sistemas com LLM fora de um estudo específico. A ideia é reaproveitar o que os estudos ad hoc (`rag-study`, `llm-benchmark`, `guardrails-study` e outros) montavam cada vez: dataset rotulado, sistema sob teste, métrica e um registro reproduzível da execução.

Este repositório ainda não tem resultado experimental. O que está aqui é o núcleo offline, três conjuntos sintéticos e os testes deles.

## Objetivo

Oferecer uma biblioteca estável para três coisas:

1. carregar exemplos rotulados de um JSONL com schema fechado;
2. rodar um sistema (qualquer callable) e anotar saída, notas, latência e erro;
3. identificar a execução por hash do dataset, hash da configuração, seed e versões.

As métricas embutidas são determinísticas. Um juiz de modelo entra só como interface: quem chama fornece o juiz. O pacote não publica nota de modelo.

## Escopo

O repositório contém:

- pacote `eval_harness` em `src/`, Python 3.11+, uv, ruff e pytest;
- loader JSONL (`id`, `input`, `expected`, `metadata` opcional) e cartão por conjunto;
- protocolo de sistema e um fake determinístico (`echo`, `lookup`, `scripted`);
- métricas `exact_match`, `normalized_match`, `token_f1`, `accuracy` e `json_field_accuracy`;
- precisão, recall por classe e macro-F1 sobre um conjunto fechado de rótulos;
- interface `Judge` / `JudgeMetric`, sem chamada de rede;
- runner e `RunManifest`;
- adaptador OpenAI importado só na primeira chamada, extra opcional;
- oito linhas sintéticas em `data/placeholder.jsonl` para o loader ter um arquivo real;
- três conjuntos versionados em `data/eval/`, com cartão e validação estrita;
- CI com lint e testes.

Fora deste repositório, por enquanto: CLI, relatório em JSON ou Markdown e execução contra API. O placeholder não é um conjunto de avaliação. Os conjuntos ficam em `data/eval/`.

## Metodologia planejada

A unidade é um exemplo com identificador, entrada e gabarito em texto. O loader recusa o arquivo inteiro se uma linha quebra o schema, para o hash não mudar em silêncio. O hash do dataset é o SHA-256 do JSON canônico dos exemplos (schema, ordem do arquivo, chaves ordenadas). Caminho do arquivo não entra. O cartão não entra nesse hash: a versão do cartão cobre a regra de anotação.

O sistema devolve texto e, se souber, contagem de tokens. Ausência de uso fica nula. O runner não preenche token. A latência é o tempo da chamada do sistema, em milissegundos, medido com `time.perf_counter`. Erro de um exemplo guarda a primeira linha da exceção, deixa a nota vazia e segue para o próximo. Erro de métrica guarda a saída já produzida e não roda as métricas seguintes daquele exemplo.

O manifesto guarda `dataset_hash`, `config_hash`, `seed`, versões de Python e do pacote, nome do sistema, nomes das métricas e a quantidade de exemplos. O hash de configuração cobre seed, sistema, ordem das métricas e parâmetros públicos do sistema. Nome de segredo (`api_key`, `token` e equivalentes) é recusado. O seed é argumento de `run_eval`; a variável de ambiente não é lida por dentro do runner. O runner não sorteia: um sistema estocástico, no futuro, é que deve consumir o seed registrado.

Definição operacional das métricas embutidas, sujeita a revisão se um estudo precisar de outra:

- `exact_match`: igualdade de string, 1 ou 0;
- `normalized_match`: NFKC, casefold e espaço colapsado; acento e pontuação permanecem; também 1 ou 0, não é substring;
- `token_f1`: F1 do multiconjunto de tokens depois dessa normalização; os dois lados vazios valem 1;
- `accuracy`: igualdade da string inteira com o rótulo, 1 ou 0. A comparação é a mesma de `exact_match`; o nome marca a tarefa de classe fechada. A acurácia do conjunto é a média dessas notas;
- precisão e recall por classe, e `macro_f1`: calculados por `score_classification` sobre os gabaritos, as saídas e o conjunto fixo de rótulos. Cada classe do conjunto entra na macro, também quando não há exemplo dela. Sem caso no denominador, a taxa vale 0. Predição fora do conjunto não abre classe nova: entra como erro na acurácia e como falso negativo do gabarito. O runner grava a `accuracy` por exemplo. A tabela agregada fica para o relatório do PR3;
- `json_field_accuracy`: fração dos campos do objeto gabarito com o mesmo valor JSON; campo extra no candidato não entra; `true` não equivale a `1`; `1` não equivale a `1.0`; texto que não é objeto vale 0.

O fake tem três modos e não inventa texto. `echo` devolve a entrada. `lookup` devolve a tabela pelo id. `scripted` consome uma fita na ordem e não rebobina. Acabou o roteiro, a chamada falha.

O adaptador `OpenAISystem` valida modelo e chave antes de importar o SDK. A importação ocorre na primeira chamada. Temperatura 0. O seed do manifesto ainda não é enviado na requisição. CI não instala o extra e não faz chamada real.

Agregação no relatório (média, contagem de erro, tabela) fica para o PR3. Até lá, nota por exemplo existe na estrutura `RunResult` e não vira número publicado. `score_classification` devolve a conta do conjunto para quem quiser inspecionar um rótulo fechado sem esperar o relatório.

## Datasets de exemplo

Três conjuntos sintéticos em `data/eval/`, cada um com cartão (`card.json`, schema `eval_harness.dataset_card.v1`) e exemplos (`examples.jsonl`). O exemplo continua no schema do núcleo: `id`, `input`, `expected` e `metadata`. A referência é o campo `expected`. As tags ficam em `metadata.tags`, com as chaves `difficulty` e `category`. Os valores de tag estão em português. O placeholder de oito linhas continua em `data/placeholder.jsonl` e não entra nesta lista.

O cartão traz nome, versão, tipo de tarefa, métricas, nota de licença, nota de construção, esquema de tags e a contagem de exemplos. `load_carded_dataset` recusa o diretório inteiro quando o id repete, a tag sai do esquema, o rótulo de classificação sai do conjunto ou o JSON de referência não fecha com os campos e os tipos declarados. As regras estão em [docs/spec-pr2-datasets.md](docs/spec-pr2-datasets.md). São 54 exemplos no total, 18 em cada conjunto.

| conjunto | exemplos | dificuldade | categoria |
| --- | ---: | --- | --- |
| qa_curta | 18 | fácil 8, médio 5, difícil 5 | geografia 4, aritmética 4, ciências 4, língua 3, tempo 3 |
| classificacao_tom | 18 | fácil 9, médio 6, difícil 3 | produto 6, entrega 6, atendimento 6 |
| extracao_pedido | 18 | fácil 9, médio 6, difícil 3 | papelaria 6, mercado 6, farmácia 6 |

| conjunto | gabarito |
| --- | --- |
| classificacao_tom | positivo 6, negativo 6, neutro 6 |
| extracao_pedido | urgente true 7, urgente false 11 |

### Como foram construídos

Os três conjuntos foram escritos à mão para este repositório, versão 1.0.0 em cada cartão. Não foram amostrados de um modelo, não foram traduzidos de um benchmark e não copiam item de conjunto público. Cada gabarito tem um autor só. Não houve segunda anotação nem medida de acordo. A nota de licença registra texto sintético de uso livre, sem trecho de terceiros.

`qa_curta` é pergunta de resposta curta. As métricas alvo são `exact_match`, `normalized_match` e `token_f1`. O gabarito é uma string canônica: fato público ou conta fechada, com uma forma escrita. Acento faz parte dessa string. Sinônimo não foi listado.

`classificacao_tom` classifica o tom de uma frase curta no conjunto `positivo`, `negativo`, `neutro`. As métricas alvo são `accuracy`, precisão e recall por classe e `macro_f1`. Três itens de dificuldade `difícil` estão na fronteira, e a nota de construção registra a decisão do autor.

`extracao_pedido` extrai um pedido para um objeto com `produto`, `quantidade` e `urgente`. A métrica alvo é `json_field_accuracy`. A referência é JSON com essas chaves, nessa ordem. A nota de construção define o singular do produto, o inteiro da quantidade e as pistas de urgência, inclusive o caso `não é urgente`.

### Limitações

O tamanho é de demonstração: 18 exemplos por tarefa, em português, com um esquema cada. Não representa tráfego real. A fronteira de tom e a singularização dos produtos são convenção do autor, descrita no cartão. Pergunta sem resposta, resposta longa e JSON de esquema aberto não aparecem. Sinônimo de uma resposta curta conta como divergência nas métricas de string. Nenhum modelo foi executado nestes conjuntos. As tabelas acima contam exemplos e gabaritos. Não há nota de sistema.

## Como rodar

Requisitos: Python 3.11+ e [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

A configuração é por variável de ambiente. O arquivo versionado é só `.env.example`, com valores vazios. Não há `.env` no repositório, e o código não lê esse arquivo. Para um shell local:

```bash
cp .env.example .env
set -a && source .env && set +a
```

`EVAL_HARNESS_SEED` vazia vale 0 em `load_settings`. `OPENAI_API_KEY` e `OPENAI_MODEL` vazias fazem o adaptador recusar a chamada, sem imprimir o valor. O extra, se for usado fora da suíte:

```bash
uv sync --extra openai
```

O trecho abaixo só exercita o fake de eco. Não é resultado de modelo. O hash impresso identifica o dataset e a configuração; não é uma nota.

```python
from eval_harness import ExactMatch, FakeSystem, TokenF1, load_dataset, run_eval

dataset = load_dataset("data/placeholder.jsonl")
result = run_eval(dataset, FakeSystem(mode="echo"), [ExactMatch(), TokenF1()], seed=0)
print(result.manifest.dataset_hash)
print(result.manifest.config_hash)
print(result.manifest.example_count)
```

O trecho seguinte só lê um conjunto versionado. Também não é resultado de modelo.

```python
from eval_harness import load_carded_dataset

carded = load_carded_dataset("data/eval/qa_curta")
print(carded.card.name)
print(carded.card.version)
print(carded.card.task_type)
print(len(carded.dataset.examples))
```

## Roadmap

- PR2, neste repositório: três conjuntos sintéticos versionados em `data/eval/`, com cartão e métricas de classificação.
- PR3: CLI offline e relatórios (resumo JSON e tabela Markdown). A média e a macro-F1 ainda não viram relatório.
- PR4: execuções reais, com o seed aplicado ao provedor e o registro do que foi pedido.

## O que fica de fora

A suíte não chama API. Este arquivo não traz tabela de modelo, custo ou latência de provedor. O fake não substitui essa tabela: a saída dele verifica o encanamento. Os conjuntos de exemplo descrevem gabarito e contagem, não uma execução. Números dos estudos irmãos continuam nos repositórios deles.
