# eval-harness

Harness pequeno para avaliar sistemas com LLM fora de um estudo específico. A ideia é reaproveitar o que os estudos ad hoc (`rag-study`, `llm-benchmark`, `guardrails-study` e outros) montavam cada vez: dataset rotulado, sistema sob teste, métrica e um registro reproduzível da execução.

Este repositório ainda não tem resultado experimental. O que está aqui é o núcleo offline e os testes dele.

## Objetivo

Oferecer uma biblioteca estável para três coisas:

1. carregar exemplos rotulados de um JSONL com schema fechado;
2. rodar um sistema (qualquer callable) e anotar saída, notas, latência e erro;
3. identificar a execução por hash do dataset, hash da configuração, seed e versões.

As métricas embutidas são determinísticas. Um juiz de modelo entra só como interface: quem chama fornece o juiz. O pacote não publica nota de modelo.

## Escopo

Neste PR:

- pacote `eval_harness` em `src/`, Python 3.11+, uv, ruff e pytest;
- loader JSONL (`id`, `input`, `expected`, `metadata` opcional);
- protocolo de sistema e um fake determinístico (`echo`, `lookup`, `scripted`);
- métricas `exact_match`, `normalized_match`, `token_f1` e `json_field_accuracy`;
- interface `Judge` / `JudgeMetric`, sem chamada de rede;
- runner e `RunManifest`;
- adaptador OpenAI importado só na primeira chamada, extra opcional;
- oito linhas sintéticas em `data/placeholder.jsonl` para o loader ter um arquivo real;
- CI com lint e testes.

Fora deste PR: CLI, relatório em JSON ou Markdown, datasets rotulados de exemplo e execução contra API. O placeholder não é um conjunto de avaliação.

## Metodologia planejada

A unidade é um exemplo com identificador, entrada e gabarito em texto. O loader recusa o arquivo inteiro se uma linha quebra o schema, para o hash não mudar em silêncio. O hash do dataset é o SHA-256 do JSON canônico dos exemplos (schema, ordem do arquivo, chaves ordenadas). Caminho do arquivo não entra.

O sistema devolve texto e, se souber, contagem de tokens. Ausência de uso fica nula. O runner não preenche token. A latência é o tempo da chamada do sistema, em milissegundos, medido com `time.perf_counter`. Erro de um exemplo guarda a primeira linha da exceção, deixa a nota vazia e segue para o próximo. Erro de métrica guarda a saída já produzida e não roda as métricas seguintes daquele exemplo.

O manifesto guarda `dataset_hash`, `config_hash`, `seed`, versões de Python e do pacote, nome do sistema, nomes das métricas e a quantidade de exemplos. O hash de configuração cobre seed, sistema, ordem das métricas e parâmetros públicos do sistema. Nome de segredo (`api_key`, `token` e equivalentes) é recusado. O seed é argumento de `run_eval`; a variável de ambiente não é lida por dentro do runner. O runner não sorteia: um sistema estocástico, no futuro, é que deve consumir o seed registrado.

Definição operacional das métricas embutidas, sujeita a revisão se um estudo precisar de outra:

- `exact_match`: igualdade de string, 1 ou 0;
- `normalized_match`: NFKC, casefold e espaço colapsado; acento e pontuação permanecem; também 1 ou 0, não é substring;
- `token_f1`: F1 do multiconjunto de tokens depois dessa normalização; os dois lados vazios valem 1;
- `json_field_accuracy`: fração dos campos do objeto gabarito com o mesmo valor JSON; campo extra no candidato não entra; `true` não equivale a `1`; `1` não equivale a `1.0`; texto que não é objeto vale 0.

O fake tem três modos e não inventa texto. `echo` devolve a entrada. `lookup` devolve a tabela pelo id. `scripted` consome uma fita na ordem e não rebobina. Acabou o roteiro, a chamada falha.

O adaptador `OpenAISystem` valida modelo e chave antes de importar o SDK. A importação ocorre na primeira chamada. Temperatura 0. O seed do manifesto ainda não é enviado na requisição. CI não instala o extra e não faz chamada real.

Agregação (média, contagem de erro, tabela) fica para o relatório do PR3. Até lá, nota por exemplo existe na estrutura `RunResult` e não vira número publicado.

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

## Roadmap

- PR2: datasets rotulados de exemplo, no lugar do placeholder de oito linhas.
- PR3: CLI offline e relatórios (resumo JSON e tabela Markdown).
- PR4: execuções reais, com o seed aplicado ao provedor e o registro do que foi pedido.

## O que fica de fora

A suíte não chama API. Este arquivo não traz tabela de modelo, custo ou latência de provedor. O fake não substitui essa tabela: a saída dele verifica o encanamento. Números dos estudos irmãos continuam nos repositórios deles.
