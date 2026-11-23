# Especificação: CLI e relatórios (PR3)

## Requisitos

- FR-1: `eval-harness run` e `python -m eval_harness` leem um diretório com cartão, um sistema (`oracle`, `echo`, `scripted-noisy` ou `openai`; o padrão é `oracle`), métricas opcionais, seed e diretório de saída.
- FR-2: A execução grava `records.jsonl`, `summary.json` e `report.md`. O padrão cai em `results/<conjunto>/<sistema>/seed-<n>/`, já ignorado pelo git.
- FR-3: O resumo traz média por métrica e por tag, mais o manifesto. `created_at` fica só no manifesto. Exemplo sem nota sai da média. Se ninguém teve nota, a média é nula.
- FR-4: `eval-harness compare` aceita dois ou mais diretórios e grava delta por métrica e por tag. Hash de dataset diferente termina com código 2, sem arquivo, a menos que `--force` seja passado.
- FR-5: `openai` exige `--allow-live`, `OPENAI_MODEL` e `OPENAI_API_KEY`. Sem isso o processo termina com código 2 e não envia requisição.
- FR-6: O mesmo dataset, o mesmo sistema, as mesmas métricas e o mesmo seed repetem o `summary.json` byte a byte, fora `created_at`.
- FR-7: O README em pt-BR explica como rodar, como ler o relatório e como plugar sistema ou métrica. Número de sistema fake é checagem de encanamento, não achado. O arquivo não publica resultado.

## Critérios de aceite

- AC-1: Dado `qa_curta` e os três fakes, quando `run` executa, então os três arquivos existem, o oracle casa com o gabarito nas métricas de igualdade e o eco e a fita barulhenta divergem dele.
- AC-2: Dado dois runs do mesmo conjunto, quando `compare` executa, então o delta é o outro valor menos a base, por métrica e por tag.
- AC-3: Dado `--system openai` sem `--allow-live`, ou com modelo ou chave vazios, quando o CLI roda, então o código é 2, a mensagem diz que não houve requisição e nenhum cliente é construído.
- AC-4: Dados dois runs com `dataset_hash` diferente, quando `compare` roda sem `--force`, então o código é 2 e o diretório de saída não é criado.
- AC-5: Dadas duas execuções com o mesmo seed, o mesmo sistema e o mesmo dataset, quando os resumos são lidos, então coincidem depois de remover `created_at`, e essa chave não aparece fora do manifesto.
- AC-6: Dado um resumo com média nula, quando o Markdown é renderizado, então a célula é `n/a` e a tabela tem cabeçalho e separador. O texto dos fakes diz que o número verifica o encanamento.

## Casos de borda

- EC-1: Todos os exemplos falham. A média de cada métrica é nula e a macro-F1 de classificação não é calculada.
- EC-2: Uma métrica falha depois de outra ter gravado nota. A nota já gravada entra na média dela. A métrica que falhou, e as que não rodaram, ficam de fora. O erro no registro é a primeira linha da exceção.
- EC-3: `--metric macro_f1` ou um nome desconhecido recusa a execução. `macro_f1` aparece no relatório de classificação, não como métrica por exemplo.
- EC-4: `--seed` ganha de `EVAL_HARNESS_SEED`. O runner continua sem ler o ambiente.
- EC-5: A fita `scripted-noisy` é função do gabarito, do índice e do seed. Um terço repete o gabarito, um terço altera, um terço erra. O mesmo seed repete a fita.

## Restrições

- A suíte não chama API e não grava chave. O valor de `OPENAI_API_KEY` não entra em stderr, stdout nem no relatório.
- Código em inglês. README, relatório Markdown e esta especificação em pt-BR.
- `results/` não é versionado. Este PR não publica tabela de nota.
- O seed do manifesto ainda não é enviado ao provedor. Execução real registrada fica para o PR4.

## Integração

- O CLI usa `load_carded_dataset`, `run_eval` e, na classificação com `accuracy`, `score_classification`.
- O manifesto do runner continua sem relógio. O resumo copia esses campos e acrescenta `created_at`.
- Latência fica em `records.jsonl` e não entra no resumo, porque o relógio da chamada não é reproduzível.
