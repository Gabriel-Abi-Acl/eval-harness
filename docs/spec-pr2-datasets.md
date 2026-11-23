# Especificação: datasets de exemplo (PR2)

## Requisitos

- FR-1: Três conjuntos em `data/eval/`, cada um com 15 a 20 exemplos em português, cartão versionado e tags `difficulty` e `category`.
- FR-2: QA curta exercita match exato, match normalizado e F1 de tokens.
- FR-3: Classificação de conjunto fechado exercita acurácia, precisão e recall por classe e macro-F1.
- FR-4: Extração de campos JSON exercita `json_field_accuracy`.
- FR-5: O loader do cartão recusa id repetido, rótulo fora do conjunto, JSON de referência inválido, tag fora do esquema e contagem divergente.
- FR-6: O README em pt-BR publica a contagem por conjunto e por tag, a construção e as limitações, sem nota de modelo.

## Critérios de aceite

- AC-1: Dado um diretório com cartão e JSONL válidos, quando `load_carded_dataset` lê o diretório, então os exemplos preservam a ordem e o hash do JSONL.
- AC-2: Dado um rótulo fora do conjunto, uma referência que não é objeto JSON ou uma tag fora do esquema, quando o loader lê o diretório, então o diretório inteiro é recusado.
- AC-3: Dado pares construídos à mão, quando a acurácia e `score_classification` rodam, então precisão, recall e macro-F1 seguem o conjunto fechado. Classe sem exemplo entra com taxa 0. Predição fora do conjunto não abre classe nova.
- AC-4: Dado o README, quando as contagens são lidas dos arquivos, então o texto reproduz essas contagens e declara que nenhum modelo foi executado.

## Casos de borda

- EC-1: Classe do conjunto sem exemplo entra na macro-F1 com precisão, recall e F1 iguais a 0.
- EC-2: `true` não equivale a `1` no JSON de referência, e `1.0` não equivale a `1`.
- EC-3: Campo extra no cartão, versão fora de `major.minor.patch` e `example_count` divergente recusam o cartão.
- EC-4: O placeholder de oito linhas permanece fixture do loader e não é um dos três conjuntos.

## Restrições

- Dados sintéticos escritos para este repositório. Sem chave, sem rede e sem nota de modelo.
- Código em inglês. Cartão, README e esta especificação em pt-BR.
- Relatório agregado e CLI ficam para o PR3.

## Integração

- O exemplo continua no schema `eval_harness.example.v1`. `expected` é a referência. As tags ficam em `metadata.tags`.
- O cartão usa o schema `eval_harness.dataset_card.v1`. O hash do manifesto cobre os exemplos, não o cartão.
- `accuracy` entra no runner como métrica por exemplo. Precisão, recall e macro-F1 saem de `score_classification` e não viram nota por exemplo neste PR.
