# Especificação: núcleo do eval-harness (PR1)

## Requisitos

- FR-1: Pacote `eval_harness` em `src/`, Python 3.11+, uv, ruff e pytest.
- FR-2: Loader JSONL com schema fechado (`id`, `input`, `expected`, `metadata` opcional).
- FR-3: Sistema é um callable que devolve texto e uso opcional. O fake cobre eco, tabela e fita.
- FR-4: Métricas plugáveis. Embutidas: match exato, match normalizado, F1 de tokens e acurácia de campo JSON. A interface de juiz não chama modelo.
- FR-5: O runner grava saída, notas, latência e erro por exemplo, e um manifesto com hash do dataset, hash da configuração, seed e versões.
- FR-6: O SDK OpenAI só é importado na primeira chamada ao vivo e fica num extra opcional. A chave não entra no manifesto.
- FR-7: CI roda lint e testes com `astral-sh/setup-uv@v10.3.3`. `.env.example` tem valores vazios.

## Critérios de aceite

- AC-1: Dado um JSONL válido, quando o loader lê o arquivo, então a ordem é preservada e o hash não muda com ordem de chave ou espaço fora do valor.
- AC-2: Dado um JSONL com linha inválida, id repetido ou campo desconhecido, quando o loader lê o arquivo, então o arquivo inteiro é recusado com o número da linha.
- AC-3: Dado o mesmo dataset, o mesmo fake e o mesmo seed, quando o runner executa duas vezes, então hashes, saídas, notas e erros coincidem. Outro seed muda só o hash de configuração.
- AC-4: Dado um exemplo que faz o sistema falhar, quando o runner segue, então esse exemplo fica com erro e sem nota, e os demais ainda são registrados.
- AC-5: Dado o pacote importado sem o extra `openai`, quando nada chama o adaptador, então `openai` não é importado. Chave vazia ou extra ausente falha sem repetir a chave.

## Casos de borda

- EC-1: Arquivo vazio, linha em branco, JSON que não é objeto, metadata que não é objeto, id com espaço.
- EC-2: Fita do fake mais curta que o dataset. A fita não rebobina e não inventa texto.
- EC-3: Nota do juiz booleana, fora de [0, 1] ou não numérica.
- EC-4: Uso de tokens ausente permanece nulo. Parâmetro com nome de segredo é recusado.

## Restrições

- Sem rede na suíte. Sem chave no repositório. Sem número de modelo no README.
- Código em inglês. Comentários, docstrings e docs em pt-BR.
- Relatório e CLI ficam para o PR3. Execução real registrada fica para o PR4.

## Integração

- O placeholder em `data/placeholder.jsonl` exercita o loader. Não é conjunto de avaliação.
- Estudos irmãos (`rag-study`, `llm-benchmark`, `guardrails-study`) motivam a interface. Este PR não reproduz os números deles.
