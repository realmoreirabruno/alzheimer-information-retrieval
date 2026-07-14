# Recuperação de Informação Médica: BM25, RM3 e Cross-Encoder no TREC Clinical Trials 2021

Pipeline de recuperação de informação em múltiplos estágios (BM25 → RM3 → Cross-Encoder), avaliado sobre o conjunto [TREC Clinical Trials 2021](https://trec-cds.org/2021.html) e, em seguida, sobre um estudo de caso construído especificamente para a Doença de Alzheimer.

Trabalho acadêmico desenvolvido por **Bruno Moreira** e **Cauan Gabriel de Souza**.

## Visão geral

O projeto implementa e compara três configurações de recuperação sobre o mesmo corpus de 375.580 ensaios clínicos:

1. **BM25** — busca léxica pura, via Lucene/Pyserini.
2. **BM25 + RM3** — expansão de consulta por Pseudo-Relevance Feedback.
3. **BM25 + RM3 + Cross-Encoder** — re-ranqueamento neural dos candidatos com `cross-encoder/ms-marco-MiniLM-L-6-v2`.

Além da avaliação sobre os 75 tópicos oficiais da track, o projeto inclui um estudo de caso para a Doença de Alzheimer: como nenhum dos tópicos oficiais trata de fato da doença, foram elaboradas 15 consultas adicionais e um gabarito de relevância próprio (construído por pooling e julgamento manual), permitindo comparar o comportamento dos três pipelines em um domínio clínico específico e de vocabulário concentrado.

## Requisitos

- Python 3.12
- Java 21 (necessário para o Pyserini/Lucene)

No macOS, com Homebrew:

```bash
xcode-select --install
brew install openjdk@21
sudo ln -sfn /opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk \
  /Library/Java/JavaVirtualMachines/openjdk-21.jdk
```

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

O pipeline é dividido em fases independentes, executadas por número:

```bash
python3 codigo.py            # roda as fases 1 a 4 em sequência
python3 codigo.py 2 3 4      # roda só as fases indicadas (ex.: pular o download)
```

| Fase | O que faz |
|---|---|
| 1 | Baixa o TREC-CT 2021 (tópicos, qrels e corpus) via `ir_datasets` e gera os arquivos estáticos |
| 2 | Indexa o corpus com Lucene e roda a busca BM25 |
| 3 | Roda a busca com expansão RM3 |
| 4 | Re-ranqueia os candidatos do RM3 com o Cross-Encoder |
| 5 | Roda os três pipelines sobre as consultas de Alzheimer e monta o pool de julgamento |
| 6 | Calcula as métricas dos três pipelines no subconjunto de Alzheimer e mostra a análise de erro |

As fases 5 e 6 exigem que o índice já tenha sido construído (fase 2). A fase 6 exige, além disso, que `alzheimer_qrels.csv` já exista, gerado a partir do julgamento manual de `alzheimer_pool_para_julgar.csv` produzido na fase 5.

As etapas lentas (download e indexação) são puladas automaticamente se os arquivos de saída já existirem.

## Estrutura do repositório

```
codigo.py                          # pipeline completo (fases 1 a 6)
alzheimer_queries.py               # as 15 consultas sintéticas do estudo de caso Alzheimer
alzheimer_pool_para_julgar.csv     # pool de julgamento (top-5 por pipeline, deduplicado)
alzheimer_qrels.csv                # gabarito de relevância julgado manualmente
requirements.txt
```

Arquivos gerados pelo pipeline (corpus, índice, tópicos e candidatos do TREC) não são versionados, por serem grandes e facilmente regeneráveis rodando as fases 1 a 5.

## Resultados

### Conjunto geral (75 tópicos oficiais do TREC-CT 2021)

| Pipeline | NDCG@10 | P@10 | Recall@100 |
|---|---|---|---|
| BM25 | 0,2928 | 0,4067 | 0,1369 |
| BM25 + RM3 | **0,3555** | **0,4680** | **0,1879** |
| BM25 + RM3 + Cross-Encoder | 0,2704 | 0,3667 | 0,1879 |

### Estudo de caso: Doença de Alzheimer (15 consultas, gabarito próprio)

| Pipeline | NDCG@10 | P@10 | Recall@100 |
|---|---|---|---|
| BM25 | 0,4745 | 0,3533 | 0,7233 |
| BM25 + RM3 | 0,4312 | 0,3267 | **0,8377** |
| BM25 + RM3 + Cross-Encoder | **0,6228** | **0,4467** | **0,8377** |

O padrão se inverte entre os dois cenários: no conjunto geral e heterogêneo de tópicos, o RM3 melhora o ranqueamento e o Cross-Encoder o piora; no subconjunto de Alzheimer, de vocabulário mais concentrado, ocorre o oposto, com o Cross-Encoder produzindo o maior ganho do estudo. A discussão completa desses resultados, incluindo a análise de erro qualitativa, está no artigo do projeto.

## Metodologia do gabarito de Alzheimer

Como as 15 consultas de Alzheimer não têm julgamentos de relevância oficiais, o gabarito (`alzheimer_qrels.csv`) foi construído por *pooling*: os 5 documentos mais bem ranqueados por cada um dos três pipelines foram reunidos por consulta, sem duplicatas, formando um pool de 179 pares consulta-documento. Cada par foi julgado de forma binária pelos autores, comparando o ensaio clínico com a necessidade de informação da vinheta correspondente. É um gabarito de menor escala e maior subjetividade do que o processo oficial do TREC, o que é discutido como limitação no artigo do projeto.
