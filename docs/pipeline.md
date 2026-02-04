# Pipeline de Processamento

Este documento descreve o fluxo completo de processamento das avaliações.

## Diagrama do Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         COLETA DE DADOS                                  │
│                    (Google Maps - Outscraper)                            │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                         dataset_full.csv
                         (~340.000 reviews)
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    ETAPA 1: EXTRAÇÃO DE TÓPICOS                          │
│                    01_extrair_topicos_bertopic.py                        │
│                                                                          │
│  • Limpeza de texto                                                      │
│  • Geração de embeddings (all-MiniLM-L6-v2)                             │
│  • Treinamento BERTopic (HDBSCAN + UMAP)                                │
│  • Geração de JSON para curadoria                                        │
│                                                                          │
│  Tempo: ~2-3 horas                                                       │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                    topicos_para_selecao.json
                         (89 tópicos)
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      CURADORIA MANUAL                                    │
│                                                                          │
│  • Selecionar tópicos relevantes                                         │
│  • Atribuir categorias gerenciais                                        │
│  • Definir merges de tópicos similares                                   │
│                                                                          │
│  Resultado: 59 tópicos curados                                           │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                 ETAPA 2: APLICAR MERGES E CATEGORIAS                     │
│                 02_aplicar_merges_categorias.py                          │
│                                                                          │
│  • Aplicar merges recursivos                                             │
│  • Mapear categorias gerenciais                                          │
│  • Adicionar colunas ao dataset                                          │
│                                                                          │
│  Tempo: ~1 minuto                                                        │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                    dataset_full.csv (atualizado)
                    + colunas: topic_final, categoria
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                ETAPA 3: ANÁLISE DE SENTIMENTOS (LLM)                     │
│                03_analise_sentimentos_llm.py                             │
│                                                                          │
│  • Processar cada review individualmente                                 │
│  • Extrair categorias mencionadas                                        │
│  • Classificar sentimento por categoria                                  │
│  • Identificar evidências textuais                                       │
│                                                                          │
│  Modelo: Llama 3.1 8B (via Ollama)                                       │
│  Tempo: ~10-20 horas                                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                  dataset_com_sentimentos.xlsx
                  + colunas: llm_analise_json, llm_num_categorias
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              ETAPA 5: ANÁLISE DE PROBLEMAS E PREÇOS                      │
│              04_analises_categorias.py                                   │
│                                                                          │
│  • Classificar PROBLEMAS em subcategorias via LLM                        │
│  • Extrair PRODUTO e MOTIVADOR de menções de preço                       │
│  • Calcular scores médios de sentimento                                  │
│  • Gerar relatório estatístico                                           │
│                                                                          │
│  Modelo: Llama 3.1 8B (via Ollama)                                       │
│  Tempo: ~2-4 horas                                                       │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                  dataset_analises_completas.xlsx
                  + colunas: problemas_*, preco_*
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    ETAPA 6: GERAÇÃO DE GRÁFICOS                          │
│                    05_gerar_graficos_analises.py                         │
│                                                                          │
│  • 4 gráficos de PROBLEMAS (frequência, gravidade, matriz, heatmap)      │
│  • 5 gráficos de PREÇO (sentimento, produtos, motivadores, correlação)   │
│                                                                          │
│  Tempo: ~1 minuto                                                        │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                        outputs/*.png
                        (9 gráficos)
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              ETAPA 7: ANÁLISE DE POSICIONAMENTO DIGITAL                  │
│              06_analise_posicionamento_digital.py                        │
│                                                                          │
│  • Impacto da resposta do dono no rating                                 │
│  • Comportamento de Local Guides vs usuários comuns                      │
│  • Interação entre resposta do dono e Local Guide                        │
│  • Testes estatísticos (t-test)                                          │
│                                                                          │
│  Tempo: ~1 minuto                                                        │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                        outputs/*.png
                        (5 gráficos adicionais)
```

## Categorias Gerenciais

| Categoria | Descrição | Exemplos de palavras-chave |
|-----------|-----------|---------------------------|
| **comida** | Qualidade dos produtos | pão, café, doce, salgado, sabor, fresco |
| **atendimento** | Interação com funcionários | garçom, funcionário, rápido, educado |
| **ambiente** | Infraestrutura do local | limpo, decoração, confortável, espaço |
| **preco** | Percepção de valor | caro, barato, justo, vale a pena |
| **problemas** | Reclamações e falhas | demora, frio, errado, piorou |

## Arquivos Gerados

### Etapa 1
- `topicos_para_selecao.json` - Tópicos para curadoria
- `bertopic_model/` - Modelo BERTopic salvo
- `topics.pkl` - Tópicos atribuídos
- `df_com_topicos.pkl` - DataFrame com índices originais

### Etapa 2
- `dataset_full.csv` - Dataset atualizado
- `dataset_full.pkl` - Versão pickle (mais rápido)
- Backup automático com timestamp

### Etapa 3
- `dataset_com_sentimentos.xlsx` - Resultado final
- `dataset_com_sentimentos.pkl` - Versão pickle
- `checkpoint_llm.json` - Progresso (para retomada)
- `log_analise_llm.txt` - Log de execução

## Formato do JSON de Análise LLM

```json
{
  "analises": [
    {
      "categoria": "comida",
      "sentimento": "positivo",
      "evidencia": "Pão de queijo maravilhoso, sempre fresquinho"
    },
    {
      "categoria": "preco",
      "sentimento": "negativo",
      "evidencia": "achei o preço um pouco salgado"
    }
  ]
}
```

## Configurações Principais

### BERTopic
- `min_cluster_size`: 500
- `min_samples`: 50
- `n_components` (UMAP): 5
- `n_neighbors` (UMAP): 15
- Embedding: all-MiniLM-L6-v2 (384 dimensões)

### LLM
- Modelo: Llama 3.1 8B
- Temperature: 0.1
- Max tokens: 500
- Timeout: 90s

## Requisitos de Hardware

| Etapa | RAM | GPU | Tempo |
|-------|-----|-----|-------|
| Etapa 1 (BERTopic) | 32GB+ | Opcional | 2-3h |
| Etapa 2 (Merges) | 8GB | Não | 1min |
| Etapa 3 (LLM) | 16GB+ | Recomendado | 10-20h |

## Troubleshooting

### Erro de memória na Etapa 1
- Reduza `batch_size` de embeddings
- Use `--low-memory` flag se disponível

### Ollama não conecta na Etapa 3
```bash
# Verificar se Ollama está rodando
curl http://localhost:11434/api/tags

# Iniciar Ollama
ollama serve

# Baixar modelo
ollama pull llama3.1:8b
```

### Retomar processamento interrompido
O script 03 salva checkpoints automaticamente e retoma do último ponto salvo.
