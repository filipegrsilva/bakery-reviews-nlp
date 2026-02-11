# Análise de sentimentos em avaliações on-line: Uma abordagem para a gestão estratégica do relacionamento com o cliente


Pipeline de processamento de avaliações do Google Maps para análise de sentimentos e extração de tópicos em padarias, desenvolvido como parte de dissertação de mestrado na FEA-USP.

## 📋 Visão Geral

Este repositório contém os scripts utilizados para:
1. **Extração de tópicos** - Identificação automática de temas nas avaliações usando BERTopic
2. **Categorização gerencial** - Classificação dos tópicos em categorias de negócio
3. **Análise de sentimentos** - Classificação de sentimentos por categoria usando LLM (Llama 3.1)
4. **Análise de problemas e preços** - Detalhamento de subcategorias, produtos e motivadores
5. **Geração de gráficos** - Visualizações para análise gerencial
6. **Análise posicionamento digitals** - Teste de hipótese entre resposta de proprietários e nota
7. **Geração de figuras**  - Geração de todas as figuras apresentadas no texto
8. **Extrair Coordenadas** - Extração de latitude e longitude de padarias, a partir do review_link
9. **Gerar KML** - Geração da camada KML para Google Maps
## 🗂️ Estrutura do Projeto

```
bakery-reviews-nlp/
├── scripts/
│   ├── 01_extrair_topicos_bertopic.py    # Extração de tópicos com BERTopic
│   ├── 02_aplicar_merges_categorias.py   # Aplicar merges e categorias
│   ├── 03_analise_sentimentos_llm.py     # Análise de sentimentos com LLM
│   ├── 04_analises_categorias.py         # Detalhamento de problemas e preços
│   ├── 05_gerar_graficos_analises.py     # Geração de gráficos de análises
│   ├── 06_analise_posicionamento_digital.py # Análise resposta do dono e Local Guide
│   └── 07_gerar_figuras_dissertacao.py   # Geração de TODAS as figuras (11-30)
│   └── 08_extrair_coordenadas.py         # Extrair coordenadas de cada padaria a partir dos links dos reviews
│   └── 09_gerar_kml_mapa.py              # Gera camada KML para Google Maps
  
├── config/
│   └── exemplo_topicos_para_selecao.json # Exemplo de JSON para curadoria
├── data/
│   └── (dataset de entrada - não incluído)
├── outputs/
│   └── (arquivos gerados)
├── docs/
│   └── (documentação adicional)
├── run_pipeline.sh                        # Script para executar pipeline
├── requirements.txt
├── LICENSE
└── README.md
```

## 🔧 Requisitos

- Python 3.9+
- CUDA (opcional, para GPU)
- Ollama (para análise de sentimentos com LLM)

### Instalação

```bash
# Clonar repositório
git clone https://github.com/seu-usuario/bakery-reviews-nlp.git
cd bakery-reviews-nlp

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt
```

## 🚀 Pipeline de Execução

### Etapa 1: Extração de Tópicos

```bash
python scripts/01_extrair_topicos_bertopic.py
```

**Entrada:** `data/dataset_full.csv`  
**Saída:** 
- `topicos_para_selecao.json` - Tópicos para curadoria
- `dataset_full.csv` - Atualizado com coluna `topic`
- `bertopic_model/` - Modelo salvo

**Tempo estimado:** 2-3 horas

### Etapa 2: Curadoria Manual

Edite o arquivo `topicos_para_selecao.json`:

```json
{
  "topicos": {
    "0": {
      "nome": "cafe_manha_padaria",
      "selecionado": true,
      "categoria": "comida",
      "merge_para": null
    }
  }
}
```

**Campos a editar:**
- `selecionado`: `true` para tópicos relevantes
- `categoria`: `comida`, `atendimento`, `ambiente`, `preco`, `problemas`
- `merge_para`: ID do tópico destino (para unir tópicos similares)

### Etapa 3: Aplicar Merges e Categorias

```bash
python scripts/02_aplicar_merges_categorias.py
```

**Entrada:** `dataset_full.csv`, `topicos_para_selecao.json`  
**Saída:** `dataset_full.csv` com colunas adicionais

**Colunas adicionadas:**
| Coluna | Descrição |
|--------|-----------|
| `topic_original` | Tópico original do BERTopic |
| `topic_final` | Tópico após aplicar merges |
| `categoria` | Categoria gerencial |
| `nome_topic_original` | Nome do tópico |
| `topic_selecionado` | Se foi selecionado na curadoria |

### Etapa 4: Análise de Sentimentos (LLM)

```bash
# Iniciar Ollama (em outro terminal)
ollama serve

# Executar análise
python scripts/03_analise_sentimentos_llm.py
```

**Entrada:** `dataset_full.csv`  
**Saída:** `dataset_com_sentimentos.xlsx`

**Colunas adicionadas:**
| Coluna | Descrição |
|--------|-----------|
| `llm_analise_json` | JSON com análise detalhada |
| `llm_num_categorias` | Número de categorias identificadas |

### Etapa 5: Análise de Problemas e Preços

```bash
python scripts/04_analises_categorias.py
```

**Entrada:** `dataset_com_sentimentos.xlsx`  
**Saída:** 
- `dataset_analises_completas.xlsx` - Dataset final com todas as análises
- `analises_problemas_precos.txt` - Relatório estatístico

**Colunas adicionadas:**
| Coluna | Descrição |
|--------|-----------|
| `problemas_subcategorias` | ATENDIMENTO, DEMORA, PRODUTO, HIGIENE, etc. |
| `problemas_score_medio` | Score médio de sentimento (-1 a +1) |
| `preco_produtos` | Produtos mencionados nas menções de preço |
| `preco_motivadores` | Motivadores da percepção de preço |
| `preco_score_medio` | Score médio de sentimento sobre preço |

### Etapa 6: Geração de Gráficos

```bash
python scripts/05_gerar_graficos_analises.py
```

**Entrada:** `dataset_analises_completas.xlsx`  
**Saída:** 9 gráficos PNG na pasta `outputs/`

**Gráficos gerados:**
1. `fig_problemas_frequencia.png` - Frequência por subcategoria
2. `fig_problemas_gravidade.png` - Score médio por subcategoria
3. `fig_problemas_matriz.png` - Matriz de priorização
4. `fig_coocorrencia_problemas.png` - Heatmap de co-ocorrência
5. `fig_preco_distribuicao_sentimento.png` - Distribuição de sentimento
6. `fig_preco_score_produto.png` - Score por produto
7. `fig_preco_score_motivador.png` - Score por motivador
8. `fig_correlacao_preco_rating.png` - Correlação score × rating
9. `fig_efeito_resposta_dono.png` - Efeito da resposta do dono

### Etapa 7: Análise de Posicionamento Digital

```bash
python scripts/06_analise_posicionamento_digital.py
```

**Entrada:** `dataset_analises_completas.xlsx`  
**Saída:** 5 gráficos PNG na pasta `outputs/`

**Gráficos gerados:**
1. `fig_resposta_rating_medio.png` - Rating médio com/sem resposta do dono
2. `fig_resposta_pct_por_rating.png` - % de respostas por rating
3. `fig_localguide_rating_medio.png` - Rating médio Local Guide vs Não Guide
4. `fig_localguide_distribuicao_rating.png` - Distribuição de ratings por tipo
5. `fig_interacao_resposta_localguide.png` - Interação Resposta × Local Guide

### Etapa 8: Geração de TODAS as Figuras da Dissertação

```bash
python scripts/07_gerar_figuras_dissertacao.py
```

**Entrada:** `dataset_analises_completas.xlsx`  
**Saída:** 20 figuras PNG na pasta `outputs/`

**Figuras geradas (11-30):**

| Figura | Descrição |
|--------|-----------|
| 11 | Distribuição clusters por tópicos (UMAP) |
| 12 | Distribuição clusters por categoria gerencial (UMAP) |
| 13 | Matriz de similaridade semântica entre categorias |
| 14 | Matriz de concordância BERTopic vs LLM |
| 15 | Distribuição de avaliações por número de categorias |
| 16 | Frequência de menções por categoria gerencial |
| 17 | Polaridade de sentimentos por categoria gerencial |
| 18 | Boxplot tamanho das avaliações por sentimento |
| 19 | Curvas de densidade do tamanho por sentimento |
| 20 | Distribuição de notas com/sem resposta do dono |
| 21 | Percentual de resposta por nível de rating |
| 22 | Rating médio Local Guide vs Não Guide |
| 23 | Distribuição de ratings por tipo de usuário |
| 24 | Distribuição de sentimentos por nota atribuída |
| 25 | Score médio de sentimento por nota |
| 26 | Frequência de problemas por subcategoria |
| 27 | Score médio de sentimento por subcategoria |
| 28 | Distribuição dos motivadores por subcategoria |
| 29 | Frequência de menções por subcategoria e motivador |
| 30 | Mapa de priorização de ações corretivas |

## 📊 Categorias Gerenciais

| Categoria | Descrição |
|-----------|-----------|
| **comida** | Qualidade de pães, doces, salgados, café, sabor |
| **atendimento** | Funcionários, rapidez, educação, cordialidade |
| **ambiente** | Limpeza, localização, espaço, decoração, conforto |
| **preco** | Valor, custo-benefício, caro/barato |
| **problemas** | Reclamações, declínio de qualidade, melhorias |

## 📈 Resultados

O pipeline processa ~340.000 avaliações e gera:
- **89 tópicos** identificados pelo BERTopic
- **59 tópicos** após curadoria e consolidação
- **5 categorias gerenciais** para análise

## 🛠️ Configurações

Os scripts possuem seção `CONFIG` no início para personalização:

```python
CONFIG = {
    "input_file": "dataset_full.csv",
    "csv_separator": "|",
    "min_cluster_size": 500,      # BERTopic
    "embedding_model": "all-MiniLM-L6-v2",
    "llm_model": "llama3.1:8b",   # Ollama
}
```

## 📝 Citação

Se utilizar este código, por favor cite:

```bibtex
@mastersthesis{Ramos 2025,
  author  = {Filipe Guilherme Ramos},
  title   = {Análise de sentimentos em avaliações on-line: Uma abordagem para a gestão estratégica do relacionamento com o cliente},
  school  = {FEA-USP},
  year    = {2025},
}
```

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## 🤝 Contribuições

Contribuições são bem-vindas! Abra uma issue ou pull request.

## 📧 Contato

- **Autor:** Filipe Guilherme Ramos 
- **Email:** filipegrs@usp.br
