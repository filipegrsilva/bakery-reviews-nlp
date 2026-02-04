# bakery-reviews-nlp

Pipeline de processamento de avaliações do Google Maps para análise de sentimentos e extração de tópicos em padarias, desenvolvido como parte de dissertação de mestrado na FEA-USP.

## 📋 Visão Geral

Este repositório contém os scripts utilizados para:
1. **Extração de tópicos** - Identificação automática de temas nas avaliações usando BERTopic
2. **Categorização gerencial** - Classificação dos tópicos em categorias de negócio
3. **Análise de sentimentos** - Classificação de sentimentos por categoria usando LLM (Llama 3.1)

## 🗂️ Estrutura do Projeto

```
bakery-reviews-nlp/
├── scripts/
│   ├── 01_extrair_topicos_bertopic.py    # Extração de tópicos com BERTopic
│   ├── 02_aplicar_merges_categorias.py   # Aplicar merges e categorias
│   └── 03_analise_sentimentos_llm.py     # Análise de sentimentos com LLM
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
@mastersthesis{sobrenome2025,
  author  = {Seu Nome},
  title   = {Análise de Sentimentos em Avaliações de Padarias},
  school  = {FEA-USP},
  year    = {2025},
}
```

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## 🤝 Contribuições

Contribuições são bem-vindas! Abra uma issue ou pull request.

## 📧 Contato

- **Autor:** Filipe
- **Email:** seu-email@usp.br
- **LinkedIn:** [seu-perfil](https://linkedin.com/in/seu-perfil)
