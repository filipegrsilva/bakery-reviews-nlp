#!/usr/bin/env python3
"""
================================================================================
GERAÇÃO DE TODAS AS FIGURAS DA DISSERTAÇÃO
================================================================================

Este script gera TODAS as figuras de resultados empíricos (11-30) da dissertação.

FIGURAS GERADAS:
----------------
CATEGORIZAÇÃO E TÓPICOS:
  Fig 11 - Distribuição clusters por tópicos (UMAP)
  Fig 12 - Distribuição clusters por categoria gerencial (UMAP)
  Fig 13 - Matriz de similaridade semântica entre categorias
  Fig 14 - Matriz de concordância BERTopic vs LLM

DISTRIBUIÇÃO E FREQUÊNCIA:
  Fig 15 - Distribuição de avaliações por número de categorias
  Fig 16 - Frequência de menções por categoria gerencial
  Fig 17 - Polaridade de sentimentos por categoria gerencial

TAMANHO DO REVIEW:
  Fig 18 - Boxplot tamanho das avaliações por sentimento
  Fig 19 - Curvas de densidade do tamanho por sentimento

POSICIONAMENTO DIGITAL:
  Fig 20 - Distribuição de notas com/sem resposta do dono
  Fig 21 - Percentual de resposta por nível de rating
  Fig 22 - Rating médio Local Guide vs Não Guide
  Fig 23 - Distribuição de ratings por tipo de usuário

NOTA VS SENTIMENTO:
  Fig 24 - Distribuição de sentimentos por nota atribuída
  Fig 25 - Score médio de sentimento por nota

ANÁLISE DE PROBLEMAS:
  Fig 26 - Frequência de problemas por subcategoria
  Fig 27 - Score médio de sentimento por subcategoria
  Fig 28 - Distribuição dos motivadores por subcategoria
  Fig 29 - Frequência de menções por subcategoria e motivador
  Fig 30 - Mapa de priorização de ações corretivas

ENTRADA:
  - dataset_analises_completas.xlsx (saída do script 04)

SAÍDA:
  - 20 arquivos PNG na pasta outputs/

EXECUÇÃO:
  python 07_gerar_figuras_dissertacao.py

================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from collections import Counter
from scipy import stats
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import re
import gc
import warnings
warnings.filterwarnings('ignore')

# Tentar importar bibliotecas opcionais para UMAP
try:
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠ sentence-transformers ou umap não instalados. Figuras 11-12 serão puladas.")

# ============================================================
# CONFIGURAÇÕES GLOBAIS
# ============================================================

CONFIG = {
    "input_file": "dataset_analises_completas.xlsx",
    "output_dir": "outputs",
    "dpi": 300,
}

# Paleta de cores - Categorias
CORES_CATEGORIAS = {
    'comida': '#66c2a5',       # Verde menta
    'atendimento': '#8da0cb',  # Azul lavanda
    'ambiente': '#a6d854',     # Verde limão
    'preco': '#ffd92f',        # Amarelo
    'problemas': '#e78ac3',    # Rosa
}

# Paleta de cores - Sentimentos
CORES_SENTIMENTO = {
    'positivo': '#66c2a5',
    'Positivo': '#2E7D32',
    'neutro': '#8da0cb',
    'Neutro': '#F9A825',
    'negativo': '#fc8d62',
    'Negativo': '#C62828'
}

# Paleta de cores - Problemas
CORES_PROBLEMAS = {
    'ATENDIMENTO': '#e74c3c',
    'DEMORA': '#e67e22',
    'PRODUTO': '#f39c12',
    'HIGIENE': '#9b59b6',
    'INFRAESTRUTURA': '#3498db',
    'FALTA': '#1abc9c',
    'COBRANCA': '#34495e',
    'OUTRO': '#95a5a6'
}

ORDEM_CATEGORIAS = ['comida', 'atendimento', 'ambiente', 'preco', 'problemas']
ORDEM_SENTIMENTO = ['positivo', 'neutro', 'negativo']

# Configuração visual global
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
sns.set_style("whitegrid")

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def formatar_numero(n):
    """Formata número com separador de milhar brasileiro."""
    return f"{n:,.0f}".replace(",", ".")


def sanitizar_label(texto):
    """Remove caracteres especiais de labels."""
    if pd.isna(texto):
        return texto
    texto = str(texto)
    texto = texto.replace('$', '')
    texto = re.sub(r'[_^{}\\]', '', texto)
    return texto.strip() if texto.strip() else 'indefinido'


def extrair_sentimento_predominante(json_str):
    """Extrai o sentimento predominante de uma avaliação."""
    if pd.isna(json_str):
        return None
    try:
        parsed = json.loads(json_str)
        sentimentos = [a.get('sentimento', '').lower().strip() 
                      for a in parsed.get('analises', [])]
        sentimentos = [s for s in sentimentos if s in ['positivo', 'negativo', 'neutro']]
        if not sentimentos:
            return None
        contagem = Counter(sentimentos)
        return contagem.most_common(1)[0][0]
    except:
        return None


def extrair_todos_sentimentos(json_str):
    """Extrai lista de todos os sentimentos de uma avaliação."""
    if pd.isna(json_str):
        return []
    try:
        parsed = json.loads(json_str)
        sentimentos = [a.get('sentimento', '').lower().strip() 
                      for a in parsed.get('analises', [])]
        return [s for s in sentimentos if s in ['positivo', 'negativo', 'neutro']]
    except:
        return []


def calcular_score_sentimento(json_str):
    """Calcula score: +1 (positivo), 0 (neutro), -1 (negativo)."""
    sentimentos = extrair_todos_sentimentos(json_str)
    if not sentimentos:
        return None
    score_map = {'positivo': 1, 'neutro': 0, 'negativo': -1}
    scores = [score_map[s] for s in sentimentos]
    return np.mean(scores)


def extrair_categorias(json_str):
    """Extrai lista de categorias de uma avaliação."""
    if pd.isna(json_str):
        return []
    try:
        parsed = json.loads(json_str)
        categorias = [a.get('categoria', '').lower().strip() 
                     for a in parsed.get('analises', [])]
        return [c for c in categorias if c in ORDEM_CATEGORIAS]
    except:
        return []


def extrair_primeira_categoria_llm(json_str):
    """Extrai a primeira categoria do LLM."""
    try:
        dados = json.loads(json_str)
        analises = dados.get('analises', [])
        if analises:
            return analises[0].get('categoria')
    except:
        pass
    return None


# ============================================================
# FIGURAS 15-17: DISTRIBUIÇÃO DE CATEGORIAS E SENTIMENTOS
# ============================================================

def gerar_fig15_distribuicao_categorias(df, output_dir):
    """Fig 15: Distribuição de avaliações por número de categorias."""
    print("  Gerando: fig_15_distribuicao_num_categorias.png")
    
    # Contar número de categorias por review
    df['num_categorias'] = df['llm_analise_json'].apply(
        lambda x: len(extrair_categorias(x))
    )
    
    # Criar distribuição
    dist_cat = df['num_categorias'].value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(dist_cat.index, dist_cat.values, color='#1565C0', edgecolor='white')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                formatar_numero(height),
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Número de categorias identificadas')
    ax.set_ylabel('')
    ax.set_yticks([])
    ax.set_title('Figura 15 - Distribuição do número de categorias por avaliação')
    sns.despine(left=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_15_distribuicao_num_categorias.png'), 
                dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()


def gerar_fig16_frequencia_categorias(df, output_dir):
    """Fig 16: Frequência de menções por categoria gerencial."""
    print("  Gerando: fig_16_frequencia_categorias.png")
    
    # Contar menções por categoria
    todas_categorias = []
    for json_str in df['llm_analise_json'].dropna():
        todas_categorias.extend(extrair_categorias(json_str))
    
    freq_cat = pd.Series(Counter(todas_categorias))
    freq_cat = freq_cat.reindex(ORDEM_CATEGORIAS).fillna(0).astype(int)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cores = [CORES_CATEGORIAS.get(cat, '#95a5a6') for cat in freq_cat.index]
    bars = ax.barh(freq_cat.index.str.capitalize(), freq_cat.values, color=cores, edgecolor='white')
    ax.invert_yaxis()
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1000, bar.get_y() + bar.get_height()/2.,
                formatar_numero(width),
                ha='left', va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_title('Figura 16 - Frequência de menções por categoria gerencial')
    sns.despine(left=True, bottom=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_16_frequencia_categorias.png'),
                dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()


def gerar_fig17_polaridade_sentimentos(df, output_dir):
    """Fig 17: Polaridade de sentimentos por categoria gerencial."""
    print("  Gerando: fig_17_polaridade_sentimentos.png")
    
    # Extrair cruzamento categoria x sentimento
    dados_cruzamento = []
    for json_str in df['llm_analise_json'].dropna():
        try:
            parsed = json.loads(json_str)
            for a in parsed.get('analises', []):
                cat = a.get('categoria', '').lower().strip()
                sent = a.get('sentimento', '').lower().strip()
                if cat in ORDEM_CATEGORIAS and sent in ORDEM_SENTIMENTO:
                    dados_cruzamento.append({'categoria': cat, 'sentimento': sent})
        except:
            continue
    
    df_cruz = pd.DataFrame(dados_cruzamento)
    
    # Calcular proporções
    crosstab = pd.crosstab(df_cruz['categoria'], df_cruz['sentimento'], normalize='index') * 100
    crosstab = crosstab.reindex(ORDEM_CATEGORIAS)[ORDEM_SENTIMENTO]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    left = np.zeros(len(ORDEM_CATEGORIAS))
    cores_sent = ['#2E7D32', '#F9A825', '#C62828']  # Positivo, Neutro, Negativo
    
    for sent, cor in zip(ORDEM_SENTIMENTO, cores_sent):
        valores = crosstab[sent].values
        bars = ax.barh([c.capitalize() for c in ORDEM_CATEGORIAS], valores, 
                      left=left, label=sent.capitalize(), color=cor)
        
        for i, (val, l) in enumerate(zip(valores, left)):
            if val > 5:
                ax.text(l + val/2, i, f'{val:.1f}%',
                       ha='center', va='center', fontsize=10,
                       fontweight='bold', color='white')
        left += valores
    
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('Figura 17 - Distribuição de sentimentos por categoria gerencial')
    ax.legend(title='Sentimento', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    sns.despine(left=True, bottom=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_17_polaridade_sentimentos.png'),
                dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()


# ============================================================
# FIGURAS 18-19: TAMANHO DO REVIEW VS SENTIMENTO
# ============================================================

def gerar_fig18_boxplot_tamanho(df, output_dir):
    """Fig 18: Boxplot tamanho das avaliações por sentimento."""
    print("  Gerando: fig_18_boxplot_tamanho_sentimento.png")
    
    # Calcular métricas
    df_temp = df.copy()
    df_temp['n_chars'] = df_temp['review_text'].astype(str).apply(len)
    df_temp['sentimento_pred'] = df_temp['llm_analise_json'].apply(extrair_sentimento_predominante)
    
    # Filtrar válidos e outliers extremos
    df_valid = df_temp[df_temp['sentimento_pred'].notna()].copy()
    df_plot = df_valid[df_valid['n_chars'] <= df_valid['n_chars'].quantile(0.99)].copy()
    df_plot['Sentimento'] = df_plot['sentimento_pred'].map({
        'positivo': 'Positivo', 'neutro': 'Neutro', 'negativo': 'Negativo'
    })
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.boxplot(
        data=df_plot,
        x='Sentimento',
        y='n_chars',
        order=['Positivo', 'Neutro', 'Negativo'],
        palette=['#66c2a5', '#8da0cb', '#fc8d62'],
        ax=ax,
        showfliers=False
    )
    
    # Adicionar médias
    medias = df_plot.groupby('Sentimento')['n_chars'].mean()
    for i, sent in enumerate(['Positivo', 'Neutro', 'Negativo']):
        if sent in medias.index:
            ax.scatter(i, medias[sent], color='red', marker='D', s=80, zorder=5,
                      edgecolor='white', linewidth=1)
    
    ax.set_xlabel('Sentimento predominante')
    ax.set_ylabel('Número de caracteres')
    ax.set_title('Figura 18 - Distribuição do tamanho das avaliações por sentimento')
    
    legend_elements = [Line2D([0], [0], marker='D', color='w', markerfacecolor='red',
                              markersize=10, label='Média')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_18_boxplot_tamanho_sentimento.png'),
                dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Retornar estatísticas para tabela
    return df_valid.groupby('sentimento_pred')['n_chars'].agg(['count', 'mean', 'median', 'std'])


def gerar_fig19_densidade_tamanho(df, output_dir):
    """Fig 19: Curvas de densidade do tamanho por sentimento."""
    print("  Gerando: fig_19_densidade_tamanho_sentimento.png")
    
    df_temp = df.copy()
    df_temp['n_chars'] = df_temp['review_text'].astype(str).apply(len)
    df_temp['sentimento_pred'] = df_temp['llm_analise_json'].apply(extrair_sentimento_predominante)
    
    df_valid = df_temp[df_temp['sentimento_pred'].notna()].copy()
    df_plot = df_valid[df_valid['n_chars'] <= df_valid['n_chars'].quantile(0.99)].copy()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cores = {'positivo': '#66c2a5', 'neutro': '#8da0cb', 'negativo': '#fc8d62'}
    
    for sent in ORDEM_SENTIMENTO:
        dados = df_plot[df_plot['sentimento_pred'] == sent]['n_chars']
        if len(dados) > 0:
            sns.kdeplot(dados, ax=ax, color=cores[sent], label=sent.capitalize(),
                       fill=True, alpha=0.3, linewidth=2)
    
    ax.set_xlabel('Número de caracteres')
    ax.set_ylabel('Densidade')
    ax.set_title('Figura 19 - Curvas de densidade do tamanho das avaliações')
    ax.legend(title='Sentimento')
    ax.set_xlim(0, None)
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_19_densidade_tamanho_sentimento.png'),
                dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()


# ============================================================
# FIGURAS 20-23: POSICIONAMENTO DIGITAL
# ============================================================

def gerar_fig20_rating_resposta(df, output_dir):
    """Fig 20: Distribuição de notas com/sem resposta do dono."""
    print("  Gerando: fig_20_rating_resposta_dono.png")
    
    df_temp = df.copy()
    df_temp['tem_resposta'] = df_temp['response_from_owner_text'].notna()
    
    stats_resp = df_temp.groupby('tem_resposta')['rating'].agg(['mean', 'count']).reset_index()
    stats_resp['label'] = stats_resp['tem_resposta'].map({True: 'Com Resposta', False: 'Sem Resposta'})
    stats_resp = stats_resp.sort_values('tem_resposta', ascending=False)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bars = ax.bar(stats_resp['label'], stats_resp['mean'], color='#3498DB', edgecolor='white')
    
    for i, (_, row) in enumerate(stats_resp.iterrows()):
        ax.text(i, row['mean'] + 0.05, f"{row['mean']:.2f}\n(n={int(row['count']):,})",
                ha='center', va='bottom', fontsize=11)
    
    ax.axhline(y=df_temp['rating'].mean(), color='gray', linestyle='--', alpha=0.7,
               label=f'Média geral ({df_temp["rating"].mean():.2f})')
    
    ax.set_ylabel('Rating Médio')
    ax.set_xlabel('')
    ax.set_ylim(0, 5.5)
    ax.set_title('Figura 20 - Distribuição de notas com e sem resposta do dono')
    ax.legend(loc='lower right')
    sns.despine()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_20_rating_resposta_dono.png'),
                dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()


def gerar_fig21_pct_resposta_rating(df, output_dir):
    """Fig 21: Percentual de resposta por nível de rating."""
    print("  Gerando: fig_21_pct_resposta_por_rating.png")
    
    df_temp = df.copy()
    df_temp['tem_resposta'] = df_temp['response_from_owner_text'].notna()
    
    pct_resp = df_temp.groupby('rating')['tem_resposta'].mean() * 100
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bars = ax.bar(pct_resp.index, pct_resp.values, color='#3498DB', edgecolor='white')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=11)
    
    ax.set_ylabel('% Reviews com Resposta do Dono')
    ax.set_xlabel('Rating')
    ax.set_title('Figura 21 - Percentual de resposta por nível de rating')
    sns.despine()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_21_pct_resposta_por_rating.png'),
                dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()


def gerar_fig22_rating_local_guide(df, output_dir):
    """Fig 22: Rating médio Local Guide vs Não Guide."""
    print("  Gerando: fig_22_rating_local_guide.png")
    
    df_temp = df.copy()
    df_temp['is_local_guide'] = df_temp['is_local_guide'].fillna(False).astype(bool)
    
    stats_guide = df_temp.groupby('is_local_guide')['rating'].agg(['mean', 'count']).reset_index()
    stats_guide['label'] = stats_guide['is_local_guide'].map({True: 'Local Guide', False: 'Não Guide'})
    stats_guide = stats_guide.sort_values('is_local_guide', ascending=False)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bars = ax.bar(stats_guide['label'], stats_guide['mean'], color='#3498DB', edgecolor='white')
    
    for i, (_, row) in enumerate(stats_guide.iterrows()):
        ax.text(i, row['mean'] + 0.05, f"{row['mean']:.2f}\n(n={int(row['count']):,})",
                ha='center', va='bottom', fontsize=11)
    
    ax.axhline(y=df_temp['rating'].mean(), color='gray', linestyle='--', alpha=0.7,
               label=f'Média geral ({df_temp["rating"].mean():.2f})')
    
    ax.set_ylabel('Rating Médio')
    ax.set_xlabel('')
    ax.set_ylim(0, 5.5)
    ax.set_title('Figura 22 - Rating médio entre Local Guide e Não Guide')
    ax.legend(loc='lower right')
    sns.despine()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_22_rating_local_guide.png'),
                dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()


def gerar_fig23_distribuicao_rating_guide(df, output_dir):
    """Fig 23: Distribuição de ratings por tipo de usuário."""
    print("  Gerando: fig_23_distribuicao_ratings_guide.png")
    
    df_temp = df.copy()
    df_temp['is_local_guide'] = df_temp['is_local_guide'].fillna(False).astype(bool)
    
    dist_data = []
    for guide in [True, False]:
        subset = df_temp[df_temp['is_local_guide'] == guide]
        total = len(subset)
        for rating in [1, 2, 3, 4, 5]:
            count = len(subset[subset['rating'] == rating])
            pct = count / total * 100 if total > 0 else 0
            dist_data.append({
                'rating': rating,
                'grupo': 'Local Guide' if guide else 'Não Guide',
                'pct': pct
            })
    
    dist_df = pd.DataFrame(dist_data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.barplot(data=dist_df, x='rating', y='pct', hue='grupo', ax=ax,
                palette=['#3498DB', '#95A5A6'])
    
    ax.set_ylabel('% das Avaliações')
    ax.set_xlabel('Rating')
    ax.set_title('Figura 23 - Distribuição dos ratings entre Local Guide e Não Guide')
    ax.legend(title='')
    sns.despine()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_23_distribuicao_ratings_guide.png'),
                dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()


# ============================================================
# FIGURAS 24-25: NOTA VS SENTIMENTO
# ============================================================

def gerar_fig24_sentimento_por_nota(df, output_dir):
    """Fig 24: Distribuição de sentimentos por nota atribuída."""
    print("  Gerando: fig_24_sentimento_por_nota.png")
    
    df_temp = df.copy()
    df_temp['sentimento_pred'] = df_temp['llm_analise_json'].apply(extrair_sentimento_predominante)
    df_valid = df_temp[(df_temp['sentimento_pred'].notna()) & (df_temp['rating'].notna())].copy()
    df_valid['rating'] = df_valid['rating'].astype(int)
    
    # Calcular proporções
    crosstab = pd.crosstab(df_valid['rating'], df_valid['sentimento_pred'], normalize='index') * 100
    cols_order = [s for s in ORDEM_SENTIMENTO if s in crosstab.columns]
    crosstab = crosstab[cols_order]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = crosstab.index.values
    bottom = np.zeros(len(x))
    cores = {'positivo': '#66c2a5', 'neutro': '#8da0cb', 'negativo': '#fc8d62'}
    
    for sent in cols_order:
        valores = crosstab[sent].values
        bars = ax.bar(x, valores, bottom=bottom, label=sent.capitalize(),
                     color=cores[sent], edgecolor='white', linewidth=0.5)
        
        for i, (val, b) in enumerate(zip(valores, bottom)):
            if val > 5:
                ax.text(x[i], b + val/2, f'{val:.0f}%',
                       ha='center', va='center', fontsize=9,
                       color='white', fontweight='bold')
        bottom += valores
    
    ax.set_xlabel('Nota (Rating)')
    ax.set_ylabel('Proporção (%)')
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_ylim(0, 100)
    ax.set_title('Figura 24 - Distribuição de sentimentos por nota atribuída')
    ax.legend(title='Sentimento', loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_24_sentimento_por_nota.png'),
                dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()


def gerar_fig25_score_por_nota(df, output_dir):
    """Fig 25: Score médio de sentimento por nota."""
    print("  Gerando: fig_25_score_por_nota.png")
    
    df_temp = df.copy()
    df_temp['score_sentimento'] = df_temp['llm_analise_json'].apply(calcular_score_sentimento)
    df_valid = df_temp[(df_temp['score_sentimento'].notna()) & (df_temp['rating'].notna())].copy()
    df_valid['rating'] = df_valid['rating'].astype(int)
    
    score_por_nota = df_valid.groupby('rating')['score_sentimento'].agg(['mean', 'std']).reset_index()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cores = ['#fc8d62' if m < 0 else '#66c2a5' for m in score_por_nota['mean']]
    
    bars = ax.bar(score_por_nota['rating'], score_por_nota['mean'],
                  color=cores, edgecolor='black', linewidth=0.5, alpha=0.8)
    
    ax.errorbar(score_por_nota['rating'], score_por_nota['mean'],
               yerr=score_por_nota['std'], fmt='none', color='black', capsize=4, capthick=1)
    
    for _, row in score_por_nota.iterrows():
        y_pos = row['mean'] + 0.08 if row['mean'] >= 0 else row['mean'] - 0.08
        va = 'bottom' if row['mean'] >= 0 else 'top'
        ax.text(row['rating'], y_pos, f"{row['mean']:.2f}",
               ha='center', va=va, fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Nota (Rating)')
    ax.set_ylabel('Score médio de sentimento')
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_ylim(-1.1, 1.1)
    ax.set_title('Figura 25 - Score médio de sentimento por nota')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.7, linewidth=1)
    
    ax.text(5.3, 0.9, 'Positivo', fontsize=9, color='#66c2a5', fontweight='bold')
    ax.text(5.3, -0.9, 'Negativo', fontsize=9, color='#fc8d62', fontweight='bold')
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_25_score_por_nota.png'),
                dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()


# ============================================================
# FIGURAS 26-30: ANÁLISE DE PROBLEMAS
# ============================================================

def expandir_problemas(df):
    """Expande coluna problemas_subcategorias em linhas individuais."""
    problemas_list = []
    
    for idx, row in df.iterrows():
        if pd.notna(row.get('problemas_subcategorias')):
            subcats = str(row['problemas_subcategorias']).split('|')
            for subcat in subcats:
                subcat = subcat.strip()
                if subcat:
                    problemas_list.append({
                        'review_id': row.get('review_id'),
                        'rating': row.get('rating'),
                        'subcategoria': subcat,
                        'score': row.get('problemas_score_medio')
                    })
    
    return pd.DataFrame(problemas_list)


def gerar_fig26_frequencia_problemas(df_problemas, output_dir):
    """Fig 26: Frequência de problemas por subcategoria."""
    print("  Gerando: fig_26_frequencia_problemas.png")
    
    freq = df_problemas['subcategoria'].value_counts()
    freq_pct = (freq / freq.sum() * 100).round(1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df_freq = pd.DataFrame({
        'subcategoria': freq.index,
        'contagem': freq.values,
        'percentual': freq_pct.values
    })
    
    cores = [CORES_PROBLEMAS.get(cat, '#95a5a6') for cat in df_freq['subcategoria']]
    bars = ax.barh(df_freq['subcategoria'], df_freq['contagem'], color=cores, edgecolor='white')
    ax.invert_yaxis()
    
    for bar, pct, count in zip(bars, df_freq['percentual'], df_freq['contagem']):
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                f'{formatar_numero(count)} ({pct:.1f}%)', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_title('Figura 26 - Frequência de problemas por subcategoria')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_26_frequencia_problemas.png'),
                dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()


def gerar_fig27_score_problemas(df_problemas, output_dir):
    """Fig 27: Score médio de sentimento por subcategoria de problema."""
    print("  Gerando: fig_27_score_problemas.png")
    
    score_por_subcat = df_problemas.groupby('subcategoria')['score'].mean().sort_values()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cores = [CORES_PROBLEMAS.get(cat, '#95a5a6') for cat in score_por_subcat.index]
    bars = ax.barh(score_por_subcat.index, score_por_subcat.values, color=cores, edgecolor='white')
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width - 0.05, bar.get_y() + bar.get_height()/2,
                f'{width:.2f}', va='center', ha='right', fontsize=10, 
                fontweight='bold', color='white')
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlim(-1.1, 0.1)
    ax.set_xlabel('Score médio de sentimento')
    ax.set_ylabel('')
    ax.set_title('Figura 27 - Score médio de sentimento por subcategoria de problema')
    sns.despine()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_27_score_problemas.png'),
                dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()


def expandir_problemas_com_motivadores(df):
    """Expande coluna problemas com motivadores (se existir)."""
    problemas_list = []
    
    # Verificar se existem colunas de motivadores processados
    col_motiv = None
    for col in ['problemas_motivadores', 'motivador', 'motivadores']:
        if col in df.columns:
            col_motiv = col
            break
    
    for idx, row in df.iterrows():
        if pd.notna(row.get('problemas_subcategorias')):
            subcats = str(row['problemas_subcategorias']).split('|')
            
            # Se tiver motivadores
            motivadores = []
            if col_motiv and pd.notna(row.get(col_motiv)):
                motivadores = str(row[col_motiv]).split('|')
            
            for i, subcat in enumerate(subcats):
                subcat = subcat.strip()
                if subcat:
                    motiv = motivadores[i].strip() if i < len(motivadores) else 'NAO_ESPECIFICADO'
                    problemas_list.append({
                        'review_id': row.get('review_id'),
                        'rating': row.get('rating'),
                        'subcategoria': subcat,
                        'motivador': motiv,
                        'score': row.get('problemas_score_medio')
                    })
    
    return pd.DataFrame(problemas_list)


def gerar_fig28_motivadores_por_subcategoria(df, output_dir):
    """Fig 28: Distribuição dos motivadores por subcategoria."""
    print("  Gerando: fig_28_motivadores_por_subcategoria.png")
    
    df_prob = expandir_problemas_com_motivadores(df)
    
    if len(df_prob) == 0 or 'motivador' not in df_prob.columns:
        print("    AVISO: Sem dados de motivadores. Pulando figura 28.")
        return
    
    # Top 4 subcategorias
    top_subcats = df_prob['subcategoria'].value_counts().head(4).index.tolist()
    df_top = df_prob[df_prob['subcategoria'].isin(top_subcats)]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, subcat in enumerate(top_subcats):
        ax = axes[i]
        df_sub = df_top[df_top['subcategoria'] == subcat]
        
        # Top 5 motivadores
        top_motiv = df_sub['motivador'].value_counts().head(5)
        
        cores = [CORES_PROBLEMAS.get(subcat, '#95a5a6')] * len(top_motiv)
        bars = ax.barh(top_motiv.index, top_motiv.values, color=cores, alpha=0.8, edgecolor='white')
        ax.invert_yaxis()
        
        for bar in bars:
            width = bar.get_width()
            pct = width / len(df_sub) * 100
            ax.text(width + 5, bar.get_y() + bar.get_height()/2,
                    f'{formatar_numero(width)} ({pct:.1f}%)', va='center', fontsize=9)
        
        ax.set_title(f'{subcat}', fontweight='bold')
        ax.set_xlabel('')
        ax.set_xticks([])
        sns.despine(ax=ax, left=True, bottom=True)
    
    plt.suptitle('Figura 28 - Distribuição dos motivadores por subcategoria', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_28_motivadores_por_subcategoria.png'),
                dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()


def gerar_fig29_heatmap_subcat_motivador(df, output_dir):
    """Fig 29: Frequência de menções por subcategoria e motivador."""
    print("  Gerando: fig_29_heatmap_subcat_motivador.png")
    
    df_prob = expandir_problemas_com_motivadores(df)
    
    if len(df_prob) == 0 or 'motivador' not in df_prob.columns:
        print("    AVISO: Sem dados de motivadores. Pulando figura 29.")
        return
    
    # Criar tabela cruzada
    crosstab = pd.crosstab(df_prob['subcategoria'], df_prob['motivador'])
    
    # Filtrar top 8 subcategorias e top 10 motivadores
    top_subcats = df_prob['subcategoria'].value_counts().head(8).index.tolist()
    top_motiv = df_prob['motivador'].value_counts().head(10).index.tolist()
    
    crosstab_filtrado = crosstab.loc[
        [s for s in top_subcats if s in crosstab.index],
        [m for m in top_motiv if m in crosstab.columns]
    ]
    
    if crosstab_filtrado.empty:
        print("    AVISO: Dados insuficientes para heatmap. Pulando figura 29.")
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sns.heatmap(crosstab_filtrado, annot=True, fmt='d', cmap='YlOrRd',
                ax=ax, linewidths=0.5, linecolor='white',
                annot_kws={'fontsize': 9})
    
    ax.set_xlabel('Motivador')
    ax.set_ylabel('Subcategoria')
    ax.set_title('Figura 29 - Frequência de menções por subcategoria e motivador')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_29_heatmap_subcat_motivador.png'),
                dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()


def gerar_fig30_matriz_priorizacao(df_problemas, output_dir):
    """Fig 30: Mapa de priorização de ações corretivas."""
    print("  Gerando: fig_30_matriz_priorizacao.png")
    
    # Calcular frequência e gravidade por subcategoria
    stats = df_problemas.groupby('subcategoria').agg({
        'score': ['mean', 'count']
    }).reset_index()
    stats.columns = ['subcategoria', 'score_medio', 'frequencia']
    stats['gravidade'] = -stats['score_medio']  # Inverter para que mais negativo = mais grave
    stats['freq_pct'] = stats['frequencia'] / stats['frequencia'].sum() * 100
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cores = [CORES_PROBLEMAS.get(cat, '#95a5a6') for cat in stats['subcategoria']]
    
    scatter = ax.scatter(stats['freq_pct'], stats['gravidade'], 
                        s=stats['frequencia']/10, c=cores, alpha=0.7, edgecolor='black')
    
    for _, row in stats.iterrows():
        ax.annotate(row['subcategoria'], (row['freq_pct'], row['gravidade']),
                   xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
    
    # Linhas de quadrantes
    ax.axhline(y=stats['gravidade'].median(), color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=stats['freq_pct'].median(), color='gray', linestyle='--', alpha=0.5)
    
    # Labels dos quadrantes
    ax.text(0.95, 0.95, 'CRÍTICO\n(Alta frequência + Alta gravidade)', 
            transform=ax.transAxes, ha='right', va='top', fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.5))
    ax.text(0.05, 0.95, 'MONITORAR\n(Baixa frequência + Alta gravidade)', 
            transform=ax.transAxes, ha='left', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='#ffffcc', alpha=0.5))
    
    ax.set_xlabel('Frequência (%)')
    ax.set_ylabel('Gravidade (score negativo)')
    ax.set_title('Figura 30 - Mapa de priorização de ações corretivas')
    sns.despine()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_30_matriz_priorizacao.png'),
                dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()


# ============================================================
# FIGURAS 11-12: VISUALIZAÇÃO UMAP
# ============================================================

def gerar_embeddings_umap(df, output_dir, sample_size=50000):
    """Gera embeddings e projeção UMAP para visualizações."""
    
    if not UMAP_AVAILABLE:
        print("  AVISO: sentence-transformers/umap não disponíveis. Pulando figuras 11-12.")
        return None, None, None
    
    print("  Preparando embeddings para UMAP...")
    
    # Filtrar reviews com texto válido
    df_valid = df[df['review_text'].notna() & (df['review_text'].str.len() > 10)].copy()
    
    # Amostrar se necessário
    if len(df_valid) > sample_size:
        df_sample = df_valid.sample(n=sample_size, random_state=42).reset_index(drop=True)
    else:
        df_sample = df_valid.reset_index(drop=True)
    
    print(f"    Amostra: {len(df_sample):,} reviews")
    
    # Cache de embeddings
    cache_emb = os.path.join(output_dir, 'embeddings_cache.npy')
    cache_umap = os.path.join(output_dir, 'umap_cache.npy')
    
    textos = df_sample['review_text'].fillna('').tolist()
    
    # Verificar cache de embeddings
    if os.path.exists(cache_emb):
        embeddings = np.load(cache_emb)
        if len(embeddings) == len(textos):
            print(f"    Embeddings carregados do cache")
        else:
            embeddings = None
    else:
        embeddings = None
    
    # Gerar embeddings se necessário
    if embeddings is None:
        print("    Gerando embeddings (pode demorar)...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        batch_size = 1000
        embeddings_list = []
        
        for i in range(0, len(textos), batch_size):
            batch = textos[i:i+batch_size]
            batch_emb = model.encode(batch, show_progress_bar=False, batch_size=32)
            embeddings_list.append(batch_emb)
            if (i // batch_size + 1) % 10 == 0:
                print(f"      Processados: {i+len(batch):,}/{len(textos):,}")
        
        embeddings = np.vstack(embeddings_list)
        np.save(cache_emb, embeddings)
        
        del model
        gc.collect()
    
    # Verificar cache de UMAP
    if os.path.exists(cache_umap):
        X_2d = np.load(cache_umap)
        if len(X_2d) == len(textos):
            print(f"    Projeção UMAP carregada do cache")
        else:
            X_2d = None
    else:
        X_2d = None
    
    # Projeção UMAP se necessário
    if X_2d is None:
        print("    Aplicando UMAP (pode demorar)...")
        umap_model = UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine',
            random_state=42,
            verbose=False
        )
        X_2d = umap_model.fit_transform(embeddings)
        np.save(cache_umap, X_2d)
    
    return df_sample, embeddings, X_2d


def gerar_fig11_umap_topicos(df_sample, X_2d, output_dir):
    """Fig 11: Distribuição clusters por tópicos (UMAP)."""
    print("  Gerando: fig_11_umap_topicos.png")
    
    if X_2d is None or df_sample is None:
        print("    AVISO: Dados UMAP não disponíveis. Pulando figura 11.")
        return
    
    # Verificar se tem coluna de tópicos
    col_topico = None
    for col in ['topic', 'topic_merged', 'topic_final', 'topico']:
        if col in df_sample.columns:
            col_topico = col
            break
    
    if col_topico is None:
        print("    AVISO: Coluna de tópicos não encontrada. Pulando figura 11.")
        return
    
    # Top 15 tópicos mais frequentes
    top_topicos = df_sample[col_topico].value_counts().head(15).index.tolist()
    
    # Paleta de cores
    cores_topicos = plt.cm.tab20(np.linspace(0, 1, 15))
    
    # Zoom baseado em percentis
    x_min, x_max = np.percentile(X_2d[:, 0], [1, 99])
    y_min, y_max = np.percentile(X_2d[:, 1], [1, 99])
    margem_x = (x_max - x_min) * 0.05
    margem_y = (y_max - y_min) * 0.05
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Plotar "outros" primeiro (cinza)
    mask_outros = ~df_sample[col_topico].isin(top_topicos)
    if mask_outros.sum() > 0:
        ax.scatter(X_2d[mask_outros, 0], X_2d[mask_outros, 1],
                  c='#bdc3c7', s=5, alpha=0.2, label=f'Outros ({mask_outros.sum():,})')
    
    # Plotar top 15 tópicos
    patches = []
    for i, topico in enumerate(top_topicos):
        mask = df_sample[col_topico] == topico
        count = mask.sum()
        if count > 0:
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                      c=[cores_topicos[i]], s=8, alpha=0.6)
            patches.append(mpatches.Patch(color=cores_topicos[i], 
                          label=f'Tópico {topico} ({count:,})'))
    
    ax.set_xlim(x_min - margem_x, x_max + margem_x)
    ax.set_ylim(y_min - margem_y, y_max + margem_y)
    ax.axis('off')
    ax.set_title('Figura 11 - Distribuição de clusters por tópicos (UMAP)', fontsize=14, fontweight='bold')
    
    # Legenda em duas colunas
    ax.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.02),
              fontsize=8, ncol=5, columnspacing=0.8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_11_umap_topicos.png'),
                dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()


def gerar_fig12_umap_categorias(df_sample, X_2d, output_dir):
    """Fig 12: Distribuição clusters por categoria gerencial (UMAP)."""
    print("  Gerando: fig_12_umap_categorias.png")
    
    if X_2d is None or df_sample is None:
        print("    AVISO: Dados UMAP não disponíveis. Pulando figura 12.")
        return
    
    # Extrair categoria do LLM se não tiver coluna
    if 'categoria' not in df_sample.columns:
        df_sample['categoria'] = df_sample['llm_analise_json'].apply(extrair_primeira_categoria_llm)
    
    # Zoom baseado em percentis
    x_min, x_max = np.percentile(X_2d[:, 0], [1, 99])
    y_min, y_max = np.percentile(X_2d[:, 1], [1, 99])
    margem_x = (x_max - x_min) * 0.05
    margem_y = (y_max - y_min) * 0.05
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Ordem de plotagem: sem_categoria/comida primeiro (ficam no fundo)
    ordem_plot = ['sem_categoria', 'comida', 'atendimento', 'ambiente', 'preco', 'problemas']
    
    patches = []
    for categoria in ordem_plot:
        if categoria == 'sem_categoria':
            cor = '#bdc3c7'
            alpha = 0.2
        elif categoria == 'comida':
            cor = '#d5d5d5'  # Cinza claro para comida (maior volume)
            alpha = 0.3
        else:
            cor = CORES_CATEGORIAS.get(categoria, '#bdc3c7')
            alpha = 0.7
        
        mask = df_sample['categoria'] == categoria
        count = mask.sum()
        
        if count > 0:
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                      c=cor, s=8, alpha=alpha)
            patches.append(mpatches.Patch(color=cor, label=f'{categoria.capitalize()} ({count:,})'))
    
    ax.set_xlim(x_min - margem_x, x_max + margem_x)
    ax.set_ylim(y_min - margem_y, y_max + margem_y)
    ax.axis('off')
    ax.set_title('Figura 12 - Distribuição de clusters por categoria gerencial (UMAP)', 
                 fontsize=14, fontweight='bold')
    
    ax.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.02),
              fontsize=10, ncol=len(patches), columnspacing=1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_12_umap_categorias.png'),
                dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()


# ============================================================
# FIGURAS 13-17: CATEGORIZAÇÃO E DISTRIBUIÇÃO
# ============================================================

# ============================================================
# FIGURA 13: MATRIZ DE SIMILARIDADE SEMÂNTICA
# ============================================================

def gerar_fig13_matriz_similaridade(df, output_dir):
    """Fig 13: Matriz de similaridade semântica entre categorias."""
    print("  Gerando: fig_13_matriz_similaridade.png")
    
    # Extrair todas as análises por categoria
    embeddings_por_categoria = {cat: [] for cat in ORDEM_CATEGORIAS}
    
    for json_str in df['llm_analise_json'].dropna():
        try:
            parsed = json.loads(json_str)
            for a in parsed.get('analises', []):
                cat = a.get('categoria', '').lower().strip()
                evidencia = a.get('evidencia', '')
                if cat in ORDEM_CATEGORIAS and evidencia:
                    embeddings_por_categoria[cat].append(evidencia)
        except:
            continue
    
    # Calcular similaridade baseada em palavras-chave compartilhadas
    # Criar documento representativo por categoria (concatenar evidências)
    docs = []
    cats_com_dados = []
    for cat in ORDEM_CATEGORIAS:
        if embeddings_por_categoria[cat]:
            # Pegar amostra de até 1000 evidências
            sample = embeddings_por_categoria[cat][:1000]
            docs.append(' '.join(sample))
            cats_com_dados.append(cat)
    
    if len(cats_com_dados) < 2:
        print("    AVISO: Dados insuficientes para matriz de similaridade.")
        return
    
    # TF-IDF e similaridade
    vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
    tfidf_matrix = vectorizer.fit_transform(docs)
    sim_matrix = cosine_similarity(tfidf_matrix)
    
    # Criar DataFrame
    df_sim = pd.DataFrame(sim_matrix, 
                         index=[c.capitalize() for c in cats_com_dados],
                         columns=[c.capitalize() for c in cats_com_dados])
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    sns.heatmap(df_sim, annot=True, fmt='.2f', cmap='YlOrRd',
                ax=ax, linewidths=1, linecolor='white',
                annot_kws={'fontsize': 11}, vmin=0, vmax=1,
                cbar_kws={'label': 'Similaridade (cosseno)'})
    
    ax.set_title('Figura 13 - Matriz de similaridade semântica entre categorias')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_13_matriz_similaridade.png'),
                dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()


# ============================================================
# FIGURA 14: MATRIZ DE CONCORDÂNCIA
# ============================================================

def gerar_fig14_matriz_concordancia(df, output_dir):
    """Fig 14: Matriz de concordância BERTopic vs LLM."""
    print("  Gerando: fig_14_matriz_concordancia.png")
    
    df_temp = df.copy()
    
    # Extrair categoria do LLM
    df_temp['categoria_llm'] = df_temp['llm_analise_json'].apply(extrair_primeira_categoria_llm)
    
    # Verificar se tem coluna 'categoria' (BERTopic)
    if 'categoria' not in df_temp.columns:
        print("    AVISO: Coluna 'categoria' (BERTopic) não encontrada. Pulando figura 14.")
        return
    
    # Filtrar válidos
    df_comp = df_temp[
        df_temp['categoria'].isin(ORDEM_CATEGORIAS) &
        df_temp['categoria_llm'].isin(ORDEM_CATEGORIAS)
    ].copy()
    
    if len(df_comp) == 0:
        print("    AVISO: Sem dados para matriz de concordância. Pulando figura 14.")
        return
    
    # Calcular concordância
    concordancia = (df_comp['categoria'] == df_comp['categoria_llm']).mean() * 100
    
    # Matriz de confusão
    cm = confusion_matrix(df_comp['categoria'], df_comp['categoria_llm'], labels=ORDEM_CATEGORIAS)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[c.capitalize() for c in ORDEM_CATEGORIAS],
                yticklabels=[c.capitalize() for c in ORDEM_CATEGORIAS],
                ax=ax, linewidths=1, linecolor='white', annot_kws={'fontsize': 11})
    
    ax.set_xlabel('Categoria LLM')
    ax.set_ylabel('Categoria BERTopic')
    ax.set_title(f'Figura 14 - Matriz de Concordância entre Métodos\n(Concordância geral: {concordancia:.1f}%)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_14_matriz_concordancia.png'),
                dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def main():
    """Executa a geração de todas as figuras."""
    
    print("="*70)
    print("GERAÇÃO DE FIGURAS DA DISSERTAÇÃO")
    print("="*70)
    
    # Criar diretório de saída
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Carregar dados
    print(f"\n📂 Carregando {CONFIG['input_file']}...")
    
    if not os.path.exists(CONFIG['input_file']):
        print(f"❌ Arquivo não encontrado: {CONFIG['input_file']}")
        print("   Verifique se o arquivo existe ou ajuste o caminho.")
        return
    
    df = pd.read_excel(CONFIG['input_file'])
    print(f"   → {len(df):,} avaliações carregadas")
    
    # =========================================================
    # FIGURAS 11-12: VISUALIZAÇÃO UMAP
    # =========================================================
    print("\n📊 Gerando figuras UMAP (11-12)...")
    
    if UMAP_AVAILABLE:
        df_sample_umap, embeddings, X_2d = gerar_embeddings_umap(df, CONFIG['output_dir'])
        
        if X_2d is not None:
            gerar_fig11_umap_topicos(df_sample_umap, X_2d, CONFIG['output_dir'])
            gerar_fig12_umap_categorias(df_sample_umap, X_2d, CONFIG['output_dir'])
    else:
        print("  AVISO: Bibliotecas UMAP não disponíveis. Pulando figuras 11-12.")
    
    # =========================================================
    # FIGURAS 13-17: CATEGORIZAÇÃO
    # =========================================================
    print("\n📊 Gerando figuras de categorização (13-17)...")
    
    gerar_fig13_matriz_similaridade(df, CONFIG['output_dir'])
    gerar_fig14_matriz_concordancia(df, CONFIG['output_dir'])
    gerar_fig15_distribuicao_categorias(df, CONFIG['output_dir'])
    gerar_fig16_frequencia_categorias(df, CONFIG['output_dir'])
    gerar_fig17_polaridade_sentimentos(df, CONFIG['output_dir'])
    
    # =========================================================
    # FIGURAS 18-19: TAMANHO VS SENTIMENTO
    # =========================================================
    print("\n📊 Gerando figuras de tamanho vs sentimento (18-19)...")
    
    stats_tamanho = gerar_fig18_boxplot_tamanho(df, CONFIG['output_dir'])
    gerar_fig19_densidade_tamanho(df, CONFIG['output_dir'])
    
    # Salvar estatísticas
    if stats_tamanho is not None:
        stats_tamanho.to_csv(os.path.join(CONFIG['output_dir'], 'tabela_04_estatisticas_tamanho.csv'))
        print("  Salvo: tabela_04_estatisticas_tamanho.csv")
    
    # =========================================================
    # FIGURAS 20-23: POSICIONAMENTO DIGITAL
    # =========================================================
    print("\n📊 Gerando figuras de posicionamento digital (20-23)...")
    
    gerar_fig20_rating_resposta(df, CONFIG['output_dir'])
    gerar_fig21_pct_resposta_rating(df, CONFIG['output_dir'])
    gerar_fig22_rating_local_guide(df, CONFIG['output_dir'])
    gerar_fig23_distribuicao_rating_guide(df, CONFIG['output_dir'])
    
    # =========================================================
    # FIGURAS 24-25: NOTA VS SENTIMENTO
    # =========================================================
    print("\n📊 Gerando figuras de nota vs sentimento (24-25)...")
    
    gerar_fig24_sentimento_por_nota(df, CONFIG['output_dir'])
    gerar_fig25_score_por_nota(df, CONFIG['output_dir'])
    
    # =========================================================
    # FIGURAS 26-30: ANÁLISE DE PROBLEMAS
    # =========================================================
    print("\n📊 Gerando figuras de análise de problemas (26-30)...")
    
    # Verificar se tem coluna de problemas
    if 'problemas_subcategorias' in df.columns:
        df_problemas = expandir_problemas(df)
        
        if len(df_problemas) > 0:
            gerar_fig26_frequencia_problemas(df_problemas, CONFIG['output_dir'])
            gerar_fig27_score_problemas(df_problemas, CONFIG['output_dir'])
            gerar_fig28_motivadores_por_subcategoria(df, CONFIG['output_dir'])
            gerar_fig29_heatmap_subcat_motivador(df, CONFIG['output_dir'])
            gerar_fig30_matriz_priorizacao(df_problemas, CONFIG['output_dir'])
        else:
            print("  AVISO: Sem dados de problemas para gerar figuras 26-30.")
    else:
        print("  AVISO: Coluna 'problemas_subcategorias' não encontrada.")
    
    # =========================================================
    # RESUMO
    # =========================================================
    print("\n" + "="*70)
    print("✅ GERAÇÃO CONCLUÍDA!")
    print("="*70)
    
    # Listar arquivos gerados
    arquivos = sorted([f for f in os.listdir(CONFIG['output_dir']) if f.startswith('fig_')])
    print(f"\nFiguras geradas ({len(arquivos)}):")
    for arq in arquivos:
        print(f"  - {arq}")
    
    print(f"\nDiretório de saída: {CONFIG['output_dir']}/")
    print("="*70)


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    main()
