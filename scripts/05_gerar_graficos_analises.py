#!/usr/bin/env python3
"""
================================================================================
GERAÇÃO DE GRÁFICOS - ANÁLISES DE PROBLEMAS E PREÇO
================================================================================

Este script gera os gráficos da análise de problemas e preço:

GRÁFICOS DE PROBLEMAS:
  1. fig_problemas_frequencia.png    - Frequência por subcategoria
  2. fig_problemas_gravidade.png     - Score médio por subcategoria
  3. fig_problemas_matriz.png        - Matriz de priorização
  4. fig_coocorrencia_problemas.png  - Heatmap de co-ocorrência

GRÁFICOS DE PREÇO:
  5. fig_preco_distribuicao_sentimento.png  - Distribuição positivo/negativo/neutro
  6. fig_preco_score_produto.png            - Score por produto (top 30)
  7. fig_preco_score_motivador.png          - Score por motivador
  8. fig_correlacao_preco_rating.png        - Correlação score × rating
  9. fig_efeito_resposta_dono.png           - Impacto da resposta do dono

ENTRADA:
  - dataset_analises_completas.xlsx (saída do script 04)

SAÍDA:
  - 9 arquivos PNG na pasta outputs/

EXECUÇÃO:
  python 05_gerar_graficos_analises.py

================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy import stats
import os
import re
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURAÇÕES
# ============================================================

CONFIG = {
    "input_file": "dataset_analises_completas.xlsx",
    "output_dir": "outputs",
    "dpi": 300,
}

# Paleta de cores
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

CORES_SENTIMENTO = {
    'positivo': '#27ae60',
    'neutro': '#f39c12',
    'negativo': '#e74c3c'
}

# Configuração visual
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
sns.set_style("whitegrid")
sns.set_palette("husl")

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def sanitizar_label(texto):
    """Remove caracteres que matplotlib interpreta como LaTeX."""
    if pd.isna(texto):
        return texto
    texto = str(texto)
    texto = texto.replace('$', '')
    texto = re.sub(r'[_^{}\\]', '', texto)
    return texto.strip() if texto.strip() else 'indefinido'


def formatar_numero(n):
    """Formata número com separador de milhar."""
    return f"{n:,.0f}".replace(",", ".")


def expandir_problemas(df):
    """Expande a coluna problemas_subcategorias em linhas individuais."""
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


def expandir_precos(df):
    """Expande as colunas de preço em linhas individuais."""
    precos_list = []
    
    for idx, row in df.iterrows():
        if pd.notna(row.get('preco_produtos')) and pd.notna(row.get('preco_motivadores')):
            produtos = str(row['preco_produtos']).split('|')
            motivadores = str(row['preco_motivadores']).split('|')
            
            for prod, mot in zip(produtos, motivadores):
                prod = sanitizar_label(prod)
                mot = mot.strip()
                if prod and mot:
                    precos_list.append({
                        'review_id': row.get('review_id'),
                        'rating': row.get('rating'),
                        'produto': prod,
                        'motivador': mot,
                        'score': row.get('preco_score_medio')
                    })
    
    return pd.DataFrame(precos_list)


def analisar_coocorrencia(df):
    """Analisa quais problemas aparecem juntos."""
    coocorrencias = Counter()
    
    for subcats in df['problemas_subcategorias'].dropna():
        categorias = [c.strip() for c in str(subcats).split('|') if c.strip()]
        if len(categorias) > 1:
            for i in range(len(categorias)):
                for j in range(i+1, len(categorias)):
                    par = tuple(sorted([categorias[i], categorias[j]]))
                    coocorrencias[par] += 1
    
    return coocorrencias


# ============================================================
# FUNÇÕES DE GRÁFICOS
# ============================================================

def plot_problemas_frequencia(df_problemas, output_path):
    """Gráfico 1: Frequência de Problemas por Subcategoria."""
    print("  Gerando: fig_problemas_frequencia.png")
    
    freq_problemas = df_problemas['subcategoria'].value_counts()
    freq_problemas_pct = (freq_problemas / freq_problemas.sum() * 100).round(1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df_freq = pd.DataFrame({
        'subcategoria': freq_problemas.index,
        'contagem': freq_problemas.values,
        'percentual': freq_problemas_pct.values
    })
    
    cores = [CORES_PROBLEMAS.get(cat, '#95a5a6') for cat in df_freq['subcategoria']]
    bars = ax.barh(df_freq['subcategoria'], df_freq['contagem'], color=cores, edgecolor='white')
    ax.invert_yaxis()
    
    for bar, pct, count in zip(bars, df_freq['percentual'], df_freq['contagem']):
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                f'{formatar_numero(count)} ({pct:.1f}%)', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_title('Frequência de Problemas por Subcategoria', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()


def plot_problemas_gravidade(df_problemas, output_path):
    """Gráfico 2: Gravidade de Problemas (Score de Sentimento)."""
    print("  Gerando: fig_problemas_gravidade.png")
    
    score_por_subcat = df_problemas.groupby('subcategoria')['score'].mean().sort_values()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df_grav = pd.DataFrame({
        'subcategoria': score_por_subcat.index,
        'score': score_por_subcat.values
    }).sort_values('score')
    
    cores = [CORES_PROBLEMAS.get(cat, '#95a5a6') for cat in df_grav['subcategoria']]
    bars = ax.barh(df_grav['subcategoria'], df_grav['score'], color=cores, edgecolor='white')
    
    for bar in bars:
        x_pos = bar.get_width() - 0.05 if bar.get_width() < 0 else bar.get_width() + 0.05
        ha = 'right' if bar.get_width() < 0 else 'left'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                f'{bar.get_width():.2f}', va='center', ha=ha, fontsize=11, fontweight='bold')
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    
    media_score = df_problemas['score'].mean()
    ax.axvline(x=media_score, color='#e74c3c', linestyle='--', linewidth=2,
               label=f'Média geral ({media_score:.2f})')
    ax.legend(loc='lower right')
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_xlim(-1.2, 0.5)
    ax.set_title('Gravidade: Score Médio de Sentimento por Subcategoria\n(mais negativo = mais grave)',
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()


def plot_problemas_matriz(df_problemas, output_path):
    """Gráfico 3: Matriz de Priorização."""
    print("  Gerando: fig_problemas_matriz.png")
    
    freq_problemas = df_problemas['subcategoria'].value_counts()
    freq_problemas_pct = (freq_problemas / freq_problemas.sum() * 100).round(1)
    
    matriz = pd.DataFrame({
        'frequencia': freq_problemas,
        'frequencia_pct': freq_problemas_pct,
        'score_medio': df_problemas.groupby('subcategoria')['score'].mean(),
        'gravidade': -df_problemas.groupby('subcategoria')['score'].mean()
    })
    matriz = matriz.sort_values('frequencia', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    x = matriz['frequencia_pct']
    y = matriz['gravidade']
    labels = matriz.index
    
    cores_scatter = [CORES_PROBLEMAS.get(cat, '#95a5a6') for cat in labels]
    scatter = ax.scatter(x, y, s=300, c=cores_scatter, alpha=0.8, edgecolors='black', linewidths=2)
    
    for i, label in enumerate(labels):
        ax.annotate(label, (x.iloc[i], y.iloc[i]), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=10, fontweight='bold')
    
    ax.axhline(y=y.median(), color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=x.median(), color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    ax.text(x.max()*0.85, y.max()*0.95, 'CRÍTICO', ha='center', fontsize=11, color='#e74c3c', fontweight='bold')
    ax.text(x.min()*2, y.max()*0.95, 'IMPORTANTE', ha='center', fontsize=11, color='#e67e22', fontweight='bold')
    ax.text(x.max()*0.85, y.min()*1.1, 'MONITORAR', ha='center', fontsize=11, color='#3498db', fontweight='bold')
    ax.text(x.min()*2, y.min()*1.1, 'BAIXA\nPRIORIDADE', ha='center', fontsize=11, color='gray', fontweight='bold')
    
    ax.set_xlabel('Frequência (%)', fontsize=12)
    ax.set_ylabel('Gravidade (Score invertido: -score)', fontsize=12)
    ax.set_title('Matriz de Priorização de Problemas', fontsize=14, fontweight='bold', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()


def plot_coocorrencia_problemas(df, output_path):
    """Gráfico 4: Heatmap de Co-ocorrência de Problemas."""
    print("  Gerando: fig_coocorrencia_problemas.png")
    
    coocorrencias = analisar_coocorrencia(df)
    cooc_filtrado = {k: v for k, v in coocorrencias.items() if k[0] != k[1]}
    
    if not cooc_filtrado:
        print("    ⚠️ Não há co-ocorrências para plotar")
        return
    
    todas_subcats = list(set([cat for par in cooc_filtrado.keys() for cat in par]))
    matriz_cooc = pd.DataFrame(0, index=todas_subcats, columns=todas_subcats)
    
    for (cat1, cat2), count in cooc_filtrado.items():
        matriz_cooc.loc[cat1, cat2] = count
        matriz_cooc.loc[cat2, cat1] = count
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matriz_cooc, annot=True, fmt='d', cmap='YlOrRd',
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Frequência'},
                annot_kws={'fontsize': 11, 'fontweight': 'bold'})
    ax.set_title('Co-ocorrência de Problemas\n(excluindo duplicados)', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()


def plot_preco_distribuicao_sentimento(df_precos, output_path):
    """Gráfico 5: Distribuição de Sentimento nas Menções de Preço."""
    print("  Gerando: fig_preco_distribuicao_sentimento.png")
    
    def classificar_sentimento(score):
        if pd.isna(score):
            return None
        if score > 0:
            return 'positivo'
        elif score < 0:
            return 'negativo'
        else:
            return 'neutro'
    
    df_precos = df_precos.copy()
    df_precos['sentimento'] = df_precos['score'].apply(classificar_sentimento)
    dist_sentimento = df_precos['sentimento'].value_counts()
    dist_sentimento_pct = (dist_sentimento / dist_sentimento.sum() * 100).round(1)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    df_sent = pd.DataFrame({
        'sentimento': dist_sentimento.index,
        'contagem': dist_sentimento.values,
        'percentual': dist_sentimento_pct.values
    })
    ordem = ['negativo', 'neutro', 'positivo']
    df_sent['sentimento'] = pd.Categorical(df_sent['sentimento'], categories=ordem, ordered=True)
    df_sent = df_sent.sort_values('sentimento')
    
    cores = [CORES_SENTIMENTO.get(s, '#95a5a6') for s in df_sent['sentimento']]
    bars = ax.bar(df_sent['sentimento'], df_sent['contagem'], color=cores, edgecolor='white', width=0.6)
    
    for bar, pct, count in zip(bars, df_sent['percentual'], df_sent['contagem']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                f'{formatar_numero(count)}\n({pct:.1f}%)', ha='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    ax.set_title('Distribuição de Sentimento nas Menções de Preço', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()


def plot_preco_score_produto(df_precos, output_path):
    """Gráfico 6: Score por Produto (Top 15 negativos + Top 15 positivos)."""
    print("  Gerando: fig_preco_score_produto.png")
    
    score_por_produto = df_precos.groupby('produto')['score'].agg(['mean', 'count'])
    score_por_produto = score_por_produto[score_por_produto['count'] >= 50]
    score_por_produto = score_por_produto.sort_values('mean')
    
    top_negativos = score_por_produto.head(15)
    top_positivos = score_por_produto.tail(15)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    df_prod_viz = pd.concat([top_negativos, top_positivos]).drop_duplicates()
    df_prod_viz = df_prod_viz.sort_values('mean')
    
    labels_safe = [sanitizar_label(l) for l in df_prod_viz.index]
    cores = ['#27ae60' if v > 0 else '#e74c3c' if v < 0 else '#f39c12' for v in df_prod_viz['mean']]
    
    bars = ax.barh(range(len(df_prod_viz)), df_prod_viz['mean'], color=cores, edgecolor='white')
    ax.set_yticks(range(len(df_prod_viz)))
    ax.set_yticklabels(labels_safe)
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    
    for i, (bar, (idx, row)) in enumerate(zip(bars, df_prod_viz.iterrows())):
        x_pos = bar.get_width() + 0.03 if bar.get_width() >= 0 else bar.get_width() - 0.03
        ha = 'left' if bar.get_width() >= 0 else 'right'
        ax.text(x_pos, i, f'{row["mean"]:.2f} (n={int(row["count"])})',
                va='center', ha=ha, fontsize=9, fontweight='bold')
    
    ax.set_xlabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_xlim(-1.3, 1.3)
    ax.set_title('Score Médio de Preço por Produto\n(Top 15 negativos + Top 15 positivos, mín. 50 ocorrências)',
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()


def plot_preco_score_motivador(df_precos, output_path):
    """Gráfico 7: Score por Motivador."""
    print("  Gerando: fig_preco_score_motivador.png")
    
    score_por_motivador = df_precos.groupby('motivador')['score'].agg(['mean', 'count'])
    score_por_motivador = score_por_motivador.sort_values('mean')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df_mot = score_por_motivador.reset_index()
    df_mot.columns = ['motivador', 'mean', 'count']
    df_mot = df_mot.sort_values('mean')
    
    cores = ['#27ae60' if v > 0 else '#e74c3c' if v < 0 else '#f39c12' for v in df_mot['mean']]
    bars = ax.barh(df_mot['motivador'], df_mot['mean'], color=cores, edgecolor='white')
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    
    for bar, (idx, row) in zip(bars, df_mot.iterrows()):
        x_pos = bar.get_width() + 0.03 if bar.get_width() >= 0 else bar.get_width() - 0.03
        ha = 'left' if bar.get_width() >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                f'{row["mean"]:.2f} (n={formatar_numero(row["count"])})',
                va='center', ha=ha, fontsize=10, fontweight='bold')
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_xlim(-1.3, 1.3)
    ax.set_title('Score Médio de Preço por Motivador', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()


def plot_correlacao_preco_rating(df, output_path):
    """Gráfico 8: Correlação entre Percepção de Preço e Rating."""
    print("  Gerando: fig_correlacao_preco_rating.png")
    
    df_com_preco = df[df['preco_score_medio'].notna()].copy()
    
    if len(df_com_preco) < 10:
        print("    ⚠️ Dados insuficientes para correlação")
        return
    
    corr_preco, p_preco = stats.pearsonr(
        df_com_preco['preco_score_medio'],
        df_com_preco['rating']
    )
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sample_size = min(5000, len(df_com_preco))
    df_sample = df_com_preco.sample(n=sample_size, random_state=42)
    
    jitter_x = df_sample['preco_score_medio'] + np.random.normal(0, 0.03, len(df_sample))
    jitter_y = df_sample['rating'] + np.random.normal(0, 0.08, len(df_sample))
    
    ax.scatter(jitter_x, jitter_y, alpha=0.2, s=20, c='#3498db', edgecolors='none')
    
    z = np.polyfit(df_com_preco['preco_score_medio'], df_com_preco['rating'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(-1, 1, 100)
    ax.plot(x_line, p(x_line), color='#e74c3c', linestyle='--', linewidth=3,
            label=f'Tendência linear (r = {corr_preco:.3f})')
    
    ax.set_xlabel('Score de Sentimento sobre Preço (-1 a +1)', fontsize=12)
    ax.set_ylabel('Rating da Review (1-5)', fontsize=12)
    ax.set_title(f'Correlação entre Percepção de Preço e Rating\n(n = {len(df_com_preco):,})',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(0.5, 5.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()


def plot_efeito_resposta_dono(df, output_path):
    """Gráfico 9: Efeito da Resposta do Dono no Score de Preço."""
    print("  Gerando: fig_efeito_resposta_dono.png")
    
    if 'response_from_owner_text' not in df.columns:
        print("    ⚠️ Coluna 'response_from_owner_text' não encontrada")
        return
    
    df_preco_resp = df[df['preco_score_medio'].notna()].copy()
    df_preco_resp['tem_resposta'] = df_preco_resp['response_from_owner_text'].notna()
    
    score_com_resposta = df_preco_resp[df_preco_resp['tem_resposta']]['preco_score_medio'].mean()
    score_sem_resposta = df_preco_resp[~df_preco_resp['tem_resposta']]['preco_score_medio'].mean()
    n_com_resposta = df_preco_resp['tem_resposta'].sum()
    n_sem_resposta = (~df_preco_resp['tem_resposta']).sum()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    categorias = ['Com Resposta', 'Sem Resposta']
    scores = [score_com_resposta, score_sem_resposta]
    ns = [n_com_resposta, n_sem_resposta]
    cores = ['#27ae60' if s > 0 else '#e74c3c' for s in scores]
    
    bars = ax.bar(categorias, scores, color=cores, edgecolor='white', width=0.5)
    
    for bar, n, score in zip(bars, ns, scores):
        y_pos = bar.get_height() + 0.02 if bar.get_height() >= 0 else bar.get_height() - 0.05
        ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'{score:.3f}\n(n={formatar_numero(n)})', ha='center', fontsize=12, fontweight='bold')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    ax.set_ylim(-0.5, 0.3)
    ax.set_title('Efeito da Resposta do Dono no Score de Preço', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def main():
    print("="*60)
    print("GERAÇÃO DE GRÁFICOS - PROBLEMAS E PREÇO")
    print("="*60)
    
    # Criar diretório de saída
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Carregar dataset
    print(f"\n📂 Carregando {CONFIG['input_file']}...")
    
    if not os.path.exists(CONFIG['input_file']):
        print(f"❌ Arquivo não encontrado: {CONFIG['input_file']}")
        print("   Execute primeiro o script 04_analises_problemas_precos.py")
        return
    
    df = pd.read_excel(CONFIG['input_file'])
    print(f"   → {len(df):,} avaliações carregadas")
    
    # Verificar colunas necessárias
    required_cols = ['problemas_subcategorias', 'problemas_score_medio', 
                     'preco_produtos', 'preco_motivadores', 'preco_score_medio']
    
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"❌ Colunas não encontradas: {missing}")
        print("   Execute primeiro o script 04_analises_problemas_precos.py")
        return
    
    # Preparar dados
    print("\n📊 Preparando dados...")
    df_problemas = expandir_problemas(df)
    df_precos = expandir_precos(df)
    
    print(f"   → {len(df_problemas):,} evidências de problemas")
    print(f"   → {len(df_precos):,} evidências de preço")
    
    # Gerar gráficos
    print("\n🎨 Gerando gráficos...")
    
    # Gráficos de Problemas
    if len(df_problemas) > 0:
        plot_problemas_frequencia(df_problemas, os.path.join(CONFIG['output_dir'], 'fig_problemas_frequencia.png'))
        plot_problemas_gravidade(df_problemas, os.path.join(CONFIG['output_dir'], 'fig_problemas_gravidade.png'))
        plot_problemas_matriz(df_problemas, os.path.join(CONFIG['output_dir'], 'fig_problemas_matriz.png'))
        plot_coocorrencia_problemas(df, os.path.join(CONFIG['output_dir'], 'fig_coocorrencia_problemas.png'))
    else:
        print("  ⚠️ Sem dados de problemas para gerar gráficos")
    
    # Gráficos de Preço
    if len(df_precos) > 0:
        plot_preco_distribuicao_sentimento(df_precos, os.path.join(CONFIG['output_dir'], 'fig_preco_distribuicao_sentimento.png'))
        plot_preco_score_produto(df_precos, os.path.join(CONFIG['output_dir'], 'fig_preco_score_produto.png'))
        plot_preco_score_motivador(df_precos, os.path.join(CONFIG['output_dir'], 'fig_preco_score_motivador.png'))
        plot_correlacao_preco_rating(df, os.path.join(CONFIG['output_dir'], 'fig_correlacao_preco_rating.png'))
        plot_efeito_resposta_dono(df, os.path.join(CONFIG['output_dir'], 'fig_efeito_resposta_dono.png'))
    else:
        print("  ⚠️ Sem dados de preço para gerar gráficos")
    
    print("\n" + "="*60)
    print("✅ GRÁFICOS GERADOS COM SUCESSO!")
    print(f"   Pasta: {CONFIG['output_dir']}/")
    print("="*60)


if __name__ == "__main__":
    main()
