#!/usr/bin/env python3
"""
================================================================================
ANÁLISE DE POSICIONAMENTO DIGITAL
================================================================================

Este script analisa o impacto de variáveis de posicionamento digital:
- Resposta do dono às avaliações
- Status de Local Guide do avaliador

GRÁFICOS GERADOS:
  1. fig_resposta_rating_medio.png        - Rating médio com/sem resposta
  2. fig_resposta_pct_por_rating.png      - % de respostas por rating
  3. fig_localguide_rating_medio.png      - Rating médio Local Guide vs Não
  4. fig_localguide_distribuicao_rating.png - Distribuição de ratings
  5. fig_interacao_resposta_localguide.png - Interação Resposta × Local Guide

ENTRADA:
  - dataset_analises_completas.xlsx (ou dataset com colunas necessárias)

SAÍDA:
  - 5 gráficos PNG na pasta outputs/
  - Sumário estatístico no console

EXECUÇÃO:
  python 06_analise_posicionamento_digital.py

================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
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

# Cor principal
COR_PRINCIPAL = '#3498DB'
COR_SECUNDARIA = '#95A5A6'

# Configurações visuais
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
sns.set_style("whitegrid")

# ============================================================
# FUNÇÕES DE GRÁFICOS
# ============================================================

def plot_resposta_rating_medio(df, output_path):
    """Gráfico 1: Rating médio com/sem resposta do dono."""
    print("  Gerando: fig_resposta_rating_medio.png")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    stats_resp = df.groupby('tem_resposta_dono')['rating'].agg(['mean', 'count']).reset_index()
    stats_resp['label'] = stats_resp['tem_resposta_dono'].map({True: 'Com Resposta', False: 'Sem Resposta'})
    stats_resp = stats_resp.sort_values('tem_resposta_dono', ascending=False)
    
    sns.barplot(data=stats_resp, x='label', y='mean', ax=ax, color=COR_PRINCIPAL)
    
    for i, row in enumerate(stats_resp.itertuples()):
        ax.text(i, row.mean + 0.05, f"{row.mean:.2f}\n(n={int(row.count):,})", 
                ha='center', va='bottom', fontsize=11)
    
    ax.axhline(y=df['rating'].mean(), color='gray', linestyle='--', alpha=0.7,
               label=f'Média geral ({df["rating"].mean():.2f})')
    ax.set_ylabel('Rating Médio')
    ax.set_xlabel('')
    ax.set_ylim(0, 5.2)
    ax.legend(loc='lower right')
    sns.despine(ax=ax)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()


def plot_resposta_pct_por_rating(df, output_path):
    """Gráfico 2: % de respostas por rating."""
    print("  Gerando: fig_resposta_pct_por_rating.png")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    pct_por_rating = df.groupby('rating')['tem_resposta_dono'].mean() * 100
    pct_df = pct_por_rating.reset_index()
    pct_df.columns = ['rating', 'pct_resposta']
    
    sns.barplot(data=pct_df, x='rating', y='pct_resposta', ax=ax, color=COR_PRINCIPAL)
    
    for i, row in enumerate(pct_df.itertuples()):
        ax.text(i, row.pct_resposta + 0.3, f"{row.pct_resposta:.1f}%", 
                ha='center', va='bottom', fontsize=11)
    
    ax.set_ylabel('% Reviews com Resposta do Dono')
    ax.set_xlabel('Rating')
    sns.despine(ax=ax)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()


def plot_localguide_rating_medio(df, output_path):
    """Gráfico 3: Rating médio Local Guide vs Não Guide."""
    print("  Gerando: fig_localguide_rating_medio.png")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    stats_guide = df.groupby('is_local_guide')['rating'].agg(['mean', 'count']).reset_index()
    stats_guide['label'] = stats_guide['is_local_guide'].map({True: 'Local Guide', False: 'Não Guide'})
    stats_guide = stats_guide.sort_values('is_local_guide', ascending=False)
    
    sns.barplot(data=stats_guide, x='label', y='mean', ax=ax, color=COR_PRINCIPAL)
    
    for i, row in enumerate(stats_guide.itertuples()):
        ax.text(i, row.mean + 0.05, f"{row.mean:.2f}\n(n={int(row.count):,})", 
                ha='center', va='bottom', fontsize=11)
    
    ax.axhline(y=df['rating'].mean(), color='gray', linestyle='--', alpha=0.7,
               label=f'Média geral ({df["rating"].mean():.2f})')
    ax.set_ylabel('Rating Médio')
    ax.set_xlabel('')
    ax.set_ylim(0, 5.2)
    ax.legend(loc='lower right')
    sns.despine(ax=ax)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()


def plot_localguide_distribuicao_rating(df, output_path):
    """Gráfico 4: Distribuição de ratings por Local Guide."""
    print("  Gerando: fig_localguide_distribuicao_rating.png")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    dist_data = []
    for guide in [True, False]:
        subset = df[df['is_local_guide'] == guide]
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
    
    sns.barplot(data=dist_df, x='rating', y='pct', hue='grupo', ax=ax, 
                palette=[COR_PRINCIPAL, COR_SECUNDARIA])
    
    ax.set_ylabel('% das Avaliações')
    ax.set_xlabel('Rating')
    ax.legend(title='')
    sns.despine(ax=ax)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()


def plot_interacao_resposta_localguide(df, output_path):
    """Gráfico 5: Interação Resposta × Local Guide."""
    print("  Gerando: fig_interacao_resposta_localguide.png")
    
    df = df.copy()
    df['grupo_interacao'] = 'Outros'
    df.loc[(df['is_local_guide']==True) & (df['tem_resposta_dono']==True), 'grupo_interacao'] = 'Local Guide + Resposta'
    df.loc[(df['is_local_guide']==True) & (df['tem_resposta_dono']==False), 'grupo_interacao'] = 'Local Guide sem Resposta'
    df.loc[(df['is_local_guide']==False) & (df['tem_resposta_dono']==True), 'grupo_interacao'] = 'Não Guide + Resposta'
    df.loc[(df['is_local_guide']==False) & (df['tem_resposta_dono']==False), 'grupo_interacao'] = 'Não Guide sem Resposta'
    
    stats_inter = df.groupby('grupo_interacao')['rating'].agg(['mean', 'count']).reset_index()
    ordem = ['Local Guide + Resposta', 'Local Guide sem Resposta', 'Não Guide + Resposta', 'Não Guide sem Resposta']
    stats_inter['grupo_interacao'] = pd.Categorical(stats_inter['grupo_interacao'], categories=ordem, ordered=True)
    stats_inter = stats_inter.sort_values('grupo_interacao')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.barplot(data=stats_inter, y='grupo_interacao', x='mean', ax=ax, color=COR_PRINCIPAL, orient='h')
    
    for i, row in enumerate(stats_inter.itertuples()):
        ax.text(row.mean + 0.03, i, f"{row.mean:.2f} (n={int(row.count):,})", 
                va='center', fontsize=10)
    
    ax.axvline(x=df['rating'].mean(), color='gray', linestyle='--', alpha=0.7,
               label=f'Média geral ({df["rating"].mean():.2f})')
    ax.set_xlabel('Rating Médio')
    ax.set_ylabel('')
    ax.set_xlim(0, 5.2)
    ax.legend(loc='lower right')
    sns.despine(ax=ax)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=CONFIG['dpi'], bbox_inches='tight', facecolor='white')
    plt.close()


def gerar_sumario_estatistico(df):
    """Gera sumário estatístico das análises."""
    
    print("\n" + "="*60)
    print("SUMÁRIO ESTATÍSTICO")
    print("="*60)
    
    # Análise 1: Resposta do dono
    com_resp = df[df['tem_resposta_dono']==True]['rating']
    sem_resp = df[df['tem_resposta_dono']==False]['rating']
    t_resp, p_resp = stats.ttest_ind(com_resp, sem_resp)
    
    print(f"""
ANÁLISE 1: RESPOSTA DO DONO
- Reviews com resposta: {len(com_resp):,} ({len(com_resp)/len(df)*100:.1f}%)
- Rating COM resposta: {com_resp.mean():.2f} (±{com_resp.std():.2f})
- Rating SEM resposta: {sem_resp.mean():.2f} (±{sem_resp.std():.2f})
- Diferença: {com_resp.mean() - sem_resp.mean():+.3f}
- Teste t: t={t_resp:.3f}, p={p_resp:.4f}
- Significativo: {'SIM' if p_resp < 0.05 else 'NÃO'}
""")
    
    # Análise 2: Local Guides
    local_g = df[df['is_local_guide']==True]['rating']
    nao_g = df[df['is_local_guide']==False]['rating']
    t_guide, p_guide = stats.ttest_ind(local_g, nao_g)
    
    print(f"""
ANÁLISE 2: LOCAL GUIDES
- Reviews Local Guide: {len(local_g):,} ({len(local_g)/len(df)*100:.1f}%)
- Rating LOCAL GUIDE: {local_g.mean():.2f} (±{local_g.std():.2f})
- Rating NÃO GUIDE: {nao_g.mean():.2f} (±{nao_g.std():.2f})
- Diferença: {local_g.mean() - nao_g.mean():+.3f}
- Teste t: t={t_guide:.3f}, p={p_guide:.6f}
- Significativo: {'SIM' if p_guide < 0.05 else 'NÃO'}
""")
    
    return {
        'resposta_dono': {
            'n_com': len(com_resp),
            'n_sem': len(sem_resp),
            'media_com': com_resp.mean(),
            'media_sem': sem_resp.mean(),
            't': t_resp,
            'p': p_resp
        },
        'local_guide': {
            'n_guide': len(local_g),
            'n_nao': len(nao_g),
            'media_guide': local_g.mean(),
            'media_nao': nao_g.mean(),
            't': t_guide,
            'p': p_guide
        }
    }


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def main():
    print("="*60)
    print("ANÁLISE DE POSICIONAMENTO DIGITAL")
    print("="*60)
    
    # Criar diretório de saída
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Carregar dataset
    print(f"\n📂 Carregando {CONFIG['input_file']}...")
    
    if not os.path.exists(CONFIG['input_file']):
        print(f"❌ Arquivo não encontrado: {CONFIG['input_file']}")
        return
    
    df = pd.read_excel(CONFIG['input_file'])
    print(f"   → {len(df):,} avaliações carregadas")
    
    # Verificar colunas necessárias
    required_cols = ['rating', 'response_from_owner_text', 'is_local_guide']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"❌ Colunas não encontradas: {missing}")
        return
    
    # Preparar variáveis
    print("\n📊 Preparando variáveis...")
    df['tem_resposta_dono'] = df['response_from_owner_text'].notna()
    df['is_local_guide'] = df['is_local_guide'].fillna(False).astype(bool)
    
    print(f"   → Reviews com resposta do dono: {df['tem_resposta_dono'].sum():,} ({df['tem_resposta_dono'].mean()*100:.1f}%)")
    print(f"   → Reviews de Local Guides: {df['is_local_guide'].sum():,} ({df['is_local_guide'].mean()*100:.1f}%)")
    
    # Gerar gráficos
    print("\n🎨 Gerando gráficos...")
    
    plot_resposta_rating_medio(df, os.path.join(CONFIG['output_dir'], 'fig_resposta_rating_medio.png'))
    plot_resposta_pct_por_rating(df, os.path.join(CONFIG['output_dir'], 'fig_resposta_pct_por_rating.png'))
    plot_localguide_rating_medio(df, os.path.join(CONFIG['output_dir'], 'fig_localguide_rating_medio.png'))
    plot_localguide_distribuicao_rating(df, os.path.join(CONFIG['output_dir'], 'fig_localguide_distribuicao_rating.png'))
    plot_interacao_resposta_localguide(df, os.path.join(CONFIG['output_dir'], 'fig_interacao_resposta_localguide.png'))
    
    # Gerar sumário estatístico
    gerar_sumario_estatistico(df)
    
    print("\n" + "="*60)
    print("✅ ANÁLISE CONCLUÍDA!")
    print(f"   Pasta: {CONFIG['output_dir']}/")
    print("\nFIGURAS GERADAS:")
    print("  1. fig_resposta_rating_medio.png")
    print("  2. fig_resposta_pct_por_rating.png")
    print("  3. fig_localguide_rating_medio.png")
    print("  4. fig_localguide_distribuicao_rating.png")
    print("  5. fig_interacao_resposta_localguide.png")
    print("="*60)


if __name__ == "__main__":
    main()
