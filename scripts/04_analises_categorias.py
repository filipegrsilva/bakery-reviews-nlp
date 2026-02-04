#!/usr/bin/env python3
"""
================================================================================
ANÁLISE DETALHADA DE CATEGORIAS
================================================================================

Este script realiza duas etapas:

ETAPA 1 - DETALHAMENTO VIA LLM:
  - Extrai evidências de PROBLEMAS e PREÇO do JSON de sentimentos
  - Classifica PROBLEMAS em subcategorias (ATENDIMENTO, DEMORA, PRODUTO, etc.)
  - Extrai PRODUTO e MOTIVADOR das menções de preço
  - Calcula scores médios de sentimento

ETAPA 2 - ANÁLISES ESTATÍSTICAS:
  - Frequência e gravidade de problemas
  - Distribuição de sentimento sobre preço
  - Correlações score × rating
  - Co-ocorrência de problemas
  - Efeito da resposta do dono

ENTRADA:
  - dataset_com_sentimentos.xlsx (saída do script 03)

SAÍDA:
  - dataset_analises_completas.xlsx
  - analises_problemas_precos.txt (relatório estatístico)

EXECUÇÃO:
  python 04_analises_problemas_precos.py

REQUISITOS:
  - Ollama rodando com modelo llama3.1:8b
  - Dataset com coluna 'llm_analise_json'

================================================================================
"""

import os
import sys
import json
import time
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import stats
import threading

# ============================================================
# CONFIGURAÇÕES
# ============================================================

CONFIG = {
    # Arquivos
    "input_file": "dataset_com_sentimentos.xlsx",
    "output_file": "dataset_analises_completas.xlsx",
    "output_pickle": "dataset_analises_completas.pkl",
    "report_file": "analises_problemas_precos.txt",
    "checkpoint_file": "checkpoint_analises.json",
    "log_file": "log_analises.txt",
    
    # Ollama
    "ollama_url": "http://localhost:11434/api/generate",
    "model": "llama3.1:8b",
    
    # Processamento
    "num_workers": 8,
    "batch_checkpoint": 1500,
    "max_retries": 3,
    "timeout": 90,
}

# Score de sentimento
SCORE_MAP = {'positivo': 1, 'neutro': 0, 'negativo': -1}

# Subcategorias válidas de problemas
SUBCATEGORIAS_PROBLEMAS = [
    'ATENDIMENTO', 'DEMORA', 'PRODUTO', 'HIGIENE', 
    'INFRAESTRUTURA', 'FALTA', 'COBRANCA', 'OUTRO'
]

# Motivadores válidos de preço
MOTIVADORES_PRECO = [
    'PORCAO_PEQUENA', 'QUALIDADE_RUIM', 'CARO_REGIAO', 
    'PRECO_AUMENTOU', 'BOM_CUSTO_BENEFICIO', 'PRECO_ACESSIVEL', 
    'NAO_ESPECIFICADO'
]

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(CONFIG["log_file"]),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

checkpoint_lock = threading.Lock()

# ============================================================
# PROMPTS PARA LLM
# ============================================================

PROMPT_PROBLEMA = """Classifique este problema relatado em uma padaria.

Problema: "{evidencia}"

Categorias possíveis:
- ATENDIMENTO: funcionário rude, descaso, erro de atendente
- DEMORA: espera longa, fila, lentidão
- PRODUTO: qualidade ruim, produto velho, mal preparado
- HIGIENE: sujeira, inseto, cabelo na comida
- INFRAESTRUTURA: desconforto, barulho, banheiro sujo, estacionamento
- FALTA: produto indisponível, acabou
- COBRANCA: conta errada, cobrança indevida, troco errado

Responda apenas UMA palavra: ATENDIMENTO, DEMORA, PRODUTO, HIGIENE, INFRAESTRUTURA, FALTA ou COBRANCA"""


PROMPT_PRODUTO = """Extraia o produto EXATO mencionado nesta frase sobre preço:

Evidência: "{evidencia}"

REGRAS:
1. Se a frase menciona um produto ESPECÍFICO (pão, café, salgado, bolo, pizza, etc), responda o nome do produto.
2. Se a frase NÃO menciona nenhum produto específico, responda apenas: GERAL
3. NÃO INVENTE produtos que não estão escritos na frase.

Exemplos:
- "pão francês muito caro" → pao frances
- "salgado caro e pequenininho" → salgado
- "café caro" → cafe
- "preço justo" → GERAL
- "muito caro" → GERAL
- "preços elevados" → GERAL
- "caro demais" → GERAL

Responda APENAS o produto ou GERAL:"""


PROMPT_MOTIVADOR = """Qual o MOTIVO da percepção de preço nesta frase?

Evidência: "{evidencia}"
Sentimento: {sentimento}

OPÇÕES:
- PORCAO_PEQUENA: porção pequena, pouca quantidade
- QUALIDADE_RUIM: qualidade não condiz com preço
- CARO_REGIAO: caro comparado à região/concorrência
- PRECO_AUMENTOU: preço subiu, era mais barato
- BOM_CUSTO_BENEFICIO: vale o que paga
- PRECO_ACESSIVEL: barato, em conta
- NAO_ESPECIFICADO: não explica o motivo

Responda APENAS uma das opções acima:"""

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def verificar_ollama():
    """Verifica se Ollama está rodando."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=10)
        if resp.status_code == 200:
            logger.info("✅ Ollama está rodando")
            return True
    except:
        pass
    
    logger.error("❌ Ollama não está rodando!")
    logger.error("   Execute: ollama serve &")
    return False


def chamar_ollama(prompt, max_retries=None):
    """Chama o Ollama com retry."""
    if max_retries is None:
        max_retries = CONFIG["max_retries"]
    
    payload = {
        "model": CONFIG["model"],
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 30}
    }
    
    for tentativa in range(max_retries):
        try:
            response = requests.post(
                CONFIG["ollama_url"], 
                json=payload, 
                timeout=CONFIG["timeout"]
            )
            response.raise_for_status()
            return response.json().get('response', '').strip()
        except Exception as e:
            if tentativa < max_retries - 1:
                time.sleep(2 ** tentativa)
    return "ERRO"


def carregar_checkpoint():
    """Carrega checkpoint se existir."""
    if os.path.exists(CONFIG["checkpoint_file"]):
        with open(CONFIG["checkpoint_file"], 'r') as f:
            return json.load(f)
    return {'problemas': {}, 'precos': {}}


def salvar_checkpoint(checkpoint):
    """Salva checkpoint."""
    with checkpoint_lock:
        with open(CONFIG["checkpoint_file"], 'w') as f:
            json.dump(checkpoint, f)


def calcular_score_medio(sentimentos):
    """Calcula score médio a partir de lista de sentimentos."""
    if not sentimentos:
        return None
    scores = [SCORE_MAP.get(s, 0) for s in sentimentos]
    return sum(scores) / len(scores)


# ============================================================
# ETAPA 1: DETALHAMENTO VIA LLM
# ============================================================

def classificar_problema(evidencia):
    """Classifica um problema em subcategoria."""
    prompt = PROMPT_PROBLEMA.format(evidencia=evidencia[:300])
    resposta = chamar_ollama(prompt).upper()
    
    for cat in SUBCATEGORIAS_PROBLEMAS:
        if cat in resposta:
            return cat
    return 'OUTRO'


def classificar_preco(evidencia, sentimento):
    """Classifica preço extraindo produto e motivador."""
    # Produto
    prompt_prod = PROMPT_PRODUTO.format(evidencia=evidencia[:300])
    resp_prod = chamar_ollama(prompt_prod)
    produto = resp_prod.lower().strip().replace('.', '').replace(',', '')
    palavras = produto.split()[:2]
    produto = ' '.join(palavras) if palavras else 'geral'
    
    # Motivador
    prompt_mot = PROMPT_MOTIVADOR.format(evidencia=evidencia[:300], sentimento=sentimento)
    resp_mot = chamar_ollama(prompt_mot).upper()
    
    motivador = 'NAO_ESPECIFICADO'
    for mot in MOTIVADORES_PRECO:
        if mot in resp_mot:
            motivador = mot
            break
    
    return produto, motivador


def processar_linha(args):
    """Processa uma linha do dataset."""
    idx, json_str, checkpoint = args
    
    resultado = {
        'problemas_subcategorias': [],
        'problemas_sentimentos': [],
        'preco_produtos': [],
        'preco_motivadores': [],
        'preco_sentimentos': []
    }
    
    if pd.isna(json_str):
        return idx, resultado
    
    try:
        parsed = json.loads(json_str)
        
        for i, a in enumerate(parsed.get('analises', [])):
            categoria = a.get('categoria', '').lower().strip()
            evidencia = a.get('evidencia', '')
            sentimento = a.get('sentimento', '').lower().strip()
            
            if categoria == 'problemas':
                key = f"{idx}_{i}"
                
                if key in checkpoint['problemas']:
                    subcategoria = checkpoint['problemas'][key]
                else:
                    subcategoria = classificar_problema(evidencia)
                    with checkpoint_lock:
                        checkpoint['problemas'][key] = subcategoria
                
                resultado['problemas_subcategorias'].append(subcategoria)
                resultado['problemas_sentimentos'].append(sentimento)
            
            elif categoria == 'preco':
                key = f"{idx}_{i}"
                
                if key in checkpoint['precos']:
                    produto = checkpoint['precos'][key]['produto']
                    motivador = checkpoint['precos'][key]['motivador']
                else:
                    produto, motivador = classificar_preco(evidencia, sentimento)
                    with checkpoint_lock:
                        checkpoint['precos'][key] = {'produto': produto, 'motivador': motivador}
                
                resultado['preco_produtos'].append(produto)
                resultado['preco_motivadores'].append(motivador)
                resultado['preco_sentimentos'].append(sentimento)
    
    except Exception as e:
        logger.warning(f"Erro na linha {idx}: {e}")
    
    return idx, resultado


def executar_detalhamento_llm(df):
    """Executa o detalhamento via LLM."""
    logger.info("\n" + "="*70)
    logger.info("ETAPA 1: DETALHAMENTO VIA LLM")
    logger.info("="*70)
    
    # Verificar Ollama
    if not verificar_ollama():
        logger.error("Abortando: Ollama não está acessível")
        return df
    
    # Carregar checkpoint
    checkpoint = carregar_checkpoint()
    logger.info(f"Checkpoint: {len(checkpoint['problemas'])} problemas, {len(checkpoint['precos'])} preços já processados")
    
    # Preparar argumentos
    args_list = [(idx, row.get('llm_analise_json'), checkpoint) 
                 for idx, row in df.iterrows()]
    
    # Contar total de análises
    total_problemas = 0
    total_precos = 0
    for _, json_str, _ in args_list:
        if pd.notna(json_str):
            try:
                parsed = json.loads(json_str)
                for a in parsed.get('analises', []):
                    cat = a.get('categoria', '').lower().strip()
                    if cat == 'problemas':
                        total_problemas += 1
                    elif cat == 'preco':
                        total_precos += 1
            except:
                pass
    
    logger.info(f"\nTotal a processar:")
    logger.info(f"  → {total_problemas:,} evidências de PROBLEMAS")
    logger.info(f"  → {total_precos:,} evidências de PREÇO")
    
    # Processar
    logger.info(f"\n🚀 Processando com {CONFIG['num_workers']} workers...")
    
    resultados = {}
    processados = 0
    
    with ThreadPoolExecutor(max_workers=CONFIG['num_workers']) as executor:
        futures = {executor.submit(processar_linha, args): args[0] for args in args_list}
        
        for future in as_completed(futures):
            idx, resultado = future.result()
            resultados[idx] = resultado
            processados += 1
            
            if processados % 500 == 0:
                logger.info(f"   Linhas processadas: {processados:,}/{len(df):,} ({100*processados/len(df):.1f}%)")
                salvar_checkpoint(checkpoint)
    
    salvar_checkpoint(checkpoint)
    
    # Adicionar colunas ao dataset
    logger.info("\n📝 Adicionando colunas ao dataset...")
    
    df['problemas_subcategorias'] = None
    df['problemas_score_medio'] = None
    df['preco_produtos'] = None
    df['preco_motivadores'] = None
    df['preco_score_medio'] = None
    
    for idx, resultado in resultados.items():
        if resultado['problemas_subcategorias']:
            df.at[idx, 'problemas_subcategorias'] = '|'.join(resultado['problemas_subcategorias'])
            df.at[idx, 'problemas_score_medio'] = calcular_score_medio(resultado['problemas_sentimentos'])
        
        if resultado['preco_produtos']:
            df.at[idx, 'preco_produtos'] = '|'.join(resultado['preco_produtos'])
            df.at[idx, 'preco_motivadores'] = '|'.join(resultado['preco_motivadores'])
            df.at[idx, 'preco_score_medio'] = calcular_score_medio(resultado['preco_sentimentos'])
    
    # Limpar checkpoint
    if os.path.exists(CONFIG["checkpoint_file"]):
        os.remove(CONFIG["checkpoint_file"])
    
    logger.info("✅ Detalhamento LLM concluído")
    return df


# ============================================================
# ETAPA 2: ANÁLISES ESTATÍSTICAS
# ============================================================

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
                prod = prod.strip()
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


def executar_analises_estatisticas(df):
    """Executa todas as análises estatísticas."""
    logger.info("\n" + "="*70)
    logger.info("ETAPA 2: ANÁLISES ESTATÍSTICAS")
    logger.info("="*70)
    
    report_lines = []
    report_lines.append("="*70)
    report_lines.append("RELATÓRIO DE ANÁLISES - PROBLEMAS E PREÇO")
    report_lines.append(f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("="*70)
    
    # =========================================================
    # VISÃO GERAL
    # =========================================================
    report_lines.append("\n" + "="*70)
    report_lines.append("VISÃO GERAL DO DATASET")
    report_lines.append("="*70)
    
    total_reviews = len(df)
    reviews_problemas = df['problemas_subcategorias'].notna().sum()
    reviews_preco = df['preco_score_medio'].notna().sum()
    
    report_lines.append(f"\nTotal de reviews: {total_reviews:,}")
    report_lines.append(f"Reviews com problemas: {reviews_problemas:,} ({reviews_problemas/total_reviews*100:.1f}%)")
    report_lines.append(f"Reviews com menção a preço: {reviews_preco:,} ({reviews_preco/total_reviews*100:.1f}%)")
    
    logger.info(f"Total de reviews: {total_reviews:,}")
    logger.info(f"Reviews com problemas: {reviews_problemas:,}")
    logger.info(f"Reviews com menção a preço: {reviews_preco:,}")
    
    # =========================================================
    # ANÁLISE DE PROBLEMAS
    # =========================================================
    report_lines.append("\n" + "="*70)
    report_lines.append("ANÁLISE DE PROBLEMAS")
    report_lines.append("="*70)
    
    df_problemas = expandir_problemas(df)
    
    if len(df_problemas) > 0:
        report_lines.append(f"\nTotal de evidências de problemas: {len(df_problemas):,}")
        
        # Frequência por subcategoria
        freq_problemas = df_problemas['subcategoria'].value_counts()
        freq_problemas_pct = (freq_problemas / freq_problemas.sum() * 100).round(1)
        
        report_lines.append("\nFrequência por Subcategoria:")
        for cat, count in freq_problemas.items():
            report_lines.append(f"  {cat}: {count:,} ({freq_problemas_pct[cat]:.1f}%)")
        
        # Score médio por subcategoria
        score_por_subcat = df_problemas.groupby('subcategoria')['score'].mean().sort_values()
        
        report_lines.append("\nScore Médio de Sentimento por Subcategoria (mais negativo = mais grave):")
        for cat, score in score_por_subcat.items():
            report_lines.append(f"  {cat}: {score:.3f}")
        
        # Score médio geral
        score_medio_problemas = df_problemas['score'].mean()
        report_lines.append(f"\nScore médio geral de problemas: {score_medio_problemas:.3f}")
        
        # Co-ocorrência
        coocorrencias = analisar_coocorrencia(df)
        if coocorrencias:
            report_lines.append("\nCo-ocorrência de Problemas (top 10):")
            for par, count in coocorrencias.most_common(10):
                report_lines.append(f"  {par[0]} + {par[1]}: {count}")
        
        logger.info(f"Problemas analisados: {len(df_problemas):,} evidências")
    else:
        report_lines.append("\nNenhuma evidência de problema encontrada.")
    
    # =========================================================
    # ANÁLISE DE PREÇO
    # =========================================================
    report_lines.append("\n" + "="*70)
    report_lines.append("ANÁLISE DE PREÇO")
    report_lines.append("="*70)
    
    df_precos = expandir_precos(df)
    
    if len(df_precos) > 0:
        report_lines.append(f"\nTotal de evidências de preço: {len(df_precos):,}")
        
        # Distribuição de sentimento
        def classificar_sentimento(score):
            if pd.isna(score):
                return None
            if score > 0:
                return 'positivo'
            elif score < 0:
                return 'negativo'
            else:
                return 'neutro'
        
        df_precos['sentimento'] = df_precos['score'].apply(classificar_sentimento)
        dist_sentimento = df_precos['sentimento'].value_counts()
        dist_sentimento_pct = (dist_sentimento / dist_sentimento.sum() * 100).round(1)
        
        report_lines.append("\nDistribuição de Sentimento:")
        for sent, count in dist_sentimento.items():
            report_lines.append(f"  {sent}: {count:,} ({dist_sentimento_pct[sent]:.1f}%)")
        
        # Score por produto (top 15)
        score_por_produto = df_precos.groupby('produto')['score'].agg(['mean', 'count'])
        score_por_produto = score_por_produto[score_por_produto['count'] >= 50]
        score_por_produto = score_por_produto.sort_values('mean')
        
        report_lines.append(f"\nScore por Produto (mín. 50 ocorrências, {len(score_por_produto)} produtos):")
        report_lines.append("\nTop 10 mais negativos:")
        for prod, row in score_por_produto.head(10).iterrows():
            report_lines.append(f"  {prod}: {row['mean']:.3f} (n={int(row['count'])})")
        
        report_lines.append("\nTop 10 mais positivos:")
        for prod, row in score_por_produto.tail(10).iloc[::-1].iterrows():
            report_lines.append(f"  {prod}: {row['mean']:.3f} (n={int(row['count'])})")
        
        # Score por motivador
        score_por_motivador = df_precos.groupby('motivador')['score'].agg(['mean', 'count'])
        score_por_motivador = score_por_motivador.sort_values('mean')
        
        report_lines.append("\nScore por Motivador:")
        for mot, row in score_por_motivador.iterrows():
            report_lines.append(f"  {mot}: {row['mean']:.3f} (n={int(row['count']):,})")
        
        # Score médio geral
        score_medio_preco = df_precos['score'].mean()
        report_lines.append(f"\nScore médio geral de preço: {score_medio_preco:.3f}")
        
        logger.info(f"Preços analisados: {len(df_precos):,} evidências")
    else:
        report_lines.append("\nNenhuma evidência de preço encontrada.")
    
    # =========================================================
    # CORRELAÇÕES
    # =========================================================
    report_lines.append("\n" + "="*70)
    report_lines.append("CORRELAÇÕES")
    report_lines.append("="*70)
    
    # Correlação problemas score × rating
    df_com_problemas = df[df['problemas_score_medio'].notna()]
    if len(df_com_problemas) > 10:
        corr_prob, p_prob = stats.pearsonr(
            df_com_problemas['problemas_score_medio'],
            df_com_problemas['rating']
        )
        report_lines.append(f"\nProblemas Score × Rating: r={corr_prob:.3f} (p={p_prob:.4f})")
    
    # Correlação preço score × rating
    df_com_preco = df[df['preco_score_medio'].notna()]
    if len(df_com_preco) > 10:
        corr_preco, p_preco = stats.pearsonr(
            df_com_preco['preco_score_medio'],
            df_com_preco['rating']
        )
        report_lines.append(f"Preço Score × Rating: r={corr_preco:.3f} (p={p_preco:.4f})")
    
    # =========================================================
    # EFEITO DA RESPOSTA DO DONO
    # =========================================================
    if 'response_from_owner_text' in df.columns:
        report_lines.append("\n" + "="*70)
        report_lines.append("EFEITO DA RESPOSTA DO DONO")
        report_lines.append("="*70)
        
        df_preco_resp = df[df['preco_score_medio'].notna()].copy()
        df_preco_resp['tem_resposta'] = df_preco_resp['response_from_owner_text'].notna()
        
        score_com_resp = df_preco_resp[df_preco_resp['tem_resposta']]['preco_score_medio'].mean()
        score_sem_resp = df_preco_resp[~df_preco_resp['tem_resposta']]['preco_score_medio'].mean()
        n_com_resp = df_preco_resp['tem_resposta'].sum()
        n_sem_resp = (~df_preco_resp['tem_resposta']).sum()
        
        report_lines.append(f"\nScore de Preço:")
        report_lines.append(f"  Com resposta: {score_com_resp:.3f} (n={n_com_resp:,})")
        report_lines.append(f"  Sem resposta: {score_sem_resp:.3f} (n={n_sem_resp:,})")
        report_lines.append(f"  Diferença: {score_com_resp - score_sem_resp:+.3f}")
        
        if n_com_resp > 10 and n_sem_resp > 10:
            t_stat, p_value = stats.ttest_ind(
                df_preco_resp[df_preco_resp['tem_resposta']]['preco_score_medio'],
                df_preco_resp[~df_preco_resp['tem_resposta']]['preco_score_medio']
            )
            report_lines.append(f"  Teste t: t={t_stat:.2f}, p={p_value:.4f}")
    
    # =========================================================
    # SALVAR RELATÓRIO
    # =========================================================
    report_lines.append("\n" + "="*70)
    report_lines.append("FIM DO RELATÓRIO")
    report_lines.append("="*70)
    
    report_text = '\n'.join(report_lines)
    
    with open(CONFIG["report_file"], 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    logger.info(f"\n📄 Relatório salvo: {CONFIG['report_file']}")
    
    # Imprimir resumo
    print("\n" + report_text)
    
    return df


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def main():
    logger.info("="*70)
    logger.info("ANÁLISE DETALHADA DE CATEGORIAS")
    logger.info("="*70)
    logger.info(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Carregar dataset
    logger.info(f"\n📂 Carregando {CONFIG['input_file']}...")
    
    if not os.path.exists(CONFIG['input_file']):
        logger.error(f"Arquivo não encontrado: {CONFIG['input_file']}")
        sys.exit(1)
    
    df = pd.read_excel(CONFIG['input_file'])
    logger.info(f"   → {len(df):,} avaliações carregadas")
    
    # Verificar se coluna de análise existe
    if 'llm_analise_json' not in df.columns:
        logger.error("Coluna 'llm_analise_json' não encontrada!")
        logger.error("Execute primeiro o script 03_analise_sentimentos_llm.py")
        sys.exit(1)
    
    # ETAPA 1: Detalhamento via LLM
    df = executar_detalhamento_llm(df)
    
    # ETAPA 2: Análises estatísticas
    df = executar_analises_estatisticas(df)
    
    # Salvar dataset
    logger.info(f"\n💾 Salvando {CONFIG['output_file']}...")
    df.to_excel(CONFIG['output_file'], index=False)
    df.to_pickle(CONFIG['output_pickle'])
    
    logger.info("\n" + "="*70)
    logger.info(f"✅ Concluído: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Arquivo salvo: {CONFIG['output_file']}")
    logger.info(f"Relatório: {CONFIG['report_file']}")
    logger.info("="*70)


if __name__ == "__main__":
    main()
