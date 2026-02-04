"""
================================================================================
SCRIPT 01: EXTRAÇÃO DE TÓPICOS COM BERTOPIC
================================================================================
Pipeline da Dissertação - Etapa 1

Entrada: dataset_full.csv
Saída: 
  - topicos_para_selecao.json (tópicos para curadoria manual)
  - dataset_full.csv (atualizado com coluna 'topic')
  - bertopic_model/ (modelo salvo)
  - topics.pkl, df_com_topicos.pkl (checkpoints)

Tempo estimado: 2-3 horas
================================================================================
"""

import os
import pickle
import numpy as np
import pandas as pd
import json
import gc
import re
import shutil
from datetime import datetime

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

CONFIG = {
    "input_file": "dataset_full.csv",
    "csv_separator": "|",
    "output_json": "topicos_para_selecao.json",
    "output_csv": "dataset_full.csv",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "min_cluster_size": 500,
    "min_samples": 50,
    "umap_n_components": 5,
    "umap_n_neighbors": 15,
    "random_state": 42,
}

CATEGORIAS_VALIDAS = ['comida', 'atendimento', 'ambiente', 'preco', 'problemas']

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def verificar_bibliotecas():
    """Verifica e importa bibliotecas necessárias"""
    print("Verificando bibliotecas...")
    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        from sklearn.feature_extraction.text import CountVectorizer
        from hdbscan import HDBSCAN
        from umap import UMAP
        import nltk
        print("  ✅ Todas as bibliotecas OK")
        return True
    except ImportError as e:
        print(f"  ❌ Faltando biblioteca: {e}")
        print("  Instale com: pip install bertopic sentence-transformers umap-learn hdbscan")
        return False


def limpar_texto(texto):
    """Limpa e valida texto para processamento"""
    if pd.isna(texto):
        return None
    
    texto = str(texto).strip()
    texto = re.sub(r'http\S+|www\S+', '', texto)
    texto = re.sub(r'\S+@\S+', '', texto)
    texto = re.sub(r'[^\w\s\.,!?áàâãéèêíïóôõöúçñÁÀÂÃÉÈÊÍÏÓÔÕÖÚÇÑ-]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    
    if len(texto) < 10 or len(texto) > 5000:
        return None
    
    palavras = texto.split()
    if len(palavras) < 3:
        return None
    
    return texto


def criar_backup(arquivo):
    """Cria backup automático do arquivo"""
    if os.path.exists(arquivo):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup = f'{arquivo.rsplit(".", 1)[0]}_backup_{timestamp}.csv'
        shutil.copy2(arquivo, backup)
        print(f"  ✅ Backup: {backup}")
        return backup
    return None


# =============================================================================
# ETAPA 1: CARREGAR E LIMPAR DATASET
# =============================================================================

def carregar_e_limpar_dataset():
    """Carrega dataset e aplica limpeza de texto"""
    print("=" * 80)
    print("ETAPA 1: LIMPEZA DO DATASET")
    print("=" * 80)
    print()
    
    input_file = CONFIG["input_file"]
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Arquivo não encontrado: {input_file}")
    
    print(f"Carregando {input_file}...")
    df = pd.read_csv(input_file, sep=CONFIG["csv_separator"], low_memory=False)
    print(f"  ✅ {len(df):,} linhas carregadas")
    print()
    
    print("Limpando textos...")
    df['review_text_clean'] = df['review_text'].apply(limpar_texto)
    
    # Filtrar e preservar índice original
    df_clean = df[df['review_text_clean'].notna()].copy()
    df_clean['idx_original'] = df_clean.index.tolist()
    df_clean = df_clean.reset_index(drop=True)
    
    print(f"  ✅ Após limpeza: {len(df_clean):,} documentos")
    print(f"  ✅ Removidos: {len(df) - len(df_clean):,}")
    print()
    
    return df, df_clean


# =============================================================================
# ETAPA 2: GERAR EMBEDDINGS
# =============================================================================

def gerar_embeddings(docs):
    """Gera embeddings usando SentenceTransformer"""
    print("=" * 80)
    print("ETAPA 2: GERAÇÃO DE EMBEDDINGS")
    print("=" * 80)
    print()
    
    from sentence_transformers import SentenceTransformer
    
    print(f"Carregando modelo {CONFIG['embedding_model']}...")
    embedding_model = SentenceTransformer(CONFIG['embedding_model'])
    print()
    
    print(f"Gerando embeddings para {len(docs):,} documentos...")
    print("Tempo estimado: 30-45 minutos")
    print()
    
    inicio = datetime.now()
    
    batch_size = 1000
    embeddings_list = []
    
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        batch_embeddings = embedding_model.encode(
            batch,
            show_progress_bar=True,
            batch_size=32
        )
        embeddings_list.append(batch_embeddings)
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processados: {i+len(batch):,}/{len(docs):,}")
    
    embeddings = np.vstack(embeddings_list)
    
    tempo = (datetime.now() - inicio).total_seconds() / 60
    print(f"\n  ✅ Embeddings concluídos em {tempo:.1f} minutos")
    print(f"  ✅ Shape: {embeddings.shape}")
    print()
    
    del embedding_model, embeddings_list
    gc.collect()
    
    return embeddings


# =============================================================================
# ETAPA 3: TREINAMENTO BERTOPIC
# =============================================================================

def treinar_bertopic(docs, embeddings):
    """Treina modelo BERTopic"""
    print("=" * 80)
    print("ETAPA 3: TREINAMENTO BERTOPIC")
    print("=" * 80)
    print()
    
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer
    from hdbscan import HDBSCAN
    from umap import UMAP
    import nltk
    
    inicio = datetime.now()
    
    # Stopwords
    print("Configurando stopwords...")
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    stopwords_pt = stopwords.words('portuguese')
    stopwords_pt.extend(['muito', 'bom', 'boa', 'otimo', 'otima', 'excelente', 'lugar', 'pao'])
    print(f"  ✅ {len(stopwords_pt)} stopwords")
    print()
    
    # HDBSCAN
    print("Configurando HDBSCAN...")
    hdbscan_model = HDBSCAN(
        min_cluster_size=CONFIG["min_cluster_size"],
        min_samples=CONFIG["min_samples"],
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )
    print(f"  ✅ min_cluster_size: {CONFIG['min_cluster_size']}")
    print()
    
    # UMAP
    print("Configurando UMAP...")
    umap_model = UMAP(
        n_neighbors=CONFIG["umap_n_neighbors"],
        n_components=CONFIG["umap_n_components"],
        min_dist=0.0,
        metric='cosine',
        random_state=CONFIG["random_state"]
    )
    print(f"  ✅ n_components: {CONFIG['umap_n_components']}")
    print()
    
    # Vectorizer
    print("Configurando Vectorizer...")
    vectorizer_model = CountVectorizer(
        stop_words=stopwords_pt,
        ngram_range=(1, 2),
        min_df=10
    )
    print()
    
    # BERTopic
    print("Criando BERTopic...")
    topic_model = BERTopic(
        hdbscan_model=hdbscan_model,
        umap_model=umap_model,
        vectorizer_model=vectorizer_model,
        language='portuguese',
        calculate_probabilities=False,
        verbose=True
    )
    print()
    
    print("=" * 80)
    print("TREINANDO BERTOPIC...")
    print("Tempo estimado: 45-60 minutos")
    print("=" * 80)
    print()
    
    topics, _ = topic_model.fit_transform(docs, embeddings)
    
    tempo = (datetime.now() - inicio).total_seconds() / 60
    print(f"\n  ✅ Treinamento concluído em {tempo:.1f} minutos")
    
    # Estatísticas
    topic_info = topic_model.get_topic_info()
    n_topics = len(topic_info) - 1
    
    outlier_row = topic_info[topic_info['Topic'] == -1]
    outliers = outlier_row['Count'].values[0] if not outlier_row.empty else 0
    outlier_pct = (outliers / len(docs)) * 100
    
    print(f"\n  ✅ Tópicos descobertos: {n_topics}")
    print(f"  ✅ Outliers: {outliers:,} ({outlier_pct:.1f}%)")
    print()
    
    print("Top 20 tópicos:")
    for idx, row in topic_info[topic_info['Topic'] != -1].head(20).iterrows():
        print(f"  {row['Topic']:3d} | {row['Count']:6,} docs | {row['Name'][:60]}")
    print()
    
    return topic_model, topics, n_topics, outliers, outlier_pct


# =============================================================================
# ETAPA 4: GERAR JSON E ATUALIZAR DATASET
# =============================================================================

def gerar_saidas(df_original, df_clean, topic_model, topics, n_topics, outliers, outlier_pct):
    """Gera JSON para seleção e atualiza dataset"""
    print("=" * 80)
    print("ETAPA 4: GERAR SAÍDAS")
    print("=" * 80)
    print()
    
    df_clean['topic'] = topics
    
    # Criar JSON
    print("Criando JSON com todos os tópicos...")
    
    output = {
        "_INSTRUCOES": """
INSTRUÇÕES PARA SELEÇÃO DE TÓPICOS:

1. Analise cada tópico abaixo
2. Veja as palavras-chave (top_palavras)
3. Leia os exemplos de reviews (exemplos)
4. Se o tópico for relevante, mude: "selecionado": true
5. Adicione uma categoria: "categoria": "comida"
6. Para unir tópicos similares: "merge_para": <id_destino>

Categorias válidas: comida, atendimento, ambiente, preco, problemas
        """,
        "estatisticas": {
            "total_documentos": int(len(df_clean)),
            "total_topicos": int(n_topics),
            "outliers": int(outliers),
            "outliers_percentual": float(round(outlier_pct, 2))
        },
        "topicos": {}
    }
    
    for topic_id in sorted(df_clean['topic'].unique()):
        if topic_id == -1:
            continue
        
        topic_docs = df_clean[df_clean['topic'] == topic_id]
        count = len(topic_docs)
        pct = (count / len(df_clean)) * 100
        
        topic_words = topic_model.get_topic(topic_id)
        if topic_words:
            top_words = [str(word) for word, score in topic_words[:15]]
            topic_name = '_'.join(top_words[:5])
        else:
            top_words = []
            topic_name = f"topico_{topic_id}"
        
        n_samples = min(10, len(topic_docs))
        samples = topic_docs['review_text_clean'].sample(n_samples, random_state=42).tolist()
        
        output["topicos"][str(int(topic_id))] = {
            "id": int(topic_id),
            "nome": str(topic_name),
            "count": int(count),
            "percentual": float(round(pct, 2)),
            "top_palavras": top_words,
            "exemplos": [str(s) for s in samples],
            "selecionado": False,
            "categoria": None,
            "merge_para": None
        }
    
    # Salvar JSON
    with open(CONFIG["output_json"], 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  ✅ {CONFIG['output_json']} salvo")
    print()
    
    # Salvar checkpoints
    print("Salvando checkpoints...")
    topic_model.save("bertopic_model")
    print("  ✅ bertopic_model/ salvo")
    
    with open('topics.pkl', 'wb') as f:
        pickle.dump(topics, f)
    print("  ✅ topics.pkl salvo")
    
    df_clean.to_pickle('df_com_topicos.pkl')
    print("  ✅ df_com_topicos.pkl salvo")
    print()
    
    # Atualizar dataset original
    print("Atualizando dataset_full.csv...")
    
    criar_backup(CONFIG["output_csv"])
    
    df_original['topic'] = -2  # -2 = não processado
    df_original['categoria'] = None
    
    # Mapear usando índice original
    print("  Mapeando tópicos...")
    for idx_limpo in range(len(df_clean)):
        idx_original = df_clean.loc[idx_limpo, 'idx_original']
        topic_atribuido = df_clean.loc[idx_limpo, 'topic']
        df_original.loc[idx_original, 'topic'] = int(topic_atribuido)
    
    df_original.to_csv(CONFIG["output_csv"], index=False, sep=CONFIG["csv_separator"])
    print(f"  ✅ {CONFIG['output_csv']} atualizado")
    
    df_original.to_pickle('dataset_full.pkl')
    print("  ✅ dataset_full.pkl salvo")
    print()
    
    # Estatísticas finais
    total_com_topico = (df_original['topic'] >= 0).sum()
    total_outliers = (df_original['topic'] == -1).sum()
    total_nao_proc = (df_original['topic'] == -2).sum()
    
    print("=" * 80)
    print("📊 ESTATÍSTICAS FINAIS")
    print("=" * 80)
    print(f"  Total reviews:           {len(df_original):,}")
    print(f"  Com tópico (>=0):        {total_com_topico:,} ({100*total_com_topico/len(df_original):.1f}%)")
    print(f"  Outliers (-1):           {total_outliers:,} ({100*total_outliers/len(df_original):.1f}%)")
    print(f"  Não processados (-2):    {total_nao_proc:,} ({100*total_nao_proc/len(df_original):.1f}%)")
    print()
    
    return df_original


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("EXTRAÇÃO DE TÓPICOS COM BERTOPIC")
    print("=" * 80)
    print(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    if not verificar_bibliotecas():
        return
    
    # Etapa 1: Carregar e limpar
    df_original, df_clean = carregar_e_limpar_dataset()
    docs = df_clean['review_text_clean'].tolist()
    
    total_original = len(df_original)
    del df_original
    gc.collect()
    
    # Etapa 2: Embeddings
    embeddings = gerar_embeddings(docs)
    
    # Etapa 3: BERTopic
    topic_model, topics, n_topics, outliers, outlier_pct = treinar_bertopic(docs, embeddings)
    
    del embeddings
    gc.collect()
    
    # Etapa 4: Saídas
    df_original = pd.read_csv(CONFIG["input_file"], sep=CONFIG["csv_separator"], low_memory=False)
    gerar_saidas(df_original, df_clean, topic_model, topics, n_topics, outliers, outlier_pct)
    
    print("=" * 80)
    print("✅ PROCESSO CONCLUÍDO!")
    print("=" * 80)
    print()
    print("📁 ARQUIVOS GERADOS:")
    print(f"  1. {CONFIG['output_json']} - Tópicos para seleção manual")
    print(f"  2. {CONFIG['output_csv']} - Dataset atualizado com coluna 'topic'")
    print("  3. bertopic_model/ - Modelo salvo")
    print("  4. topics.pkl, df_com_topicos.pkl - Checkpoints")
    print()
    print("📌 PRÓXIMO PASSO:")
    print("  1. Edite topicos_para_selecao.json")
    print("  2. Marque tópicos relevantes: 'selecionado': true")
    print("  3. Adicione categorias: 'categoria': 'comida'")
    print("  4. Execute: 02_aplicar_merges_categorias.py")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Processo interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
