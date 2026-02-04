"""
================================================================================
SCRIPT 02: APLICAR MERGES E CATEGORIAS AO DATASET
================================================================================
Pipeline da Dissertação - Etapa 2

Entrada: 
  - dataset_full.csv (com coluna 'topic' do script 01)
  - topicos_para_selecao.json (editado manualmente com merges e categorias)

Saída:
  - dataset_full.csv (atualizado com colunas de merge e categoria)

Fluxo do pipeline:
  1. [Script 01] BERTopic → gera tópicos
  2. [Manual] Curadoria do JSON (merges, seleção, categorização)
  3. [Este script] Aplica merges e categorias ao dataset
  4. [Script 03] LLM → análise de sentimentos
================================================================================
"""

import pandas as pd
import json
import os
import shutil
from datetime import datetime

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

CONFIG = {
    "json_file": "topicos_para_selecao.json",
    "dataset_file": "dataset_full.csv",
    "csv_separator": "|",
    "output_file": "dataset_full.csv",  # sobrescreve o original
}

CATEGORIAS_VALIDAS = ['comida', 'atendimento', 'ambiente', 'preco', 'problemas']

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def aplicar_merge_recursivo(topic_id, merge_map, visitados=None):
    """
    Aplica merge recursivamente para casos encadeados.
    Exemplo: 7 → 4 → 2 (retorna 2)
    """
    if visitados is None:
        visitados = set()
    
    topic_id_str = str(topic_id)
    
    if topic_id_str in visitados:
        return int(topic_id)
    
    visitados.add(topic_id_str)
    
    if topic_id_str not in merge_map or merge_map[topic_id_str] is None:
        return int(topic_id)
    
    return aplicar_merge_recursivo(merge_map[topic_id_str], merge_map, visitados)


def validar_merges(topicos_dict, merge_map):
    """Valida se não há ciclos ou referências inválidas"""
    print("🔍 Validando merges...")
    
    erros = []
    
    for origem, destino in merge_map.items():
        if destino is None:
            continue
        
        # Verificar ciclos
        try:
            final = aplicar_merge_recursivo(int(origem), merge_map)
            if final == int(origem) and merge_map.get(origem) is not None:
                erros.append(f"Ciclo detectado: {origem} → {destino}")
        except RecursionError:
            erros.append(f"Ciclo infinito: {origem} → {destino}")
    
    if erros:
        print("❌ ERROS:")
        for erro in erros:
            print(f"   • {erro}")
        return False
    
    print("   ✅ OK")
    return True


def criar_backup(arquivo, separador):
    """Cria backup automático do arquivo"""
    if os.path.exists(arquivo):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup = f'{arquivo.rsplit(".", 1)[0]}_backup_{timestamp}.csv'
        df_backup = pd.read_csv(arquivo, sep=separador, low_memory=False)
        df_backup.to_csv(backup, sep=separador, index=False)
        print(f"   ✅ Backup: {backup}")
        return backup
    return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("APLICAR MERGES E CATEGORIAS")
    print("=" * 70)
    print(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # =========================================================================
    # ETAPA 1: CARREGAR JSON
    # =========================================================================
    print("=" * 70)
    print("ETAPA 1: CARREGAR JSON")
    print("=" * 70)
    print()
    
    json_file = CONFIG["json_file"]
    if not os.path.exists(json_file):
        print(f"❌ Arquivo não encontrado: {json_file}")
        return
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    topicos = data.get('topicos', data)
    print(f"   ✅ {len(topicos)} tópicos carregados")
    print()
    
    # =========================================================================
    # ETAPA 2: PROCESSAR MAPEAMENTOS
    # =========================================================================
    print("=" * 70)
    print("ETAPA 2: PROCESSAR MAPEAMENTOS")
    print("=" * 70)
    print()
    
    # Mapa de merges
    merge_map = {}
    for tid, info in topicos.items():
        merge_destino = info.get('merge_para') or info.get('merge')
        if merge_destino is not None:
            merge_map[tid] = merge_destino
    
    print(f"   Merges definidos: {len(merge_map)}")
    
    if merge_map and not validar_merges(topicos, merge_map):
        return
    
    # Mapeamentos de informações
    topic_nomes = {}
    topic_categorias = {}
    topic_selecionado = {}
    
    for tid_str, info in topicos.items():
        tid = int(tid_str)
        topic_nomes[tid] = info.get('nome_limpo') or info.get('nome', f'topico_{tid}')
        topic_categorias[tid] = info.get('categoria')
        topic_selecionado[tid] = info.get('selecionado', False)
    
    # Valores especiais
    topic_nomes[-1] = 'outlier'
    topic_nomes[-2] = 'nao_processado'
    topic_categorias[-1] = None
    topic_categorias[-2] = None
    topic_selecionado[-1] = False
    topic_selecionado[-2] = False
    
    # Contar por categoria
    categorias_count = {}
    for info in topicos.values():
        cat = info.get('categoria')
        if cat:
            categorias_count[cat] = categorias_count.get(cat, 0) + 1
    
    print(f"\n   📋 Tópicos por categoria:")
    for cat, count in sorted(categorias_count.items()):
        print(f"      {cat.upper():12s}: {count}")
    print()
    
    # =========================================================================
    # ETAPA 3: CARREGAR DATASET
    # =========================================================================
    print("=" * 70)
    print("ETAPA 3: CARREGAR DATASET")
    print("=" * 70)
    print()
    
    dataset_file = CONFIG["dataset_file"]
    if not os.path.exists(dataset_file):
        print(f"❌ Arquivo não encontrado: {dataset_file}")
        return
    
    df = pd.read_csv(dataset_file, sep=CONFIG["csv_separator"], low_memory=False)
    print(f"   ✅ {len(df):,} reviews carregadas")
    print()
    
    # =========================================================================
    # ETAPA 4: BACKUP
    # =========================================================================
    print("=" * 70)
    print("ETAPA 4: BACKUP")
    print("=" * 70)
    print()
    
    criar_backup(dataset_file, CONFIG["csv_separator"])
    print()
    
    # =========================================================================
    # ETAPA 5: APLICAR MERGES E CATEGORIAS
    # =========================================================================
    print("=" * 70)
    print("ETAPA 5: APLICAR MERGES E CATEGORIAS")
    print("=" * 70)
    print()
    
    # Garantir coluna topic_original
    if 'topic_original' not in df.columns:
        df['topic_original'] = df['topic'].copy()
        print("   Coluna 'topic_original' criada")
    
    # Aplicar merges → topic_final
    print("   Aplicando merges...")
    df['topic_final'] = df['topic_original'].apply(
        lambda x: aplicar_merge_recursivo(x, merge_map) if pd.notna(x) and x >= 0 else x
    )
    
    # Marcar merge aplicado
    df['merge_aplicado'] = (df['topic_original'] != df['topic_final']) & (df['topic_original'] >= 0)
    
    # Mapear nomes
    df['nome_topic_original'] = df['topic_original'].map(topic_nomes).fillna('desconhecido')
    df['nome_topic_final'] = df['topic_final'].map(topic_nomes).fillna('desconhecido')
    
    # Mapear categoria (do tópico)
    df['categoria'] = df['topic_final'].map(topic_categorias)
    
    # Mapear selecionado
    df['topic_selecionado'] = df['topic_final'].map(topic_selecionado).fillna(False)
    
    merges_aplicados = df['merge_aplicado'].sum()
    print(f"   ✅ {merges_aplicados:,} reviews com merge aplicado")
    print()
    
    # =========================================================================
    # ETAPA 6: SALVAR
    # =========================================================================
    print("=" * 70)
    print("ETAPA 6: SALVAR")
    print("=" * 70)
    print()
    
    df.to_csv(CONFIG["output_file"], sep=CONFIG["csv_separator"], index=False)
    print(f"   ✅ {CONFIG['output_file']} salvo")
    
    # Salvar também em pickle para carregar rápido
    pickle_file = CONFIG["output_file"].replace('.csv', '.pkl')
    df.to_pickle(pickle_file)
    print(f"   ✅ {pickle_file} salvo")
    print()
    
    # =========================================================================
    # ETAPA 7: ESTATÍSTICAS FINAIS
    # =========================================================================
    print("=" * 70)
    print("📊 ESTATÍSTICAS FINAIS")
    print("=" * 70)
    print()
    
    print(f"   Total reviews:           {len(df):,}")
    print(f"   Com tópico (>=0):        {(df['topic_final'] >= 0).sum():,}")
    print(f"   Outliers (-1):           {(df['topic_final'] == -1).sum():,}")
    print(f"   Não processados (-2):    {(df['topic_final'] == -2).sum():,}")
    print(f"   Merges aplicados:        {df['merge_aplicado'].sum():,}")
    print(f"   Tópicos selecionados:    {df['topic_selecionado'].sum():,}")
    print()
    
    print("   📋 Distribuição por categoria:")
    for cat in CATEGORIAS_VALIDAS:
        count = (df['categoria'] == cat).sum()
        if count > 0:
            print(f"      {cat:12s}: {count:>8,} ({100*count/len(df):>5.1f}%)")
    
    sem_cat = df['categoria'].isna().sum()
    print(f"      {'SEM CATEGORIA':12s}: {sem_cat:>8,} ({100*sem_cat/len(df):>5.1f}%)")
    print()
    
    # Verificar estabelecimentos
    if 'place_name' in df.columns:
        place_stats = df.groupby('place_name').agg(
            total=('topic_final', 'count'),
            nao_proc=('topic_final', lambda x: (x == -2).sum())
        )
        place_stats['pct'] = 100 * place_stats['nao_proc'] / place_stats['total']
        problematicos = len(place_stats[place_stats['pct'] == 100])
        
        if problematicos == 0:
            print("   ✅ Nenhum estabelecimento 100% não processado")
        else:
            print(f"   ⚠️ {problematicos} estabelecimentos 100% não processado")
    
    print()
    print("=" * 70)
    print("✅ CONCLUÍDO!")
    print("=" * 70)
    print()
    print("📁 COLUNAS ADICIONADAS:")
    print("   • topic_original - Tópico original do BERTopic")
    print("   • topic_final - Tópico após aplicar merges")
    print("   • merge_aplicado - Se houve merge (True/False)")
    print("   • nome_topic_original - Nome do tópico original")
    print("   • nome_topic_final - Nome do tópico final")
    print("   • categoria - Categoria gerencial do tópico")
    print("   • topic_selecionado - Se o tópico foi selecionado")
    print()
    print("📌 PRÓXIMO PASSO:")
    print("   Execute: 03_analise_sentimentos_llm.py")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
