"""
================================================================================
SCRIPT 03: ANÁLISE DE SENTIMENTOS COM LLM
================================================================================
Pipeline da Dissertação - Etapa 3

Entrada: dataset_full.csv (com colunas de tópico e categoria)
Saída: dataset_com_sentimentos.xlsx

Requisitos:
  - Ollama instalado e rodando (ollama serve)
  - Modelo llama3.1:8b baixado (ollama pull llama3.1:8b)

Fluxo do pipeline:
  1. [Script 01] BERTopic → gera tópicos
  2. [Script 02] Aplica merges e categorias
  3. [Este script] LLM → análise de sentimentos por review
================================================================================
"""

import pandas as pd
import json
import os
import requests
from datetime import datetime
from tqdm import tqdm
import time
import socket

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

CONFIG = {
    # Arquivos
    "input_file": "dataset_full.csv",
    "csv_separator": "|",
    "output_excel": "dataset_com_sentimentos.xlsx",
    "output_pickle": "dataset_com_sentimentos.pkl",
    "checkpoint_file": "checkpoint_llm.json",
    "log_file": "log_analise_llm.txt",
    
    # Ollama
    "ollama_url": "http://localhost:11434/api/generate",
    "ollama_model": "llama3.1:8b",
    
    # Processamento
    "checkpoint_interval": 1000,  # Salvar a cada N reviews
    "excel_interval": 10000,      # Salvar Excel a cada N reviews
    "max_reviews": None,          # None = processar todos
    "max_retries": 2,
    "timeout": 90,
}

CATEGORIAS = ['comida', 'atendimento', 'ambiente', 'preco', 'problemas']

PROMPT_TEMPLATE = """Você é um especialista em análise de sentimentos de reviews de padarias.

Analise o review abaixo e identifique:
1. Quais categorias são mencionadas
2. Para cada categoria, identifique a sentença específica que justifica
3. Para cada categoria, classifique o sentimento como: positivo, negativo ou neutro

CATEGORIAS POSSÍVEIS:
- comida: qualidade de pães, doces, salgados, café, pizza, sabor, frescor
- atendimento: funcionários, garçons, rapidez, educação, cordialidade
- ambiente: limpeza, localização, espaço, decoração, conforto
- preco: valor, caro, barato, justo, absurdo, vale a pena, custo-benefício
- problemas: reclamações gerais, declínio de qualidade, necessidade de melhoria

REGRAS IMPORTANTES:
1. NUNCA deixe "sentimento" ou "evidencia" vazios
2. Se não conseguir identificar a evidência, NÃO inclua essa categoria
3. A evidência deve ser uma sentença específica do review

REVIEW:
"{review}"

RESPONDA APENAS COM UM JSON no formato abaixo:
{{
  "categorias": [
    {{
      "categoria": "comida",
      "evidencia": "sentença específica do review",
      "sentimento": "positivo"
    }}
  ]
}}

Se não houver nenhuma categoria, retorne:
{{"categorias": []}}

JSON:"""

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def log_message(mensagem, nivel="INFO"):
    """Registra mensagem no console e arquivo de log"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    hostname = socket.gethostname()
    msg_completa = f"[{timestamp}] [{hostname}] [{nivel}] {mensagem}"
    print(msg_completa)
    
    with open(CONFIG["log_file"], 'a', encoding='utf-8') as f:
        f.write(msg_completa + '\n')


def testar_ollama():
    """Testa conexão com Ollama"""
    log_message("Testando conexão com Ollama...", "INFO")
    try:
        response = requests.post(
            CONFIG["ollama_url"],
            json={"model": CONFIG["ollama_model"], "prompt": "OK", "stream": False},
            timeout=30
        )
        if response.status_code == 200:
            log_message(f"Ollama conectado! Modelo: {CONFIG['ollama_model']}", "SUCCESS")
            return True
        return False
    except Exception as e:
        log_message(f"Não foi possível conectar ao Ollama: {e}", "ERROR")
        return False


def chamar_ollama(prompt):
    """Chama API do Ollama com retries"""
    for tentativa in range(CONFIG["max_retries"]):
        try:
            response = requests.post(
                CONFIG["ollama_url"],
                json={
                    "model": CONFIG["ollama_model"],
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 500
                    }
                },
                timeout=CONFIG["timeout"]
            )
            if response.status_code == 200:
                return response.json().get('response', '').strip()
            time.sleep(2)
        except requests.Timeout:
            log_message(f"Timeout na tentativa {tentativa + 1}/{CONFIG['max_retries']}", "WARNING")
            if tentativa < CONFIG["max_retries"] - 1:
                time.sleep(5)
        except Exception as e:
            log_message(f"Erro na tentativa {tentativa + 1}: {str(e)}", "WARNING")
            if tentativa < CONFIG["max_retries"] - 1:
                time.sleep(5)
    
    return None


def extrair_json(texto):
    """Extrai JSON de uma string"""
    try:
        inicio = texto.find('{')
        fim = texto.rfind('}') + 1
        if inicio != -1 and fim > inicio:
            return json.loads(texto[inicio:fim])
    except:
        pass
    return None


def analisar_review(review_text):
    """Analisa um review usando LLM"""
    if pd.isna(review_text):
        return None
    
    review_clean = str(review_text).strip()
    if len(review_clean) < 10:
        return None
    
    prompt = PROMPT_TEMPLATE.format(review=review_clean[:500])
    resposta = chamar_ollama(prompt)
    
    if resposta is None:
        return None
    
    return extrair_json(resposta)


def processar_resultado_llm(resultado):
    """Processa resultado do LLM em formato padronizado"""
    analise_json = {"analises": []}
    
    if resultado is None or 'categorias' not in resultado:
        return {
            'llm_analise_json': json.dumps(analise_json, ensure_ascii=False),
            'llm_num_categorias': 0
        }
    
    for item in resultado.get('categorias', []):
        cat = item.get('categoria', '')
        evi = item.get('evidencia', '')
        sent = item.get('sentimento', '')
        
        # Normalizar listas
        if isinstance(cat, list):
            cat = cat[0] if cat else ''
        if isinstance(evi, list):
            evi = evi[0] if evi else ''
        if isinstance(sent, list):
            sent = sent[0] if sent else ''
        
        cat = str(cat).strip().lower()
        evi = str(evi).strip()
        sent = str(sent).strip().lower()
        
        # Validar
        if not cat or not evi or not sent:
            continue
        if cat not in CATEGORIAS:
            continue
        if sent not in ['positivo', 'negativo', 'neutro']:
            continue
        
        analise_json['analises'].append({
            'categoria': cat,
            'sentimento': sent,
            'evidencia': evi
        })
    
    return {
        'llm_analise_json': json.dumps(analise_json, ensure_ascii=False),
        'llm_num_categorias': len(analise_json['analises'])
    }


def salvar_checkpoint(indice_atual, info_adicional=None):
    """Salva checkpoint de progresso"""
    checkpoint = {
        'ultimo_indice': indice_atual,
        'timestamp': datetime.now().isoformat(),
        'hostname': socket.gethostname(),
        'info_adicional': info_adicional or {}
    }
    with open(CONFIG["checkpoint_file"], 'w') as f:
        json.dump(checkpoint, f, indent=2)


def carregar_checkpoint():
    """Carrega checkpoint se existir"""
    if os.path.exists(CONFIG["checkpoint_file"]):
        try:
            with open(CONFIG["checkpoint_file"], 'r') as f:
                return json.load(f).get('ultimo_indice', 0)
        except:
            pass
    return 0


def limpar_checkpoint():
    """Remove arquivo de checkpoint"""
    if os.path.exists(CONFIG["checkpoint_file"]):
        os.remove(CONFIG["checkpoint_file"])


# =============================================================================
# MAIN
# =============================================================================

def main():
    log_message("=" * 70, "INFO")
    log_message("ANÁLISE DE SENTIMENTOS COM LLM", "INFO")
    log_message("=" * 70, "INFO")
    
    # Testar Ollama
    if not testar_ollama():
        log_message("Abortando: Ollama não está acessível", "ERROR")
        log_message("Execute: ollama serve", "INFO")
        return
    
    # Carregar dataset
    input_file = CONFIG["input_file"]
    log_message(f"Carregando dataset: {input_file}", "INFO")
    
    if not os.path.exists(input_file):
        log_message(f"ERRO: {input_file} não encontrado!", "ERROR")
        return
    
    df = pd.read_csv(input_file, sep=CONFIG["csv_separator"], low_memory=False)
    log_message(f"Dataset carregado: {len(df):,} reviews", "SUCCESS")
    
    # Verificar se há progresso anterior
    output_pickle = CONFIG["output_pickle"]
    if os.path.exists(output_pickle):
        log_message("Arquivo pickle existente encontrado", "INFO")
        df_existente = pd.read_pickle(output_pickle)
        inicio = carregar_checkpoint()
        log_message(f"Continuando do índice {inicio}", "INFO")
        if 'llm_analise_json' in df_existente.columns:
            df = df_existente.copy()
    else:
        inicio = 0
    
    # Limitar se configurado
    if CONFIG["max_reviews"] is not None:
        df = df.head(CONFIG["max_reviews"])
    
    # Inicializar colunas
    if 'llm_analise_json' not in df.columns:
        df['llm_analise_json'] = None
    if 'llm_num_categorias' not in df.columns:
        df['llm_num_categorias'] = 0
    
    log_message(f"Total a processar: {len(df) - inicio:,}", "INFO")
    inicio_tempo = datetime.now()
    
    # Processar reviews
    with tqdm(total=len(df) - inicio, desc="Analisando", unit="review") as pbar:
        for idx in range(inicio, len(df)):
            review = df.loc[idx, 'review_text']
            resultado = analisar_review(review)
            dados = processar_resultado_llm(resultado)
            
            for col, valor in dados.items():
                df.loc[idx, col] = valor
            
            pbar.update(1)
            
            # Checkpoint
            if (idx + 1) % CONFIG["checkpoint_interval"] == 0:
                df.to_pickle(output_pickle)
                
                # Excel a cada N
                if (idx + 1) % CONFIG["excel_interval"] == 0:
                    log_message("Salvando Excel...", "INFO")
                    df.to_excel(CONFIG["output_excel"], index=False, engine='openpyxl')
                
                tempo_decorrido = (datetime.now() - inicio_tempo).total_seconds()
                processados = idx + 1 - inicio
                velocidade = processados / tempo_decorrido if tempo_decorrido > 0 else 0
                
                info_checkpoint = {
                    'velocidade': round(velocidade, 2),
                    'tempo_decorrido_min': round(tempo_decorrido / 60, 2),
                    'reviews_processados': processados
                }
                salvar_checkpoint(idx + 1, info_checkpoint)
                log_message(f"Checkpoint {idx + 1}/{len(df)} | {velocidade:.2f} reviews/s", "INFO")
    
    # Salvar resultado final
    log_message("Salvando resultado final...", "INFO")
    df.to_excel(CONFIG["output_excel"], index=False, engine='openpyxl')
    df.to_pickle(output_pickle)
    
    limpar_checkpoint()
    
    tempo_total = (datetime.now() - inicio_tempo).total_seconds()
    log_message(f"Tempo total: {tempo_total/60:.1f} minutos", "INFO")
    
    # Estatísticas
    log_message("=" * 70, "INFO")
    log_message("📊 ESTATÍSTICAS", "INFO")
    log_message("=" * 70, "INFO")
    
    total_analisados = (df['llm_num_categorias'] > 0).sum()
    log_message(f"Reviews analisados: {total_analisados:,}", "INFO")
    log_message(f"Categorias identificadas: {df['llm_num_categorias'].sum():,}", "INFO")
    
    log_message("=" * 70, "INFO")
    log_message("✅ PROCESSO CONCLUÍDO!", "SUCCESS")
    log_message("=" * 70, "INFO")
    
    return df


if __name__ == "__main__":
    log_message("=" * 70, "INFO")
    log_message("INÍCIO DA EXECUÇÃO", "INFO")
    log_message("=" * 70, "INFO")
    
    try:
        df_resultado = main()
        log_message("Processo concluído com sucesso!", "SUCCESS")
    except KeyboardInterrupt:
        log_message("Processo interrompido pelo usuário", "WARNING")
    except Exception as e:
        log_message(f"ERRO CRÍTICO: {e}", "ERROR")
        import traceback
        log_message(traceback.format_exc(), "ERROR")
