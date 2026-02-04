#!/bin/bash
# =============================================================================
# EXECUTAR PIPELINE DE ANÁLISE DE SENTIMENTOS
# =============================================================================

set -e

echo "========================================"
echo "PIPELINE DE ANÁLISE DE SENTIMENTOS"
echo "========================================"
echo ""

# Verificar se dataset existe
if [ ! -f "data/dataset_full.csv" ]; then
    echo "❌ ERRO: data/dataset_full.csv não encontrado!"
    echo ""
    echo "Coloque seu dataset na pasta data/ antes de executar."
    exit 1
fi

# Menu
echo "Selecione a etapa a executar:"
echo ""
echo "  1) Etapa 1: Extração de tópicos (BERTopic) [~2-3h]"
echo "  2) Etapa 2: Aplicar merges e categorias [~1min]"
echo "  3) Etapa 3: Análise de sentimentos (LLM) [~10-20h]"
echo "  4) Etapa 4: Análise de categorias [~2-4h]"
echo "  5) Etapa 5: Gerar gráficos de análises [~1min]"
echo "  6) Etapa 6: Análise de posicionamento digital [~1min]"
echo "  7) Etapa 7: Gerar TODAS as figuras da dissertação [~5-30min]"
echo "  8) Executar etapas 4-7 (pós-processamento)"
echo "  9) Executar pipeline completo"
echo "  0) Sair"
echo ""
read -p "Opção: " opcao

case $opcao in
    1)
        echo ""
        echo "Executando Etapa 1: BERTopic..."
        python scripts/01_extrair_topicos_bertopic.py
        echo ""
        echo "✅ Etapa 1 concluída!"
        echo "📌 Próximo passo: Edite topicos_para_selecao.json"
        ;;
    2)
        echo ""
        echo "Executando Etapa 2: Merges e categorias..."
        python scripts/02_aplicar_merges_categorias.py
        echo ""
        echo "✅ Etapa 2 concluída!"
        ;;
    3)
        echo ""
        echo "Verificando Ollama..."
        if ! curl -s http://localhost:11434/api/tags > /dev/null; then
            echo "❌ Ollama não está rodando!"
            echo "Execute: ollama serve"
            exit 1
        fi
        echo "✅ Ollama OK"
        echo ""
        echo "Executando Etapa 3: Análise de sentimentos..."
        python scripts/03_analise_sentimentos_llm.py
        echo ""
        echo "✅ Etapa 3 concluída!"
        ;;
    4)
        echo ""
        echo "Executando Etapa 4: Análise de categorias..."
        python scripts/04_analises_categorias.py
        echo ""
        echo "✅ Etapa 4 concluída!"
        ;;
    5)
        echo ""
        echo "Executando Etapa 5: Gerar gráficos..."
        python scripts/05_gerar_graficos_analises.py
        echo ""
        echo "✅ Etapa 5 concluída!"
        ;;
    6)
        echo ""
        echo "Executando Etapa 6: Posicionamento digital..."
        python scripts/06_analise_posicionamento_digital.py
        echo ""
        echo "✅ Etapa 6 concluída!"
        ;;
    7)
        echo ""
        echo "Executando Etapa 7: Gerar TODAS as figuras da dissertação..."
        python scripts/07_gerar_figuras_dissertacao.py
        echo ""
        echo "✅ Etapa 7 concluída!"
        echo "📊 Figuras salvas em outputs/"
        ;;
    8)
        echo ""
        echo "Executando etapas 4-7 (pós-processamento)..."
        echo ""
        echo "=== Etapa 4: Análise de categorias ==="
        python scripts/04_analises_categorias.py
        echo ""
        echo "=== Etapa 5: Gráficos de análises ==="
        python scripts/05_gerar_graficos_analises.py
        echo ""
        echo "=== Etapa 6: Posicionamento digital ==="
        python scripts/06_analise_posicionamento_digital.py
        echo ""
        echo "=== Etapa 7: Figuras da dissertação ==="
        python scripts/07_gerar_figuras_dissertacao.py
        echo ""
        echo "✅ Pós-processamento concluído!"
        echo "📊 Todas as figuras salvas em outputs/"
        ;;
    9)
        echo ""
        echo "⚠️ Executando pipeline completo..."
        echo "Isso pode levar 15-25 horas!"
        read -p "Continuar? (s/n): " confirma
        if [ "$confirma" != "s" ]; then
            echo "Cancelado."
            exit 0
        fi
        
        echo ""
        echo "=== Etapa 1: BERTopic ==="
        python scripts/01_extrair_topicos_bertopic.py
        
        echo ""
        echo "⚠️ ATENÇÃO: Edite topicos_para_selecao.json antes de continuar!"
        read -p "Pressione ENTER quando terminar a edição..."
        
        echo ""
        echo "=== Etapa 2: Merges e categorias ==="
        python scripts/02_aplicar_merges_categorias.py
        
        echo ""
        echo "=== Etapa 3: Análise de sentimentos ==="
        python scripts/03_analise_sentimentos_llm.py
        
        echo ""
        echo "=== Etapa 4: Análise de categorias ==="
        python scripts/04_analises_categorias.py
        
        echo ""
        echo "=== Etapa 5: Gráficos de análises ==="
        python scripts/05_gerar_graficos_analises.py
        
        echo ""
        echo "=== Etapa 6: Posicionamento digital ==="
        python scripts/06_analise_posicionamento_digital.py
        
        echo ""
        echo "=== Etapa 7: Figuras da dissertação ==="
        python scripts/07_gerar_figuras_dissertacao.py
        
        echo ""
        echo "✅ Pipeline completo concluído!"
        echo "📊 Todas as figuras salvas em outputs/"
        ;;
    0)
        echo "Saindo..."
        exit 0
        ;;
    *)
        echo "Opção inválida!"
        exit 1
        ;;
esac
