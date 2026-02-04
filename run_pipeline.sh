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
echo "  4) Executar todas as etapas"
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
        echo "⚠️ Executando todas as etapas..."
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
        echo "✅ Pipeline concluído!"
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
