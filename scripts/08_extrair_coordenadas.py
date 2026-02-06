#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_extrair_coordenadas.py
=========================
Extrai latitude e longitude das padarias a partir do review_link do Google Maps.

Este script acessa cada review_link usando Playwright (navegador headless) e 
extrai as coordenadas da URL final após o redirecionamento do Google Maps.

Requisitos:
    pip install pandas openpyxl playwright tqdm
    playwright install chromium

Uso:
    python 08_extrair_coordenadas.py --input dataset.xlsx --output dataset_com_coords.xlsx

Para Google Colab:
    !pip install playwright
    !playwright install chromium
    # Depois execute as células do notebook

Autor: Filipe Silva
Data: 2025
"""

import pandas as pd
import re
import time
import argparse
import asyncio

# Tqdm - funciona tanto em notebook quanto em CLI
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Playwright
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("⚠️ Playwright não instalado. Execute:")
    print("   pip install playwright")
    print("   playwright install chromium")


# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

DEFAULT_DELAY = 2.0      # Segundos entre requisições
DEFAULT_TIMEOUT = 20     # Timeout em segundos


# =============================================================================
# FUNÇÕES
# =============================================================================

def extrair_coords_da_url(url: str) -> tuple:
    """
    Extrai latitude e longitude de uma URL do Google Maps.
    
    Args:
        url: URL do Google Maps
        
    Returns:
        Tupla (lat, lng) ou (None, None) se não encontrar
    """
    patterns = [
        r'@(-?\d+\.\d+),(-?\d+\.\d+)',      # Formato: @-23.5399,-46.5652
        r'!3d(-?\d+\.\d+)!4d(-?\d+\.\d+)',  # Formato: !3d-23.5399!4d-46.5652
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return float(match.group(1)), float(match.group(2))
    
    return None, None


async def extrair_coordenadas_async(
    input_file: str,
    output_file: str,
    delay: float = DEFAULT_DELAY,
    timeout: int = DEFAULT_TIMEOUT
) -> pd.DataFrame:
    """
    Extrai coordenadas de cada padaria única usando Playwright async.
    
    Args:
        input_file: Arquivo Excel de entrada
        output_file: Arquivo Excel de saída
        delay: Segundos entre requisições
        timeout: Timeout em segundos
        
    Returns:
        DataFrame com coordenadas adicionadas
    """
    
    if not PLAYWRIGHT_AVAILABLE:
        raise ImportError("Playwright não está instalado")
    
    # Carregar dados
    print(f"\n📂 Carregando {input_file}...")
    df = pd.read_excel(input_file)
    print(f"   {len(df):,} reviews")
    print(f"   {df['place_id'].nunique()} padarias únicas")
    
    # Pegar um review_link por place_id
    padarias = df.groupby('place_id').agg({
        'place_name': 'first',
        'review_link': 'first'
    }).reset_index()
    
    print(f"\n🚀 Iniciando extração de {len(padarias)} padarias...")
    print(f"   Método: Playwright (Chromium headless async)")
    print(f"   Delay: {delay}s | Timeout: {timeout}s\n")
    
    coordenadas = {}
    sucessos = 0
    falhas = 0
    falhas_lista = []
    
    async with async_playwright() as p:
        # Iniciar navegador
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox"]
        )
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        page = await context.new_page()
        print("✅ Chromium iniciado\n")
        
        for idx, row in tqdm(padarias.iterrows(), total=len(padarias), desc="Extraindo"):
            place_id = row['place_id']
            place_name = str(row['place_name'])
            review_link = row['review_link']
            
            lat, lng = None, None
            
            try:
                await page.goto(review_link, timeout=timeout * 1000)
                await asyncio.sleep(delay)
                
                # Aguardar até encontrar coordenadas na URL
                for _ in range(timeout):
                    lat, lng = extrair_coords_da_url(page.url)
                    if lat and lng:
                        break
                    await asyncio.sleep(1)
                    
            except Exception as e:
                pass
            
            if lat and lng:
                coordenadas[place_id] = {'lat': lat, 'lng': lng}
                sucessos += 1
            else:
                falhas += 1
                falhas_lista.append(place_name)
                tqdm.write(f"   ⚠️ Falha: {place_name[:50]}")
        
        await browser.close()
    
    # Resultado
    print(f"\n📊 Resultado:")
    print(f"   ✅ Sucesso: {sucessos}")
    print(f"   ❌ Falha: {falhas}")
    if sucessos + falhas > 0:
        print(f"   Taxa: {sucessos/(sucessos+falhas)*100:.1f}%")
    
    # Adicionar coordenadas ao dataset
    df['lat'] = df['place_id'].map(lambda x: coordenadas.get(x, {}).get('lat'))
    df['lng'] = df['place_id'].map(lambda x: coordenadas.get(x, {}).get('lng'))
    
    # Salvar
    df.to_excel(output_file, index=False)
    print(f"\n💾 Salvo: {output_file}")
    
    # Amostra
    print(f"\n📍 Amostra de coordenadas extraídas:")
    amostra = df[['place_name', 'lat', 'lng']].drop_duplicates().dropna().head(5)
    for _, r in amostra.iterrows():
        print(f"   {r['place_name'][:40]}: {r['lat']}, {r['lng']}")
    
    # Salvar lista de falhas
    if falhas_lista:
        falhas_file = output_file.replace('.xlsx', '_falhas.txt')
        with open(falhas_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(falhas_lista))
        print(f"\n⚠️ Lista de falhas salva em: {falhas_file}")
    
    return df


def extrair_coordenadas(
    input_file: str,
    output_file: str = None,
    delay: float = DEFAULT_DELAY,
    timeout: int = DEFAULT_TIMEOUT
) -> pd.DataFrame:
    """
    Wrapper síncrono para a função async.
    
    Args:
        input_file: Arquivo Excel de entrada
        output_file: Arquivo Excel de saída (opcional)
        delay: Segundos entre requisições
        timeout: Timeout em segundos
        
    Returns:
        DataFrame com coordenadas adicionadas
    """
    if output_file is None:
        output_file = input_file.replace('.xlsx', '_com_coords.xlsx')
    
    return asyncio.run(extrair_coordenadas_async(
        input_file, output_file, delay, timeout
    ))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extrai coordenadas das padarias via review_link do Google Maps'
    )
    parser.add_argument(
        '--input', '-i', 
        required=True, 
        help='Arquivo Excel de entrada'
    )
    parser.add_argument(
        '--output', '-o', 
        default=None, 
        help='Arquivo Excel de saída (default: input_com_coords.xlsx)'
    )
    parser.add_argument(
        '--delay', '-d', 
        type=float, 
        default=DEFAULT_DELAY, 
        help=f'Delay em segundos entre requisições (default: {DEFAULT_DELAY})'
    )
    parser.add_argument(
        '--timeout', '-t', 
        type=int, 
        default=DEFAULT_TIMEOUT, 
        help=f'Timeout em segundos (default: {DEFAULT_TIMEOUT})'
    )
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = args.input.replace('.xlsx', '_com_coords.xlsx')
    
    extrair_coordenadas(args.input, args.output, args.delay, args.timeout)
