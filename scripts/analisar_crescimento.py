#!/bin/bash
cd /root/confereai

python3 << 'EOF'
import pandas as pd
import numpy as np
import os

df = pd.read_csv('data/historico_pagamento_15.csv', delimiter=';')

def calc_cagr(valor_inicial, valor_final, anos):
    """Taxa de crescimento anual composta."""
    if valor_inicial <= 0 or anos <= 0:
        return np.nan
    return ((valor_final / valor_inicial) ** (1/anos) - 1) * 100

resultados = []

print("=" * 90)
print("  ANÁLISE DE CRESCIMENTO POR RUBRICA — TODOS OS CARGOS")
print("  CAGR = Compound Annual Growth Rate (taxa anual composta de crescimento)")
print("=" * 90)
print()

for cargo in sorted(df['cod_cargo'].unique()):
    df_cargo = df[df['cod_cargo'] == cargo]
    rubricas = df_cargo['isn_rubrica'].value_counts()
    
    # Pega anos disponíveis
    anos_disponiveis = sorted(df_cargo['num_ano'].unique())
    if len(anos_disponiveis) < 5:
        continue  # pula cargos com poucos anos
    
    ano_inicio = anos_disponiveis[0]
    ano_fim = anos_disponiveis[-1]
    anos_total = ano_fim - ano_inicio
    
    for rubrica in rubricas.head(5).index:  # top 5 rubricas por frequência
        df_rub = df_cargo[df_cargo['isn_rubrica'] == rubrica]
        
        media_inicio = df_rub[df_rub['num_ano'] == ano_inicio]['vlr_calculado'].mean()
        media_fim = df_rub[df_rub['num_ano'] == ano_fim]['vlr_calculado'].mean()
        
        if media_inicio > 0 and media_fim > 0:
            cagr = calc_cagr(media_inicio, media_fim, anos_total)
            cresc_total = (media_fim / media_inicio - 1) * 100
            
            resultados.append({
                'cargo': cargo,
                'rubrica': rubrica,
                'ano_inicio': ano_inicio,
                'ano_fim': ano_fim,
                'anos': anos_total,
                'valor_inicio': media_inicio,
                'valor_fim': media_fim,
                'cagr': cagr,
                'cresc_total': cresc_total
            })

df_result = pd.DataFrame(resultados)

# Ordena por CAGR
df_result = df_result.sort_values('cagr', ascending=False)

print(f"{'Cargo':<8} {'Rub':>5} {'Ini':>4} {'Fim':>4} {'Anos':>4}  {'Valor Ini':>12} {'Valor Fim':>12}  {'CAGR%':>7}  {'Cresc%':>8}")
print("-" * 90)

for _, r in df_result.head(50).iterrows():
    print(f"{r['cargo']:<8} {r['rubrica']:>5} {int(r['ano_inicio']):>4} {int(r['ano_fim']):>4} {int(r['anos']):>4}  "
          f"R$ {r['valor_inicio']:>10,.2f}  R$ {r['valor_fim']:>10,.2f}  {r['cagr']:>6.2f}%  {r['cresc_total']:>7.1f}%")

print()
print("=" * 90)
print("  TOP 10 — Maiores CAGRs (rubricas que mais cresceram)")
print("=" * 90)
for i, (_, r) in enumerate(df_result.head(10).iterrows(), 1):
    print(f"  {i}. Cargo {r['cargo']} / Rubrica {r['rubrica']}: "
          f"R$ {r['valor_inicio']:,.2f} → R$ {r['valor_fim']:,.2f} "
          f"| CAGR: {r['cagr']:.2f}%/ano | Total: {r['cresc_total']:.0f}%")

print()
print("=" * 90)
print("  TOP 10 — Menores CAGRs (rubricas mais estáveis)")
print("=" * 90)
for i, (_, r) in enumerate(df_result.sort_values('cagr').head(10).iterrows(), 1):
    print(f"  {i}. Cargo {r['cargo']} / Rubrica {r['rubrica']}: "
          f"R$ {r['valor_inicio']:,.2f} → R$ {r['valor_fim']:,.2f} "
          f"| CAGR: {r['cagr']:.2f}%/ano | Total: {r['cresc_total']:.0f}%")

# Estatísticas gerais
print()
print("=" * 90)
print("  ESTATÍSTICAS GERAIS DE CAGR")
print("=" * 90)
print(f"  Total de observações: {len(df_result)}")
print(f"  CAGR médio: {df_result['cagr'].mean():.2f}%")
print(f"  CAGR mediano: {df_result['cagr'].median():.2f}%")
print(f"  Maior CAGR: {df_result['cagr'].max():.2f}% ({df_result.iloc[0]['cargo']} / rub {df_result.iloc[0]['rubrica']})")
print(f"  Menor CAGR: {df_result['cagr'].min():.2f}%")

# Histograma por faixa de CAGR
print()
print("  Distribuição de CAGR:")
faixas = [(0, 3), (3, 6), (6, 10), (10, 15), (15, 100)]
for ini, fim in faixas:
    qtd = ((df_result['cagr'] >= ini) & (df_result['cagr'] < fim)).sum()
    bar = '█' * (qtd // 2)
    print(f"  {ini:>3}-{fim:>3}%: {qtd:>4} rubricas {bar}")

# Salvar CSV
df_result.to_csv('data/crescimento_rubricas.csv', index=False, sep=';')
print()
print("✅ Salvo em data/crescimento_rubricas.csv")

EOF
