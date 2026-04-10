"""
Baseline de Detecção de Anomalias — VERSÃO YoY (Year-over-Year) — OTIMIZADO
Passos:
  1. Normalizar por YoY: cada mês / mesmo mês do ano anterior
  2. Isolation Forest nos valores YoY-normalizados
  3. Por cargo, divisão temporal 80/20
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import os
import argparse
import json
from datetime import datetime


def run_cargo(df_cargo, cargo, output_dir, contamination):
    print(f"\n{'='*60}")
    print(f"  Cargo: {cargo} | {len(df_cargo):,} registros")

    # Pivot: rubricas em colunas
    df_pivot = df_cargo.pivot_table(
        index=['isn_vinculo', 'num_ano', 'num_mes'],
        columns='isn_rubrica',
        values='vlr_calculado',
        aggfunc='mean'
    ).fillna(0)

    # Renomeia colunas de rubrica para string
    df_pivot.columns = [str(c) for c in df_pivot.columns]
    df_pivot = df_pivot.reset_index()

    cols_context = ['isn_vinculo', 'num_ano', 'num_mes']
    cols_rub = [c for c in df_pivot.columns if c not in cols_context]
    df_pivot = df_pivot[cols_context + cols_rub]
    df_pivot = df_pivot.sort_values(['isn_vinculo', 'num_ano', 'num_mes']).reset_index(drop=True)
    n_rows = len(df_pivot)

    if n_rows < 24:
        print(f"  Pulando — poucos dados (< 2 anos)")
        return None

    print(f"  Pivot: {n_rows} × {len(cols_rub)} rubricas")

    # ── YoY vetorizado ──
    # Para cada rubrica: shift por (isn_vinculo, num_mes) em 12 linhas (1 ano)
    df_yoy = df_pivot[cols_context].copy()
    cols_yoy = []

    for rub in cols_rub:
        col_yoy = rub + '_yoy'
        cols_yoy.append(col_yoy)
        # Ordena por vínculo + mês
        tmp = df_pivot[['isn_vinculo', 'num_mes', rub]].copy()
        tmp = tmp.sort_values(['isn_vinculo', 'num_mes']).reset_index(drop=True)
        # Shift de 12 linhas = mesmo vínculo/mês 1 ano atrás
        tmp[col_yoy] = tmp.groupby(['isn_vinculo', 'num_mes'])[rub].shift(12)
        df_yoy[col_yoy] = tmp[col_yoy].values

    # YoY ratio = valor_atual / valor_ano_anterior
    for rub, yoy_col in zip(cols_rub, cols_yoy):
        df_yoy[yoy_col] = df_pivot[rub] / df_yoy[yoy_col]
        df_yoy[yoy_col] = df_yoy[yoy_col].replace([np.inf, -np.inf], np.nan).fillna(1.0)

    n_yoy_validos = df_yoy[cols_yoy].notna().sum().sum()
    n_total = n_rows * len(cols_yoy)
    pct_validos = n_yoy_validos / n_total * 100
    print(f"  YoY válidos: {n_yoy_validos:,}/{n_total:,} ({pct_validos:.1f}%)")

    # Divisão temporal 80/20
    split_idx = int(n_rows * 0.8)
    treino_pos = list(range(split_idx))
    teste_pos = list(range(split_idx, n_rows))

    X_treino = df_yoy.iloc[:split_idx][cols_yoy].values
    X_teste = df_yoy.iloc[split_idx:][cols_yoy].values

    df_treino = df_yoy.iloc[:split_idx]
    df_teste = df_yoy.iloc[split_idx:]
    print(f"  Treino: {len(df_treino)} ({int(df_treino.num_ano.min())}-{int(df_treino.num_mes.min())} a {int(df_treino.num_ano.max())}-{int(df_treino.num_mes.max())})")
    print(f"  Teste:  {len(df_teste)} ({int(df_teste.num_ano.min())}-{int(df_teste.num_mes.min())} a {int(df_teste.num_ano.max())}-{int(df_teste.num_mes.max())})")

    # Isolation Forest
    iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=200, max_samples='auto', n_jobs=-1)
    iso.fit(X_treino)
    labels_treino = iso.predict(X_treino)
    labels_teste = iso.predict(X_teste)
    scores_treino = iso.decision_function(X_treino)
    scores_teste = iso.decision_function(X_teste)

    anom_t = (labels_treino == -1).sum()
    anom_e = (labels_teste == -1).sum()
    print(f"  IF: treino {anom_t}/{len(X_treino)} ({(labels_treino==-1).mean()*100:.1f}%) | teste {anom_e}/{len(X_teste)} ({(labels_teste==-1).mean()*100:.1f}%)")

    # Montar resultado
    df_resultado = df_pivot[cols_context + cols_rub].copy()
    for rub, yoy_col in zip(cols_rub, cols_yoy):
        df_resultado[yoy_col] = df_yoy[yoy_col].values

    df_resultado['IF_label'] = -1
    df_resultado['IF_score'] = 0.0

    for pos, label, score in zip(treino_pos, labels_treino, scores_treino):
        df_resultado.iloc[pos, df_resultado.columns.get_loc('IF_label')] = label
        df_resultado.iloc[pos, df_resultado.columns.get_loc('IF_score')] = score
    for pos, label, score in zip(teste_pos, labels_teste, scores_teste):
        df_resultado.iloc[pos, df_resultado.columns.get_loc('IF_label')] = label
        df_resultado.iloc[pos, df_resultado.columns.get_loc('IF_score')] = score

    df_resultado['anomalo'] = (df_resultado['IF_label'] == -1).astype(int)

    # Salvar
    caminho_csv = os.path.join(output_dir, f'anomalias_yoy_{cargo}.csv')
    df_resultado.to_csv(caminho_csv, index=False, sep=';')
    print(f"  💾 {caminho_csv}")

    return {
        'cargo': cargo, 'n_amostras': n_rows, 'n_rubricas': len(cols_rub),
        'yoy_validos_pct': round(pct_validos, 1),
        'n_anomalias_treino': int(anom_t), 'n_anomalias_teste': int(anom_e),
        'csv_path': caminho_csv
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--contamination', type=float, default=0.05)
    parser.add_argument('--output', type=str, default='data/baseline_results_yoy')
    args = parser.parse_args()
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"# BASELINE YoY — Year-over-Year (vetorizado)")
    print(f"# Passo 1: YoY = mês_atual / mesmo_mês_ano_anterior")
    print(f"# Passo 2: Isolation Forest nos valores YoY")
    print(f"# Passo 3: Por cargo, divisão temporal 80/20")
    print(f"# Contamination: {args.contamination*100:.1f}%")
    print(f"# Saída: {output_dir}")
    print(f"{'#'*60}\n")

    df = pd.read_csv('data/historico_pagamento_15.csv', delimiter=';')
    print(f"Dados: {len(df):,} linhas | {df.cod_cargo.nunique()} cargos")

    summary = []
    start = datetime.now()
    for cargo in sorted(df['cod_cargo'].unique()):
        res = run_cargo(df[df['cod_cargo'] == cargo].copy(), cargo, output_dir, args.contamination)
        if res:
            summary.append(res)

    elapsed = (datetime.now() - start).total_seconds()

    print(f"\n{'='*70}")
    print(f"  RESUMO GERAL (tempo: {elapsed:.1f}s)")
    print(f"{'='*70}")
    print(f"\n{'Cargo':<10} {'Amostras':>10} {'YoY val%':>10} {'Anom T':>8} {'Anom E':>8}")
    print('-' * 60)
    for s in summary:
        print(f"{s['cargo']:<10} {s['n_amostras']:>10,} {s['yoy_validos_pct']:>9.1f}% {s['n_anomalias_treino']:>8} {s['n_anomalias_teste']:>8}")

    with open(os.path.join(output_dir, 'resumo_yoy.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n✅ Resumo: {output_dir}/resumo_yoy.json")
