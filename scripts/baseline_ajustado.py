"""
Baseline de Detecção de Anomalias — VERSÃO CORRIGIDA
Agora com:
  1. Ajuste de CAGR: deflaciona valores para ano-base usando crescimento histórico
  2. Normalização por ano: StandardScaler dentro de cada ano (não global)
  3. Divisão temporal: 80% mais antigo = treino, 20% mais recente = teste

Uso:
    python scripts/baseline_ajustado.py [--contamination 0.05]
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os
import argparse
import json
from datetime import datetime


# ── Helpers ──────────────────────────────────────────────────────────────────

def pivotar_por_cargo(df_cargo):
    """Pivot: rubricas em colunas, valor como média."""
    df_pivot = df_cargo.pivot_table(
        index=['isn_vinculo', 'num_ano', 'num_mes'],
        columns='isn_rubrica',
        values='vlr_calculado',
        aggfunc='mean'
    ).fillna(0)
    df_pivot = df_pivot.reset_index()
    cols_context = ['isn_vinculo', 'num_ano', 'num_mes']
    cols_rub = [c for c in df_pivot.columns if c not in cols_context]
    df_pivot = df_pivot[cols_context + cols_rub]
    df_pivot = df_pivot.sort_values(['num_ano', 'num_mes']).reset_index(drop=True)
    return df_pivot


def calc_cagr(valor_inicial, valor_final, anos):
    if valor_inicial <= 0 or anos <= 0:
        return 0.0
    return ((valor_final / valor_inicial) ** (1/anos) - 1)


def calcular_cagr_por_rubrica(df_cargo, cols_rub):
    """
    Calcula CAGR de cada rubrica dentro de um cargo.
    Usa média por ano, depois CAGR entre primeiro e último ano com dados.
    Retorna dict: {rubrica: cagr_anual}
    """
    anos_disponiveis = sorted(df_cargo['num_ano'].unique())
    if len(anos_disponiveis) < 2:
        return {r: 0.0 for r in cols_rub}

    cagrs = {}
    for rub in cols_rub:
        df_rub = df_cargo[df_cargo['isn_rubrica'] == rub]
        anos_rub = sorted(df_rub['num_ano'].unique())
        if len(anos_rub) < 2:
            cagrs[rub] = 0.0
            continue

        media_primeiro = df_rub[df_rub['num_ano'] == anos_rub[0]]['vlr_calculado'].mean()
        media_ultimo = df_rub[df_rub['num_ano'] == anos_rub[-1]]['vlr_calculado'].mean()
        anos_diff = anos_rub[-1] - anos_rub[0]

        cagr = calc_cagr(media_primeiro, media_ultimo, anos_diff)
        cagrs[rub] = max(0.0, min(cagr, 0.30))  # limita entre 0 e 30% pra não explodir
    return cagrs


def deflacionar(df_pivot, df_cargo_original, cols_rub, cagrs, ano_base):
    """
    Translada todos os valores para o ano_base usando o CAGR histórico.
    valor_ajustado = valor / (1 + cagr) ^ (ano - ano_base)
    """
    df_adj = df_pivot.copy()
    for rub in cols_rub:
        if rub not in cagrs or cagrs[rub] == 0.0:
            continue
        cagr = cagrs[rub]
        delta_anos = df_adj['num_ano'] - ano_base
        fator = (1 + cagr) ** delta_anos
        # Evita divisão por zero
        fator = fator.replace(0, 1)
        df_adj[rub] = df_adj[rub] / fator
    return df_adj


def normalizar_por_ano(df_adj, cols_rub):
    """
    Normaliza cada ano separadamente com StandardScaler.
    Garante que valores de anos diferentes são comparáveis.
    """
    df_norm = df_adj.copy()
    for ano in df_adj['num_ano'].unique():
        mask = df_adj['num_ano'] == ano
        X_ano = df_adj.loc[mask, cols_rub].values
        if X_ano.shape[0] < 2:
            continue
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X_ano)
        df_norm.loc[mask, cols_rub] = X_norm
    return df_norm


def dividir_temporal(df_pivot, proporcao=0.8):
    """Divisão temporal: 80% mais antigo = treino, 20% mais recente = teste."""
    n = len(df_pivot)
    split_idx = int(n * proporcao)
    treino_pos = list(range(split_idx))
    teste_pos = list(range(split_idx, n))
    return treino_pos, teste_pos


def run_cargo(df_cargo, cargo, output_dir, contamination):
    print(f"\n{'='*60}")
    print(f"  Cargo: {cargo} | {len(df_cargo):,} registros")
    print(f"{'='*60}")

    df_pivot = pivotar_por_cargo(df_cargo)
    n_rows, n_features = len(df_pivot), len(df_pivot.columns) - 3
    print(f"  Pivot: {n_rows} amostras × {n_features} rubricas")

    if n_rows < 20:
        print(f"  Pulando — poucos dados")
        return None

    cols_rub = [c for c in df_pivot.columns if c not in ['isn_vinculo', 'num_ano', 'num_mes']]

    # 1) Calcular CAGR por rubrica
    cagrs = calcular_cagr_por_rubrica(df_cargo, cols_rub)
    cagrs_nonzero = {k: v for k, v in cagrs.items() if v > 0}
    print(f"  Rubricas com CAGR>0: {len(cagrs_nonzero)}/{n_features}")
    if cagrs_nonzero:
        maiores = sorted(cagrs_nonzero.items(), key=lambda x: x[1], reverse=True)[:3]
        for rub, cagr in maiores:
            print(f"    Rubrica {rub}: CAGR {cagr*100:.2f}%/ano")

    # 2) Deflacionar para ano base (ano mais antigo do treino)
    ano_base = int(df_pivot['num_ano'].min())
    df_adj = deflacionar(df_pivot, df_cargo, cols_rub, cagrs, ano_base)
    print(f"  Deflacionado para ano-base: {ano_base}")

    # 3) Normalizar por ano
    df_norm = normalizar_por_ano(df_adj, cols_rub)
    X = df_norm[cols_rub].values
    print(f"  Normalizado por ano (StandardScaler)")

    # 4) Divisão temporal
    treino_pos, teste_pos = dividir_temporal(df_pivot, 0.8)
    X_treino = X[treino_pos]
    X_teste = X[teste_pos]
    print(f"  Divisão temporal: treino {len(X_treino)} | teste {len(X_teste)}")

    # 5) Treinar Isolation Forest só no treino
    iso = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=200,
        max_samples='auto',
        n_jobs=-1
    )
    iso.fit(X_treino)

    # Predizer treino e teste
    labels_treino = iso.predict(X_treino)
    labels_teste = iso.predict(X_teste)
    scores_treino = iso.decision_function(X_treino)
    scores_teste = iso.decision_function(X_teste)

    anom_treino = (labels_treino == -1).sum()
    anom_teste = (labels_teste == -1).sum()
    print(f"  IF: treino {(labels_treino==-1).sum()}/{(labels_treino==1).sum()} "
          f"| teste {(labels_teste==-1).sum()}/{(labels_teste==1).sum()}")

    # 6) Montar resultado
    df_resultado = df_pivot.copy()
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
    nome_csv = f'anomalias_ajustado_{cargo}.csv'
    caminho_csv = os.path.join(output_dir, nome_csv)
    df_resultado.to_csv(caminho_csv, index=False, sep=';')
    print(f"  💾 Salvo em {caminho_csv}")

    # Salvar CAGRs do cargo
    cagr_path = os.path.join(output_dir, f'cagr_{cargo}.json')
    with open(cagr_path, 'w') as f:
        json.dump({str(k): v for k, v in cagrs.items()}, f)

    return {
        'cargo': cargo,
        'n_amostras': n_rows,
        'n_rubricas': n_features,
        'n_rubricas_com_cagr': len(cagrs_nonzero),
        'cagrs': cagrs,
        'n_anomalias_treino': int(anom_treino),
        'n_anomalias_teste': int(anom_teste),
        'csv_path': caminho_csv
    }


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--contamination', type=float, default=0.05)
    parser.add_argument('--output', type=str, default='data/baseline_results_ajustado')
    args = parser.parse_args()

    output_dir = args.output
    contamination = args.contamination
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"# BASELINE AJUSTADO — CAGR + Normalização por Ano")
    print(f"# Contamination: {contamination*100:.1f}%")
    print(f"# Saída: {output_dir}")
    print(f"{'#'*60}\n")

    df = pd.read_csv('data/historico_pagamento_15.csv', delimiter=';')
    print(f"Dados: {len(df):,} linhas | {df.cod_cargo.nunique()} cargos")

    summary = []
    start = datetime.now()

    for cargo in sorted(df['cod_cargo'].unique()):
        df_cargo = df[df['cod_cargo'] == cargo].copy()
        res = run_cargo(df_cargo, cargo, output_dir, contamination)
        if res:
            summary.append(res)

    elapsed = (datetime.now() - start).total_seconds()

    # Resumo
    print(f"\n{'='*70}")
    print(f"  RESUMO GERAL (tempo: {elapsed:.1f}s)")
    print(f"{'='*70}")
    print(f"\n{'Cargo':<10} {'Amostras':>10} {'Rub c/ CAGR':>11} {'Anom T':>8} {'Anom E':>8}")
    print('-' * 60)
    for s in summary:
        print(f"{s['cargo']:<10} {s['n_amostras']:>10,} {s['n_rubricas_com_cagr']:>11} "
              f"{s['n_anomalias_treino']:>8} {s['n_anomalias_teste']:>8}")

    resumo_path = os.path.join(output_dir, 'resumo_ajustado.json')
    with open(resumo_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n✅ Resumo: {resumo_path}")
