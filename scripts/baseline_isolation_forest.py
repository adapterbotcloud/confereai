"""
Baseline de Detecção de Anomalias - Isolation Forest + One-Class SVM
Por cargo, usando os dados de historico_pagamento_15.csv

Uso:
    python scripts/baseline_isolation_forest.py [--contamination 0.05]
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import os
import argparse
import json
from datetime import datetime


# ── Helpers de preprocessing ────────────────────────────────────────────────

def pivotar_por_cargo(df_cargo):
    """
    Pivot: index = RangeIndex sequencial (0, 1, 2...)
           columns = isn_rubrica
           values = vlr_calculado
    Agrega por média quando há duplicatas.
    Preenche NaN com 0.
    Usa RangeIndex para garantir alinhamento com X numpy nas divisões.
    """
    df_pivot = df_cargo.pivot_table(
        index=['isn_vinculo', 'num_ano', 'num_mes'],
        columns='isn_rubrica',
        values='vlr_calculado',
        aggfunc='mean'
    )
    df_pivot = df_pivot.fillna(0)
    # Reset pra garantir RangeIndex sequencial — evita conflito com MultiIndex
    df_pivot = df_pivot.reset_index()
    # Mantém só as 3 colunas de contexto + rubricas
    cols_context = ['isn_vinculo', 'num_ano', 'num_mes']
    cols_rubricas = [c for c in df_pivot.columns if c not in cols_context]
    df_pivot = df_pivot[cols_context + cols_rubricas]
    df_pivot = df_pivot.sort_values(['num_ano', 'num_mes']).reset_index(drop=True)
    return df_pivot


def normalizar(df_pivot, scaler_type='robust'):
    """
    Normaliza as colunas de rubrica.
    Garante índice sequencial (RangeIndex) em df_norm para alinhar com X numpy.
    Retorna DataFrame normalizado + scaler.
    """
    colunas_rubricas = [c for c in df_pivot.columns if c not in ['isn_vinculo', 'num_ano', 'num_mes']]
    X = df_pivot[colunas_rubricas].values

    if scaler_type == 'robust':
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X)
    else:
        scaler = MinMaxScaler(feature_range=(0.1, 1))
        X_norm = scaler.fit_transform(X)

    # Usa RangeIndex sequencial para alinhar com índices do numpy array
    df_norm = pd.DataFrame(X_norm, index=range(len(X_norm)), columns=colunas_rubricas)
    return df_norm, scaler, colunas_rubricas


def treinar_isolation_forest(X, contamination=0.05, random_state=42):
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=200,
        max_samples='auto',
        n_jobs=-1
    )
    labels = model.fit_predict(X)
    scores = model.decision_function(X)
    return labels, scores, model


def dividir_temporal(df_pivot, proporcao=0.8):
    """
    Divisão TEMPORAL: 80% mais antigo = treino, 20% mais recente = teste.
    df_pivot já vem ordenado por (num_ano, num_mes) do pivotar_por_cargo.
    Retorna posições sequenciais (RangeIndex) para alinhar com X numpy.
    """
    n = len(df_pivot)
    split_idx = int(n * proporcao)
    treino_pos = list(range(split_idx))
    teste_pos = list(range(split_idx, n))
    return treino_pos, teste_pos


def treinar_one_class_svm(X, contamination=0.05, nu=0.05):
    """
    One-Class SVM. nu approx contamination.
    Usamos só uma fração da amostra se for muito grande (O(n²) é caro).
    """
    # Se > 5000 amostras, usa subset pra treinar (mais rápido)
    if X.shape[0] > 5000:
        idx = np.random.RandomState(42).choice(X.shape[0], 5000, replace=False)
        X_train = X[idx]
    else:
        X_train = X

    model = OneClassSVM(
        kernel='rbf',
        gamma='scale',
        nu=nu
    )
    model.fit(X_train)
    labels = model.predict(X)
    scores = model.decision_function(X)
    return labels, scores, model


def run_cargo(df_cargo, cargo, output_dir, contamination, methods):
    """
    Para um cargo: pivot → normaliza → roda métodos → salva resultados.
    """
    print(f"\n{'='*60}")
    print(f"  Cargo: {cargo} | {len(df_cargo):,} registros")
    print(f"{'='*60}")

    # 1) Pivot
    df_pivot = pivotar_por_cargo(df_cargo)
    n_rows, n_features = df_pivot.shape
    print(f"  Pivot: {n_rows} amostras × {n_features} rubricas")

    if n_rows < 20:
        print(f" Pulando — poucos dados ({n_rows} < 20)")
        return None

    # 2) Normalizar
    df_norm, scaler, colunas_rubricas = normalizar(df_pivot, scaler_type='robust')
    X = df_norm.values

    # 3) Guardar scaler
    scaler_path = os.path.join(output_dir, f'scaler_{cargo}.pkl')
    joblib.dump(scaler, scaler_path)

    # Divisão TEMPORAL: 80% mais antigo / 20% mais recente
    # Crucial para não ter vazamento de dados futuros no treino
    if n_rows > 1:
        treino_pos, teste_pos = dividir_temporal(df_pivot, proporcao=0.8)
        X_treino = X[treino_pos]
        X_teste = X[teste_pos]
    else:
        treino_pos = list(range(n_rows))
        teste_pos = list(range(n_rows))
        X_treino = X
        X_teste = X

    print(f"  Treino: {len(X_treino)} | Teste: {len(X_teste)}")

    resultados = {}

    # ── Isolation Forest ──
    if 'if' in methods:
        print("  [1/2] Isolation Forest...", end=' ')
        labels_treino, scores_treino, model_if = treinar_isolation_forest(X_treino, contamination)
        labels_teste, scores_teste, _ = treinar_isolation_forest(X_teste, contamination)

        # Juntando pra salvar
        labels_all = np.concatenate([labels_treino, labels_teste])
        scores_all = np.concatenate([scores_treino, scores_teste])
        idx_all = np.array(treino_pos + teste_pos)

        resultados['IF'] = {
            'labels': labels_all,
            'scores': scores_all,
            'idx': idx_all,
            'n_anomalias_treino': int((labels_treino == -1).sum()),
            'n_anomalias_teste': int((labels_teste == -1).sum()),
        }
        print(f"OK | treino: {(labels_treino==-1).sum()} anômalas | teste: {(labels_teste==-1).sum()} anômalas")

    # ── One-Class SVM ──
    if 'ocsvm' in methods:
        print("  [2/2] One-Class SVM...", end=' ')
        labels_treino_svm, scores_treino_svm, model_svm = treinar_one_class_svm(X_treino, contamination)
        labels_teste_svm, scores_teste_svm, _ = treinar_one_class_svm(X_teste, contamination)

        labels_all_svm = np.concatenate([labels_treino_svm, labels_teste_svm])
        scores_all_svm = np.concatenate([scores_treino_svm, scores_teste_svm])

        resultados['OCSVM'] = {
            'labels': labels_all_svm,
            'scores': scores_all_svm,
            'idx': np.array(treino_pos + teste_pos),
            'n_anomalias_treino': int((labels_treino_svm == -1).sum()),
            'n_anomalias_teste': int((labels_teste_svm == -1).sum()),
        }
        print(f"OK | treino: {(labels_treino_svm==-1).sum()} anômalas | teste: {(labels_teste_svm==-1).sum()} anômalas")

    # ── Montar DataFrame de resultados ──
    # df_pivot já tem RangeIndex sequencial — usa iloc para indexação posicional
    df_resultado = df_pivot.copy()

    if 'if' in methods:
        df_resultado['IF_label'] = -1   # placeholder
        df_resultado['IF_score'] = 0.0
        for pos, label, score in zip(resultados['IF']['idx'],
                                     resultados['IF']['labels'],
                                     resultados['IF']['scores']):
            df_resultado.iloc[pos, df_resultado.columns.get_loc('IF_label')] = label
            df_resultado.iloc[pos, df_resultado.columns.get_loc('IF_score')] = score

    if 'ocsvm' in methods:
        df_resultado['OCSVM_label'] = -1
        df_resultado['OCSVM_score'] = 0.0
        for pos, label, score in zip(resultados['OCSVM']['idx'],
                                     resultados['OCSVM']['labels'],
                                     resultados['OCSVM']['scores']):
            df_resultado.iloc[pos, df_resultado.columns.get_loc('OCSVM_label')] = label
            df_resultado.iloc[pos, df_resultado.columns.get_loc('OCSVM_score')] = score

    df_resultado['anomalo'] = 0
    if 'if' in methods:
        df_resultado['anomalo'] += (df_resultado['IF_label'] == -1).astype(int)
    if 'ocsvm' in methods:
        df_resultado['anomalo'] += (df_resultado['OCSVM_label'] == -1).astype(int)

    # Salvar CSV por cargo
    nome_csv = f'anomalias_baseline_{cargo}.csv'
    caminho_csv = os.path.join(output_dir, nome_csv)
    df_resultado.to_csv(caminho_csv, index=False, sep=';')
    print(f"  💾 Salvo em {caminho_csv}")

    # Salvar scaler
    scaler_path = os.path.join(output_dir, f'scaler_{cargo}.pkl')
    joblib.dump(scaler, scaler_path)

    return {
        'cargo': cargo,
        'n_amostras': n_rows,
        'n_rubricas': n_features,
        'resultados': {k: {kk: vv for kk, vv in v.items() if kk not in ('idx', 'labels')}
                       for k, v in resultados.items()},
        'csv_path': caminho_csv
    }


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import joblib

    parser = argparse.ArgumentParser(description='Baseline IF + OCSVM por cargo')
    parser.add_argument('--contamination', type=float, default=0.05,
                        help='Fraçao esperada de anomalias (default: 0.05 = 5%%)')
    parser.add_argument('--methods', type=str, default='if,ocsvm',
                        help='Metodos: if,ocsvm (default: ambos)')
    parser.add_argument('--output', type=str, default='data/baseline_results',
                        help='Pasta de saida (default: data/baseline_results)')
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(',')]
    contamination = args.contamination
    output_dir = args.output

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"# BASELINE - Isolation Forest + One-Class SVM")
    print(f"# Contamination: {contamination*100:.1f}%")
    print(f"# Metodos: {methods}")
    print(f"# Saida: {output_dir}")
    print(f"{'#'*60}\n")

    # Carregar dados brutos
    df = pd.read_csv('data/historico_pagamento_15.csv', delimiter=';')
    print(f"Dados carregados: {len(df):,} linhas")

    # Processar cada cargo
    summary = []
    start = datetime.now()

    for cargo in sorted(df['cod_cargo'].unique()):
        df_cargo = df[df['cod_cargo'] == cargo].copy()
        res = run_cargo(df_cargo, cargo, output_dir, contamination, methods)
        if res:
            summary.append(res)

    elapsed = (datetime.now() - start).total_seconds()

    # ── Resumo consolidado ──
    print(f"\n{'='*60}")
    print(f"  RESUMO GERAL (tempo total: {elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"\n{'Cargo':<10} {'Amostras':>10} {'Rubricas':>10} ", end='')
    if 'if' in methods:
        print(f"{'IF Anom.(%T)':>14} {'IF Anom.(%E)':>14}", end='')
    if 'ocsvm' in methods:
        print(f"{'SVM Anom.(%T)':>14} {'SVM Anom.(%E)':>14}", end='')
    print()
    print('-'*80)

    for s in summary:
        cargo = s['cargo']
        n = s['n_amostras']
        print(f"{cargo:<10} {n:>10,}", end='')
        if 'if' in methods:
            if_t = s['resultados']['IF']['n_anomalias_treino']
            if_e = s['resultados']['IF']['n_anomalias_teste']
            n_treino = n - s['resultados']['IF'].get('n_anomalias_teste', 0)
            print(f"{if_t/n*100:>13.1f}% {if_e/n*100:>13.1f}%", end='')
        if 'ocsvm' in methods:
            svm_t = s['resultados']['OCSVM']['n_anomalias_treino']
            svm_e = s['resultados']['OCSVM']['n_anomalias_teste']
            print(f"{svm_t/n*100:>13.1f}% {svm_e/n*100:>13.1f}%", end='')
        print()

    # Salvar resumo JSON
    resumo_path = os.path.join(output_dir, 'resumo_baseline.json')
    with open(resumo_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n✅ Resumo JSON: {resumo_path}")
