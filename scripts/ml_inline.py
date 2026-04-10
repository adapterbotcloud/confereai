"""
ML Analysis em tempo real para a API Flask.
Roda YoY + Isolation Forest por cargo sob demanda.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import json
import os
import uuid
from pathlib import Path
from datetime import datetime
import threading

# ─── CONFIG ───────────────────────────────────────────────────────────────────

DATA_FILE = Path(__file__).parent.parent / 'data' / 'historico_5_seplag.csv'
CACHE_DIR = Path(__file__).parent.parent / 'data' / 'ml_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

JOBS = {}  # job_id -> {'status', 'result', 'error'}

SITUACAO_MAP = {
    '0': 'Civil Ativo', '1': 'Militar Ativo',
    '2': 'Civil Afastado c/ onus', '3': 'Militar Afastado c/ onus',
    '4': 'Civil Afastado', '5': 'Militar Afastado',
    '6': 'Pensionista', '7': 'Pensão Alimento', '8': 'Liminar',
}

RUBRICAS_MAP = {}  # carregado under demand


def _load_data():
    """Carrega dados uma vez (cache em memória)."""
    if not hasattr(_load_data, '_df'):
        df = pd.read_csv(DATA_FILE, delimiter=';', dtype=str, encoding='utf-8')
        for col in df.columns:
            df[col] = df[col].str.strip().str.strip('"')
        df['vlr_calculado'] = pd.to_numeric(df['vlr_calculado'], errors='coerce')
        df['cod_situacao_funcional'] = df['cod_situacao_funcional'].str.strip()
        df['dsc_situacao'] = df['cod_situacao_funcional'].map(SITUACAO_MAP).fillna(df['cod_situacao_funcional'])
        _load_data._df = df
    return _load_data._df


def run_yoy(df_cargo, cargo, contamination=0.05):
    """YoY + Isolation Forest para um cargo."""
    
    # Pivot: rubricas em colunas
    df_pivot = df_cargo.pivot_table(
        index=['isn_vinculo', 'num_ano', 'num_mes'],
        columns='isn_rubrica',
        values='vlr_calculado',
        aggfunc='mean'
    ).fillna(0)
    
    df_pivot.columns = [str(c) for c in df_pivot.columns]
    df_pivot = df_pivot.reset_index()
    
    cols_context = ['isn_vinculo', 'num_ano', 'num_mes']
    cols_rub = [c for c in df_pivot.columns if c not in cols_context]
    df_pivot = df_pivot[cols_context + cols_rub]
    df_pivot = df_pivot.sort_values(['isn_vinculo', 'num_ano', 'num_mes']).reset_index(drop=True)
    n_rows = len(df_pivot)
    
    if n_rows < 24:
        return None
    
    # YoY vetorizado
    df_yoy = df_cargo.pivot_table(
        index=['isn_vinculo', 'num_ano', 'num_mes'],
        columns='isn_rubrica',
        values='vlr_calculado',
        aggfunc='mean'
    ).fillna(0)
    df_yoy.columns = [str(c) for c in df_yoy.columns]
    df_yoy = df_yoy.reset_index()
    df_yoy = df_yoy.sort_values(['isn_vinculo', 'num_ano', 'num_mes']).reset_index(drop=True)
    
    cols_yoy = []
    for rub in cols_rub:
        col_yoy = rub + '_yoy'
        cols_yoy.append(col_yoy)
        tmp = df_yoy[['isn_vinculo', 'num_mes', rub]].copy()
        tmp = tmp.sort_values(['isn_vinculo', 'num_mes']).reset_index(drop=True)
        tmp[col_yoy] = tmp.groupby(['isn_vinculo', 'num_mes'])[rub].shift(12)
        df_yoy[col_yoy] = tmp[col_yoy].values
    
    for rub, yoy_col in zip(cols_rub, cols_yoy):
        df_yoy[yoy_col] = df_pivot[rub] / df_yoy[yoy_col]
        df_yoy[yoy_col] = df_yoy[yoy_col].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    
    n_yoy_validos = df_yoy[cols_yoy].notna().sum().sum()
    n_total = n_rows * len(cols_yoy)
    pct_validos = n_yoy_validos / n_total * 100 if n_total > 0 else 0
    
    # Divisão temporal 80/20
    split_idx = int(n_rows * 0.8)
    treino_pos = list(range(split_idx))
    teste_pos = list(range(split_idx, n_rows))
    
    X_treino = df_yoy.iloc[:split_idx][cols_yoy].values
    X_teste = df_yoy.iloc[split_idx:][cols_yoy].values
    
    df_treino = df_yoy.iloc[:split_idx].copy()
    df_teste = df_yoy.iloc[split_idx:].copy()
    
    # Isolation Forest
    iso = IsolationForest(
        contamination=contamination, random_state=42,
        n_estimators=200, max_samples='auto', n_jobs=-1
    )
    iso.fit(X_treino)
    labels_treino = iso.predict(X_treino)
    labels_teste = iso.predict(X_teste)
    scores_treino = iso.decision_function(X_treino)
    scores_teste = iso.decision_function(X_teste)
    
    # Montar resultado final
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
    df_resultado['dsc_situacao'] = df_cargo.groupby(['isn_vinculo', 'num_ano', 'num_mes'])['dsc_situacao'].first().reindex(
        df_resultado.set_index(['isn_vinculo', 'num_ano', 'num_mes']).index
    ).values if 'dsc_situacao' in df_cargo.columns else 'N/A'
    
    # Períodos
    periodo_treino = f"{int(df_treino.num_ano.min())}/{int(df_treino.num_mes.min())} - {int(df_treino.num_ano.max())}/{int(df_treino.num_mes.max())}"
    periodo_teste = f"{int(df_teste.num_ano.min())}/{int(df_teste.num_mes.min())} - {int(df_teste.num_ano.max())}/{int(df_teste.num_mes.max())}"
    
    # Vínculos mais anômalos
    anom_teste = df_resultado.iloc[split_idx:].copy()
    anom_teste['isn_vinculo'] = anom_teste['isn_vinculo'].astype(str)
    
    if anom_teste['anomalo'].sum() > 0:
        top = (anom_teste[anom_teste['anomalo'] == 1]
               .groupby('isn_vinculo')
               .agg(
                   qtd_meses=('anomalo', 'count'),
                   score_medio=('IF_score', 'mean'),
               )
               .reset_index()
               .sort_values('score_medio', ascending=True)
               .head(30))
        top['score_medio'] = top['score_medio'].round(4)
        top_vinculos = top.to_dict('records')
    else:
        top_vinculos = []
    
    return {
        'cargo': cargo,
        'method': 'yoy',
        'n_total': n_rows,
        'n_treino': split_idx,
        'n_teste': n_rows - split_idx,
        'periodo_treino': periodo_treino,
        'periodo_teste': periodo_teste,
        'n_rubricas': len(cols_rub),
        'yoy_validos_pct': round(pct_validos, 1),
        'pct_anomalias_treino': round((labels_treino == -1).mean() * 100, 1),
        'pct_anomalias_teste': round((labels_teste == -1).mean() * 100, 1),
        'score_medio_teste': round(scores_teste.mean(), 4),
        'top_vinculos_anomalos': top_vinculos,
        'timestamp': datetime.now().isoformat(),
    }


def run_cagr(df_cargo, cargo, contamination=0.05):
    """CAGR ajustado + Isolation Forest."""
    # Similar ao YoY mas deflaciona por CAGR
    # Por enquanto, usa StandardScaler simples (CAGR real precisa calcular rubrica a rubrica)
    df_pivot = df_cargo.pivot_table(
        index=['isn_vinculo', 'num_ano', 'num_mes'],
        columns='isn_rubrica',
        values='vlr_calculado',
        aggfunc='mean'
    ).fillna(0)
    df_pivot.columns = [str(c) for c in df_pivot.columns]
    df_pivot = df_pivot.reset_index()
    
    cols_context = ['isn_vinculo', 'num_ano', 'num_mes']
    cols_rub = [c for c in df_pivot.columns if c not in cols_context]
    n_rows = len(df_pivot)
    
    if n_rows < 24:
        return None
    
    # Normalização por período
    split_idx = int(n_rows * 0.8)
    treino_pos = list(range(split_idx))
    teste_pos = list(range(split_idx, n_rows))
    
    # Pivot final
    df_pivot = df_pivot[cols_context + cols_rub]
    df_pivot = df_pivot.sort_values(['isn_vinculo', 'num_ano', 'num_mes']).reset_index(drop=True)
    
    # StandardScaler por período
    scaler_treino = StandardScaler()
    X_treino = scaler_treino.fit_transform(df_pivot.iloc[:split_idx][cols_rub].values)
    X_teste = scaler_treino.transform(df_pivot.iloc[split_idx:][cols_rub].values)
    
    iso = IsolationForest(
        contamination=contamination, random_state=42,
        n_estimators=200, max_samples='auto', n_jobs=-1
    )
    iso.fit(X_treino)
    labels_treino = iso.predict(X_treino)
    labels_teste = iso.predict(X_teste)
    scores_treino = iso.decision_function(X_treino)
    scores_teste = iso.decision_function(X_teste)
    
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
    
    df_treino = df_resultado.iloc[:split_idx]
    df_teste = df_resultado.iloc[split_idx:]
    
    anom_teste = df_teste.copy()
    if anom_teste['anomalo'].sum() > 0:
        top = (anom_teste[anom_teste['anomalo'] == 1]
               .groupby('isn_vinculo')
               .agg(qtd_meses=('anomalo','count'), score_medio=('IF_score','mean'))
               .reset_index().sort_values('score_medio', ascending=True).head(30))
        top['score_medio'] = top['score_medio'].round(4)
        top_vinculos = top.to_dict('records')
    else:
        top_vinculos = []
    
    return {
        'cargo': cargo,
        'method': 'cagr',
        'n_total': n_rows,
        'n_treino': split_idx,
        'n_teste': n_rows - split_idx,
        'periodo_treino': f"{int(df_treino.num_ano.min())}/{int(df_treino.num_mes.min())} - {int(df_treino.num_ano.max())}/{int(df_treino.num_mes.max())}",
        'periodo_teste': f"{int(df_teste.num_ano.min())}/{int(df_teste.num_mes.min())} - {int(df_teste.num_ano.max())}/{int(df_teste.num_mes.max())}",
        'n_rubricas': len(cols_rub),
        'pct_anomalias_treino': round((labels_treino == -1).mean() * 100, 1),
        'pct_anomalias_teste': round((labels_teste == -1).mean() * 100, 1),
        'score_medio_teste': round(scores_teste.mean(), 4),
        'top_vinculos_anomalos': top_vinculos,
        'timestamp': datetime.now().isoformat(),
    }


def run_background(method, cargo, job_id):
    """Executa ML em background e salva resultado."""
    try:
        df = _load_data()
        df_cargo = df[df['cod_cargo'] == cargo].copy()
        
        if len(df_cargo) == 0:
            JOBS[job_id] = {'status': 'error', 'error': f'Cargo {cargo} nao encontrado'}
            return
        
        result = None
        if method == 'yoy':
            result = run_yoy(df_cargo, cargo)
        elif method == 'cagr':
            result = run_cagr(df_cargo, cargo)
        elif method == 'temporal':
            result = run_cagr(df_cargo, cargo)  # mesmo de cagr por enquanto
        
        if result:
            # Salvar no cache
            cache_file = CACHE_DIR / f'{method}_{cargo}.json'
            with open(cache_file, 'w') as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)
            JOBS[job_id] = {'status': 'done', 'result': result}
        else:
            JOBS[job_id] = {'status': 'error', 'error': 'Dados insuficientes para analise'}
    
    except Exception as e:
        JOBS[job_id] = {'status': 'error', 'error': str(e)}


def start_job(method, cargo):
    """Inicia job de ML em background."""
    job_id = str(uuid.uuid4())[:8]
    JOBS[job_id] = {'status': 'running', 'method': method, 'cargo': cargo}
    t = threading.Thread(target=run_background, args=(method, cargo, job_id))
    t.start()
    return job_id


def get_job(job_id):
    """Retorna status/resultado de um job."""
    return JOBS.get(job_id, {'status': 'unknown'})
