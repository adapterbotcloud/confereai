#!/usr/bin/env python3
"""
API Flask para o Admin de Regras do ConfereAI.
Roda em localhost:5001 por padrão.

Endpoints:
  GET  /api/regras              - Lista todas as regras
  POST /api/regras              - Adiciona regra
  PUT  /api/regras/<id>         - Atualiza regra
  DELETE /api/regras/<id>        - Remove regra
  POST /api/regras/testar        - Testa regras contra dados
  GET  /api/regras/validar       - Valida estrutura do JSON
  GET  /api/dados/resumo         - Resumo dos dados (para teste)
"""

import json
import subprocess
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='static')
CORS(app)

REGRAS_FILE = Path(__file__).parent.parent / 'regras_ativos.json'
DATA_FILE = Path(__file__).parent.parent.parent / 'data' / 'historico_5_seplag.csv'


def carregar_regras():
    with open(REGRAS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def salvar_regras(data):
    with open(REGRAS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


# ─── API DE REGRAS ────────────────────────────────────────────────────────────

@app.route('/api/regras', methods=['GET'])
def listar_regras():
    data = carregar_regras()
    return jsonify(data)


@app.route('/api/regras', methods=['POST'])
def adicionar_regra():
    data = carregar_regras()
    nova = request.json
    
    # Gerar ID automático
    existentes = [int(r['id'][1:]) for r in data['regras'] if r['id'].startswith('R')]
    novo_id = max(existentes) + 1 if existentes else 1
    nova['id'] = f'R{novo_id:03d}'
    
    data['regras'].append(nova)
    data['_ultima_alteracao'] = '2026-04-10'
    salvar_regras(data)
    
    return jsonify({'ok': True, 'regra': nova}), 201


@app.route('/api/regras/<regra_id>', methods=['PUT'])
def atualizar_regra(regra_id):
    data = carregar_regras()
    atualizada = request.json
    
    for i, r in enumerate(data['regras']):
        if r['id'] == regra_id:
            atualizada['id'] = regra_id
            data['regras'][i] = atualizada
            data['_ultima_alteracao'] = '2026-04-10'
            salvar_regras(data)
            return jsonify({'ok': True, 'regra': atualizada})
    
    return jsonify({'ok': False, 'erro': 'Regra não encontrada'}), 404


@app.route('/api/regras/<regra_id>', methods=['DELETE'])
def remover_regra(regra_id):
    data = carregar_regras()
    
    antes = len(data['regras'])
    data['regras'] = [r for r in data['regras'] if r['id'] != regra_id]
    
    if len(data['regras']) == antes:
        return jsonify({'ok': False, 'erro': 'Regra não encontrada'}), 404
    
    data['_ultima_alteracao'] = '2026-04-10'
    salvar_regras(data)
    
    return jsonify({'ok': True})


# ─── TESTAR REGRAS ─────────────────────────────────────────────────────────────

@app.route('/api/regras/testar', methods=['POST'])
def testar_regras():
    """Roda as regras ativas contra os dados de folha."""
    import pandas as pd
    
    try:
        df = pd.read_csv(DATA_FILE, delimiter=';', dtype=str, encoding='utf-8')
        for col in df.columns:
            df[col] = df[col].str.strip().str.strip('"')
        df['vlr_calculado'] = pd.to_numeric(df['vlr_calculado'], errors='coerce')
    except Exception as e:
        return jsonify({'ok': False, 'erro': f'Erro ao carregar dados: {e}'}), 500
    
    regras_data = carregar_regras()
    todas_violacoes = []
    
    SITUACAO_MAP = {
        '0': 'Civil Ativo', '1': 'Militar Ativo',
        '2': 'Civil Afastado c/ onus', '3': 'Militar Afastado c/ onus',
        '4': 'Civil Afastado', '5': 'Militar Afastado',
        '6': 'Pensionista', '7': 'Pensao Alimento', '8': 'Liminar',
    }
    df['dsc_situacao'] = df['cod_situacao_funcional'].map(SITUACAO_MAP).fillna(df['cod_situacao_funcional'])
    
    for regra in regras_data['regras']:
        if regra.get('status') != 'ativa':
            continue
        
        situacoes = regra.get('situacao_funcional', [])
        rubricas_contem = regra.get('rubrica_contem', [])
        rubricas_nao_contem = regra.get('rubrica_nao_contem', [])
        
        # Condicao: situacao_funcional
        mask = df['cod_situacao_funcional'].isin(situacoes) if situacoes else pd.Series(True, index=df.index)
        
        # Condicao: rubrica_contem (qualquer um dos termos)
        if rubricas_contem:
            def contem(texto, termos):
                if pd.isna(texto):
                    return False
                return any(t.upper() in str(texto).upper() for t in termos)
            mask_rubrica = df['dsc_rubrica'].apply(lambda x: contem(x, rubricas_contem))
            mask = mask & mask_rubrica
        
        # Condicao: rubrica_nao_contem (NONE dos termos)
        if rubricas_nao_contem:
            def nao_contem(texto, termos):
                if pd.isna(texto):
                    return True
                return not any(t.upper() in str(texto).upper() for t in termos)
            mask_nao = df['dsc_rubrica'].apply(lambda x: nao_contem(x, rubricas_nao_contem))
            mask = mask & mask_nao
        
        violacoes = df[mask]
        if len(violacoes) > 0:
            grupo = violacoes.groupby(['isn_vinculo', 'dsc_rubrica', 'dsc_situacao'])['vlr_calculado'].agg(['count', 'sum']).reset_index()
            grupo['regra_id'] = regra['id']
            grupo['regra_nome'] = regra['nome']
            grupo.columns = ['isn_vinculo', 'dsc_rubrica', 'dsc_situacao', 'qtd', 'vlr_total', 'regra_id', 'regra_nome']
            todas_violacoes.append(grupo)
    
    if todas_violacoes:
        resultado = pd.concat(todas_violacoes, ignore_index=True)
        resultado = resultado.sort_values('vlr_total', ascending=False)
        return jsonify({
            'ok': True,
            'total_vinculos': int(resultado['isn_vinculo'].nunique()),
            'total_violacoes': int(len(resultado)),
            'valor_total': float(resultado['vlr_total'].sum()),
            'por_regra': resultado.groupby('regra_id').agg(
                qtd=('qtd','sum'), vlr_total=('vlr_total','sum'), regra_nome=('regra_nome','first')
            ).to_dict('records'),
            'detalhes': resultado.head(50).to_dict('records'),
        })
    
    return jsonify({'ok': True, 'total_vinculos': 0, 'total_violacoes': 0, 'valor_total': 0, 'por_regra': [], 'detalhes': []})


@app.route('/api/regras/validar', methods=['GET'])
def validar_regras():
    """Valida a estrutura do JSON de regras."""
    regras_data = carregar_regras()
    erros = []
    
    required = ['id', 'nome', 'descricao', 'situacao_funcional', 'rubrica_contem', 'rubrica_nao_contem', 'acao', 'severidade', 'status']
    
    for i, regra in enumerate(regras_data.get('regras', [])):
        for campo in required:
            if campo not in regra:
                erros.append(f"Regra {regra.get('id','?')} - campo '{campo}' faltando")
    
    if not erros:
        return jsonify({'ok': True, 'valido': True, 'mensagem': 'Estrutura valida'})
    return jsonify({'ok': True, 'valido': False, 'erros': erros})


# ─── DADOS RESUMO ──────────────────────────────────────────────────────────────

@app.route('/api/dados/resumo', methods=['GET'])
def dados_resumo():
    """Resumo dos dados de folha para contexto."""
    import pandas as pd
    
    try:
        df = pd.read_csv(DATA_FILE, delimiter=';', dtype=str, encoding='utf-8', nrows=1000)
        for col in df.columns:
            df[col] = df[col].str.strip().str.strip('"')
        
        situacoes = df['cod_situacao_funcional'].value_counts().to_dict()
        rubricas = df['dsc_rubrica'].nunique()
        cargos = df['cod_cargo'].nunique()
        
        return jsonify({
            'ok': True,
            'amostra': len(df),
            'situacoes_funcionais': situacoes,
            'rubricas_unicas': rubricas,
            'cargos_unicos': cargos,
        })
    except Exception as e:
        return jsonify({'ok': False, 'erro': str(e)}), 500



# ─── API DE ML ──────────────────────────────────────────────────────────────

@app.route('/api/ml/resumo', methods=['GET'])
def ml_resumo():
    """Retorna resumo de todos os métodos ML disponíveis."""
    import os, json
    from pathlib import Path
    
    methods = []
    for method in ['yoy', 'ajustado', 'temporal']:
        base = Path(__file__).parent.parent.parent / 'data' / f'baseline_results_{method}'
        resumo_file = base / f'resumo_{method}.json'
        
        if resumo_file.exists():
            methods.append({
                'method': method,
                'label': {
                    'yoy': 'YoY Year-over-Year',
                    'ajustado': 'CAGR Ajustado',
                    'temporal': 'Temporal (baseline)',
                }.get(method, method),
                'path': str(resumo_file),
            })
    
    return jsonify({'methods': methods})


@app.route('/api/ml/resultados', methods=['GET'])
def ml_resultados():
    """Retorna resultados consolidados de todos os métodos."""
    import pandas as pd
    from pathlib import Path
    
    cargo = request.args.get('cargo', 'P115')
    method = request.args.get('method', 'yoy')
    
    base = Path(__file__).parent.parent.parent / 'data' / f'baseline_results_{method}'
    csv_file = base / f'anomalias_{method}_{cargo}.csv'
    
    if not csv_file.exists():
        return jsonify({'ok': False, 'erro': f'Arquivo nao encontrado: {cargo}'}), 404
    
    df = pd.read_csv(csv_file, delimiter=';', encoding='utf-8')
    
    cols_rub = [c for c in df.columns if c not in ['isn_vinculo','num_ano','num_mes','IF_label','IF_score','anomalo']]
    
    n_total = len(df)
    n_treino = int(n_total * 0.8)
    df_treino = df.iloc[:n_treino]
    df_teste = df.iloc[n_treino:]
    
    if 'IF_score' in df.columns:
        df_sorted = df_teste.sort_values('IF_score', ascending=True)
        
        # Vínculos mais anômalos (score mais negativo)
        anom = df_sorted[df_sorted['anomalo'] == 1]
        
        if len(anom) > 0:
            agg_cols = [c for c in cols_rub if not c.endswith('_yoy')]
            vals = df_sorted[df_sorted['anomalo'] == 1][agg_cols].mean(axis=1).values
            top = anom.groupby('isn_vinculo').agg(
                qtd_meses=('anomalo', 'count'),
                score_medio=('IF_score', 'mean'),
            ).reset_index().sort_values('score_medio', ascending=True).head(20)
            top['score_medio'] = top['score_medio'].round(4)
            top = top.to_dict('records')
        else:
            top = []
        
        return jsonify({
            'ok': True,
            'cargo': cargo,
            'method': method,
            'n_treino': n_treino,
            'n_teste': len(df_teste),
            'periodo_treino': f"{int(df_treino.num_ano.min())}/{int(df_treino.num_mes.min())} - {int(df_treino.num_ano.max())}/{int(df_treino.num_mes.max())}",
            'periodo_teste': f"{int(df_teste.num_ano.min())}/{int(df_teste.num_mes.min())} - {int(df_teste.num_ano.max())}/{int(df_teste.num_mes.max())}",
            'pct_anomalias_treino': round(df_treino['anomalo'].mean() * 100, 1),
            'pct_anomalias_teste': round(df_teste['anomalo'].mean() * 100, 1),
            'top_vinculos_anomalos': top,
            'n_rubricas': len([c for c in cols_rub if not c.endswith('_yoy')]),
        })
    
    return jsonify({'ok': False, 'erro': 'Estrutura invalida'})


@app.route('/api/ml/cargos', methods=['GET'])
def ml_cargos():
    """Lista todos os cargos disponíveis com resultado ML."""
    from pathlib import Path
    
    base = Path(__file__).parent.parent.parent / 'data' / 'baseline_results_yoy'
    
    cargos = []
    for f in sorted(base.glob('anomalias_yoy_*.csv')):
        cargo = f.stem.replace('anomalias_yoy_', '')
        try:
            import pandas as pd
            n = len(pd.read_csv(f, delimiter=';', encoding='utf-8'))
            cargos.append({'cargo': cargo, 'n_registros': n, 'method': 'yoy'})
        except:
            pass
    
    return jsonify({'cargos': cargos})


@app.route('/api/ml/comparar', methods=['GET'])
def ml_comparar():
    """Compara todos os métodos para um cargo."""
    import pandas as pd
    from pathlib import Path
    
    cargo = request.args.get('cargo', 'P115')
    
    results = []
    for method in ['yoy', 'ajustado', 'temporal']:
        base = Path(__file__).parent.parent.parent / 'data' / f'baseline_results_{method}'
        csv_file = base / f'anomalias_{method}_{cargo}.csv'
        
        if not csv_file.exists():
            continue
        
        try:
            df = pd.read_csv(csv_file, delimiter=';', encoding='utf-8')
            n_total = len(df)
            n_treino = int(n_total * 0.8)
            df_teste = df.iloc[n_treino:]
            
            pct_teste = round(df_teste['anomalo'].mean() * 100, 1) if 'anomalo' in df_teste.columns else 0
            score_medio = round(df_teste['IF_score'].mean(), 4) if 'IF_score' in df_teste.columns else 0
            
            results.append({
                'method': method,
                'label': {'yoy': 'YoY', 'ajustado': 'CAGR', 'temporal': 'Sem Ajuste'}.get(method, method),
                'pct_anomalias_teste': pct_teste,
                'score_medio': score_medio,
            })
        except:
            pass
    
    return jsonify({'cargo': cargo, 'methods': results})


# ─── ML TEMPO REAL ──────────────────────────────────────────────────────────

@app.route('/api/ml/rodar', methods=['POST'])
def ml_rodar():
    """
    Inicia análise ML em background para um cargo.
    POST body: {"method": "yoy", "cargo": "P115"}
    Retorna: {"job_id": "abc123", "status": "running"}
    """
    body = request.json or {}
    method = body.get('method', 'yoy')
    cargo = body.get('cargo')
    
    if not cargo:
        return jsonify({'ok': False, 'erro': 'Cargo e obrigatorio'}), 400
    
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts'))
    from ml_inline import start_job
    
    job_id = start_job(method, cargo)
    return jsonify({'ok': True, 'job_id': job_id, 'method': method, 'cargo': cargo})


@app.route('/api/ml/status/<job_id>', methods=['GET'])
def ml_status(job_id):
    """Verifica status de um job de ML."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts'))
    from ml_inline import get_job
    
    job = get_job(job_id)
    if job.get('status') == 'unknown':
        return jsonify({'ok': False, 'erro': 'Job nao encontrado'}), 404
    return jsonify({'ok': True, **job})


@app.route('/api/ml/resultado/<job_id>', methods=['GET'])
def ml_resultado(job_id):
    """Retorna resultado de um job completo."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts'))
    from ml_inline import get_job
    
    job = get_job(job_id)
    if job.get('status') == 'unknown':
        return jsonify({'ok': False, 'erro': 'Job nao encontrado'}), 404
    
    if job.get('status') == 'running':
        return jsonify({'ok': True, 'status': 'running', 'job_id': job_id})
    
    if job.get('status') == 'error':
        return jsonify({'ok': False, 'status': 'error', 'erro': job.get('error')}), 500
    
    return jsonify({'ok': True, 'status': 'done', 'result': job.get('result')})


@app.route('/api/ml/carregar_cache', methods=['GET'])
def ml_carregar_cache():
    """Carrega resultado do cache ou roda se não existir."""
    method = request.args.get('method', 'yoy')
    cargo = request.args.get('cargo', 'P115')
    
    cache_file = Path(__file__).parent.parent.parent / 'data' / 'ml_cache' / f'{method}_{cargo}.json'
    
    if cache_file.exists():
        with open(cache_file) as f:
            return jsonify({'ok': True, 'source': 'cache', 'data': json.load(f)})
    
    # Não existe — iniciar job
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts'))
    from ml_inline import start_job
    
    job_id = start_job(method, cargo)
    return jsonify({'ok': True, 'source': 'compute', 'job_id': job_id, 'status': 'running'})

if __name__ == '__main__':
    print("=" * 50)
    print("  Admin de Regras - ConfereAI")
    print("  http://localhost:5001")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5001, debug=True)

# ─── API DE ML ──────────────────────────────────────────────────────────────

@app.route('/api/ml/resumo', methods=['GET'])
def ml_resumo():
    """Retorna resumo de todos os métodos ML disponíveis."""
    import os, json
    from pathlib import Path
    
    methods = []
    for method in ['yoy', 'ajustado', 'temporal']:
        base = Path(__file__).parent.parent.parent / 'data' / f'baseline_results_{method}'
        resumo_file = base / f'resumo_{method}.json'
        
        if resumo_file.exists():
            with open(resumo_file) as f:
                data = json.load(f)
            
            total_anomalias = sum(r.get('n_anomalias_teste', 0) for r in data if isinstance(data, list))
            methods.append({
                'method': method,
                'label': {
                    'yoy': 'YoY Year-over-Year',
                    'ajustado': 'CAGR Ajustado',
                    'temporal': 'Temporal (baseline)',
                }.get(method, method),
                'path': str(resumo_file),
                'data_ultima_analise': '2026-04-10',
            })
    
    return jsonify({'methods': methods})


@app.route('/api/ml/resultados', methods=['GET'])
def ml_resultados():
    """Retorna resultados consolidados de todos os métodos."""
    import os, json, pandas as pd
    from pathlib import Path
    
    cargo = request.args.get('cargo', 'P115')
    method = request.args.get('method', 'yoy')
    
    base = Path(__file__).parent.parent.parent / 'data' / f'baseline_results_{method}'
    csv_file = base / f'anomalias_{method}_{cargo}.csv'
    
    if not csv_file.exists():
        return jsonify({'ok': False, 'erro': f'Arquivo nao encontrado: {cargo}'}), 404
    
    df = pd.read_csv(csv_file, delimiter=';', encoding='utf-8')
    
    # Colunas de rubrica
    cols_rub = [c for c in df.columns if c not in ['isn_vinculo','num_ano','num_mes','IF_label','IF_score','anomalo']]
    cols_rub_original = [c for c in cols_rub if not c.endswith('_yoy')]
    cols_yoy = [c for c in cols_rub if c.endswith('_yoy')]
    
    n_total = len(df)
    n_treino = int(n_total * 0.8)
    df_treino = df.iloc[:n_treino]
    df_teste = df.iloc[n_treino:]
    
    # Top anomalias no teste (mais negativas = mais anômalas)
    df_teste_copy = df_teste.copy()
    if 'IF_score' in df_teste_copy.columns:
        df_teste_copy = df_teste_copy.sort_values('IF_score', ascending=True)
        
        # Vínculos mais anômalos
        top_anomalos = df_teste_copy[df_teste_copy['anomalo'] == 1].groupby('isn_vinculo').agg(
            qtd_meses=('anomalo', 'count'),
            score_medio=('IF_score', 'mean'),
            valor_total=('vlr_calculado' if 'vlr_calculado' in df_teste_copy.columns else cols_rub_original[0], 'sum')
        ).reset_index().sort_values('score_medio', ascending=True).head(20)
        
        # Período do teste
        periodo_treino = f"{int(df_treino.num_ano.min())}/{int(df_treino.num_mes.min())} a {int(df_treino.num_ano.max())}/{int(df_treino.num_mes.max())}"
        periodo_teste = f"{int(df_teste.num_ano.min())}/{int(df_teste.num_mes.min())} a {int(df_teste.num_ano.max())}/{int(df_teste.num_mes.max())}"
        
        return jsonify({
            'ok': True,
            'cargo': cargo,
            'method': method,
            'n_treino': n_treino,
            'n_teste': len(df_teste),
            'periodo_treino': periodo_treino,
            'periodo_teste': periodo_teste,
            'pct_anomalias_treino': round(df_treino['anomalo'].mean() * 100, 1),
            'pct_anomalias_teste': round(df_teste['anomalo'].mean() * 100, 1),
            'top_vinculos_anomalos': top_anomalos.to_dict('records'),
            'n_rubricas': len(cols_rub_original),
        })
    
    return jsonify({'ok': False, 'erro': 'Estrutura de dados invalida'})


@app.route('/api/ml/cargos', methods=['GET'])
def ml_cargos():
    """Lista todos os cargos disponíveis com resultado ML."""
    import pandas as pd
    from pathlib import Path
    
    base = Path(__file__).parent.parent.parent / 'data' / 'baseline_results_yoy'
    
    cargos = []
    for f in sorted(base.glob('anomalias_yoy_*.csv')):
        cargo = f.stem.replace('anomalias_yoy_', '')
        try:
            df = pd.read_csv(f, delimiter=';', encoding='utf-8', nrows=1)
            n = len(pd.read_csv(f, delimiter=';', encoding='utf-8'))
            cargos.append({
                'cargo': cargo,
                'n_registros': n,
                'method': 'yoy',
            })
        except:
            pass
    
    return jsonify({'cargos': cargos})


@app.route('/api/ml/comparar', methods=['GET'])
def ml_comparar():
    """Compara todos os métodos para um cargo."""
    import pandas as pd, json
    from pathlib import Path
    
    cargo = request.args.get('cargo', 'P115')
    
    results = []
    for method in ['yoy', 'ajustado', 'temporal']:
        base = Path(__file__).parent.parent.parent / 'data' / f'baseline_results_{method}'
        csv_file = base / f'anomalias_{method}_{cargo}.csv'
        resumo_file = base / f'resumo_{method}.json'
        
        if not csv_file.exists():
            continue
        
        try:
            df = pd.read_csv(csv_file, delimiter=';', encoding='utf-8')
            n_total = len(df)
            n_treino = int(n_total * 0.8)
            df_teste = df.iloc[n_treino:]
            
            pct_teste = df_teste['anomalo'].mean() * 100 if 'anomalo' in df_teste.columns else 0
            score_medio = df_teste['IF_score'].mean() if 'IF_score' in df_teste.columns else 0
            
            results.append({
                'method': method,
                'label': {'yoy': 'YoY', 'ajustado': 'CAGR', 'temporal': 'Sem Ajuste'}.get(method, method),
                'pct_anomalias_teste': round(pct_teste, 1),
                'score_medio': round(score_medio, 4),
            })
        except Exception as e:
            pass
    
    return jsonify({'cargo': cargo, 'methods': results})
