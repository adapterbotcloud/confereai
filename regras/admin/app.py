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


if __name__ == '__main__':
    print("=" * 50)
    print("  Admin de Regras - ConfereAI")
    print("  http://localhost:5001")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5001, debug=True)
