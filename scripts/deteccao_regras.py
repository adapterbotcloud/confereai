#!/usr/bin/env python3
"""
Regras de Auditoria baseadas no FN Payroll Audit
para detecção de irregularidades na folha de pagamento.

Regras implementadas:
  Regra 1: Pensionista não pode receber VENCIMENTO/SALARIO/PROVENTO
  Regra 2: Pensionista não pode receber Gratificações de produtividade
  Regra 3: Vínculo Afastado c/ ônus não pode receber certain rubricas
  Regra 4: Pensão Alimento recebe só pensão
"""

import pandas as pd
import json
from pathlib import Path

# ─── MAPEAMENTOS ───────────────────────────────────────────────────────────────

SITUACAO = {
    '0': 'Civil Ativo',
    '1': 'Militar Ativo',
    '2': 'Civil Afastado c/ ônus',
    '3': 'Militar Afastado c/ ônus',
    '4': 'Civil Afastado',
    '5': 'Militar Afastado',
    '6': 'Pensionista',
    '7': 'Pensão Alimento',
    '8': 'Liminar',
}

OPCAO_VENC = {
    '0': 'Carreira',
    '1': 'Cargo Comissionado',
}

# Rubricas que pensionista NUNCA deveria receber
RUBRICAS_PROIBIDAS_PENSIONISTA = [
    'VENCIMENTO/SALARIO/PROVENTO',
    'PROVENTO',  # pensionista tem provento próprio via pensão
    'GRAT DESEMP DE ANALISE DE GESTAO',
    'GRAT DESEMP DE ATIV PLANEJ ORCAMEN',
    'GRAT INCENTIVO PROFISSIONAL',
    'GRATIFICACAO DE EXERCICIO',
    'GRATIFICACAO DE REPRESENTACAO',
    'GRATIFICACAO DE REPRESENTACAO INCORPORADA',
    'GRAT DE NIVEL UNIVERSITARIO',
    'GRATIFICACAO POR TEMPO DE SERVICO',
    'GRATIFICACAO POR TITULACAO',
    'ABONO DE PERMANENCIA',
    'ADICIONAL DE FERIAS REGULAMENTADAS',
    'ADICIONAL DE PERICULOSIDADE',
    'ADICIONAL DE INSALUBRIDADE',
]

# Rubricas que só servidor ATIVO deveria receber
RUBRICAS_SO_ATIVO = [
    'GRAT DESEMP DE ANALISE DE GESTAO',
    'GRAT DESEMP DE ATIV PLANEJ ORCAMEN',
    'GRAT INCENTIVO PROFISSIONAL',
    'GRATIFICACAO DE EXERCICIO',
    'GRAT DE NIVEL UNIVERSITARIO',
    'ABONO DE PERMANENCIA',
]


def carregar_dados(caminho_csv):
    """Carrega e limpa os dados do CSV."""
    df = pd.read_csv(caminho_csv, delimiter=';', dtype=str, encoding='utf-8')
    
    # Limpar aspas e espaços
    for col in df.columns:
        df[col] = df[col].str.strip().str.strip('"')
    
    # Tipos
    df['vlr_calculado'] = pd.to_numeric(df['vlr_calculado'], errors='coerce')
    df['isn_rubrica'] = pd.to_numeric(df['isn_rubrica'], errors='coerce').astype('Int64')
    df['num_mes'] = pd.to_numeric(df['num_mes'], errors='coerce').astype('Int64')
    df['num_ano'] = pd.to_numeric(df['num_ano'], errors='coerce').astype('Int64')
    
    # Descrições
    df['dsc_situacao'] = df['cod_situacao_funcional'].map(SITUACAO).fillna(df['cod_situacao_funcional'])
    df['dsc_opcao'] = df['cod_opcao_vencimento'].map(OPCAO_VENC).fillna(df['cod_opcao_vencimento'])
    
    return df


def aplicar_regra_1(df):
    """
    Regra 1: Pensionista não pode receber VENCIMENTO/SALARIO/PROVENTO
    nem gratificações de produtividade.
    """
    mask = (
        (df['dsc_situacao'] == 'Pensionista') &
        (df['dsc_rubrica'].isin(RUBRICAS_PROIBIDAS_PENSIONISTA))
    )
    
    violacoes = df[mask].copy()
    violacoes['regra'] = 'R1 - Pensionista c/ rubrica proibida'
    violacoes['vlr_irregular'] = violacoes['vlr_calculado']
    
    return violacoes, mask


def aplicar_regra_2(df):
    """
    Regra 2: Pensionista não pode receber 'GRATIFICACAO POR TEMPO DE SERVICO'
    (é rubrica de servidor ativo).
    """
    mask = (
        (df['dsc_situacao'] == 'Pensionista') &
        (df['dsc_rubrica'].str.contains('GRATIFICACAO POR TEMPO DE SERVICO', na=False))
    )
    
    violacoes = df[mask].copy()
    violacoes['regra'] = 'R2 - Pensionista c/ GRAT TEMPO SERVICO'
    violacoes['vlr_irregular'] = violacoes['vlr_calculado']
    
    return violacoes, mask


def aplicar_regra_3(df):
    """
    Regra 3: Pensão Alimento recebe apenas pensões,
    nunca VENCIMENTO ou Gratificações.
    """
    mask = (
        (df['dsc_situacao'] == 'Pensão Alimento') &
        (~df['dsc_rubrica'].str.contains('PENSAO', na=False))
    )
    
    violacoes = df[mask].copy()
    violacoes['regra'] = 'R3 - Pensão Alimento c/ rubrica não-pensão'
    violacoes['vlr_irregular'] = violacoes['vlr_calculado']
    
    return violacoes, mask


def aplicar_regra_4(df):
    """
    Regra 4: Servidor Afastado c/ ônus pode receber proventos e algumas
    gratificações, mas NÃO deveria receber 'VENCIMENTO/SALARIO/PROVENTO'
    (deveria receber 'PROVENTO', não 'VENCIMENTO').

    Esta regra marca quando afastado recebe VENCIMENTO (errado)
    ao invés de PROVENTO (correto).
    """
    mask = (
        (df['dsc_situacao'] == 'Civil Afastado c/ ônus') &
        (df['dsc_rubrica'] == 'VENCIMENTO/SALARIO/PROVENTO')
    )
    
    violacoes = df[mask].copy()
    violacoes['regra'] = 'R4 - Afastado c/ ônus c/ VENCIMENTO (era pra ser PROVENTO)'
    violacoes['vlr_irregular'] = violacoes['vlr_calculado']
    
    return violacoes, mask


def aplicar_regra_5(df):
    """
    Regra 5: Servidor Ativo não deveria ter 'PROVENTO' no nome da rubrica
    (Provento é para inativos/pensionistas).
    """
    mask = (
        (df['dsc_situacao'] == 'Civil Ativo') &
        (df['dsc_rubrica'].str.contains('PROVENTO', na=False))
    )
    
    violacoes = df[mask].copy()
    violacoes['regra'] = 'R5 - Civil Ativo c/ PROVENTO'
    violacoes['vlr_irregular'] = violacoes['vlr_calculado']
    
    return violacoes, mask


def detectar_irregularidades(df, output_dir='data/regras_resultados'):
    """Executa todas as regras e consolida os resultados."""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    todas_violacoes = []
    resumo = {}
    
    regras = {
        'R1_Pensionista_Rubrica_Proibida': aplicar_regra_1,
        'R2_Pensionista_Grat_Tempo_Servico': aplicar_regra_2,
        'R3_Pensao_Alimento_Nao_Pensao': aplicar_regra_3,
        'R4_Afastado_Vencimento_Indevido': aplicar_regra_4,
        'R5_Civil_Ativo_Provento': aplicar_regra_5,
    }
    
    for nome, func in regras.items():
        violacoes, mask = func(df)
        
        if len(violacoes) > 0:
            # Salvar CSV por regra
            csv_path = f'{output_dir}/violacoes_{nome}.csv'
            violacoes.to_csv(csv_path, sep=';', index=False, encoding='utf-8')
            
            # Salvar no consolidado
            todas_violacoes.append(violacoes)
            
            # Resumo
            total_vlr = violacoes['vlr_irregular'].sum()
            qtd = len(violacoes)
            resumo[nome] = {
                'quantidade': qtd,
                'valor_total': round(total_vlr, 2),
                'viculos_unicos': violacoes['isn_vinculo'].nunique(),
                'rubricas_envolvidas': violacoes['dsc_rubrica'].unique().tolist(),
                'csv': csv_path,
            }
    
    # Consolidado geral
    if todas_violacoes:
        resultado = pd.concat(todas_violacoes, ignore_index=True)
        resultado.to_csv(f'{output_dir}/todas_violacoes.csv', sep=';', index=False, encoding='utf-8')
    else:
        resultado = pd.DataFrame()
    
    # Resumo JSON
    with open(f'{output_dir}/resumo_regras.json', 'w', encoding='utf-8') as f:
        json.dump({
            'total_violacoes': int(resultado['isn_vinculo'].count()) if len(resultado) else 0,
            'valor_total_irregular': round(resultado['vlr_irregular'].sum(), 2) if len(resultado) else 0,
            'regras': resumo,
        }, f, ensure_ascii=False, indent=2)
    
    return resultado, resumo


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Detecção de irregularidades por regras')
    parser.add_argument('--input', default='data/historico_5_seplag.csv', help='CSV de entrada')
    parser.add_argument('--output', default='data/regras_resultados', help='Diretório de saída')
    args = parser.parse_args()
    
    print("=" * 70)
    print("  DETECÇÃO DE IRREGULARIDADES — REGRAS FN PAYROLL AUDIT")
    print("=" * 70)
    
    df = carregar_dados(args.input)
    print(f"\nDados: {len(df):,} registros | {df['num_ano'].min()}-{df['num_ano'].max()}")
    print(f"Cargos: {df['cod_cargo'].nunique()} | Rubricas: {df['isn_rubrica'].nunique()}")
    print(f"Vínculos únicos: {df['isn_vinculo'].nunique()}")
    
    print("\n" + "=" * 70)
    print("  EXECUTANDO REGRAS")
    print("=" * 70)
    
    resultado, resumo = detectar_irregularidades(df, args.output)
    
    # Relatório
    print(f"\n{'REGRA':<45} | {'QTD':>6} | {'VALOR TOTAL':>15}")
    print("-" * 72)
    
    for nome, dados in resumo.items():
        print(f"  {nome:<43} | {dados['quantidade']:>6} | R$ {dados['valor_total']:>12,.2f}")
    
    total_vlr = sum(d['valor_total'] for d in resumo.values())
    total_qtd = sum(d['quantidade'] for d in resumo.values())
    
    print("-" * 72)
    print(f"  {'TOTAL IRREGULARIDADES':<43} | {total_qtd:>6} | R$ {total_vlr:>12,.2f}")
    print("=" * 70)
    print(f"\n✅ Результат: {args.output}/todas_violacoes.csv")
    print(f"✅ Resumo: {args.output}/resumo_regras.json")


if __name__ == '__main__':
    main()
