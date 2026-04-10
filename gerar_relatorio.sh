#!/bin/bash
cd /root/confereai

python3 << 'EOF'
import pandas as pd
import numpy as np
import os
import glob

output_dir = 'data/baseline_results'
csv_files = sorted(glob.glob(os.path.join(output_dir, 'anomalias_baseline_*.csv')))

todas_anomalias = []

for fpath in csv_files:
    fname = os.path.basename(fpath)
    cargo = fname.replace('anomalias_baseline_', '').replace('.csv', '')
    df = pd.read_csv(fpath, delimiter=';')

    # Só anomalo = 2 (ambos métodos concordam)
    df_anom = df[df['anomalo'] == 2].copy()
    if len(df_anom) == 0:
        continue

    df_anom['cargo'] = cargo
    df_anom['abs_score'] = df_anom['IF_score'].abs()

    # Identificar rubricas com valor != 0
    cols_rubricas = [c for c in df.columns if c not in ['isn_vinculo','num_ano','num_mes','IF_label','IF_score','OCSVM_label','OCSVM_score','anomalo','cargo','abs_score']]

    # Pra cada anomalia, mostrar rubricas que têm valor
    def get_rubricas_nao_zero(row):
        vals = []
        for c in cols_rubricas:
            if row[c] != 0:
                vals.append(f"{c}:{row[c]:.2f}")
        return '; '.join(vals)

    df_anom['rubricas_com_valor'] = df_anom.apply(get_rubricas_nao_zero, axis=1)

    todas_anomalias.append(df_anom[['cargo','isn_vinculo','num_ano','num_mes','IF_score','OCSVM_score','abs_score','rubricas_com_valor']])

if not todas_anomalias:
    print("Nenhuma anomalia com concordância de ambos os métodos (anomalo=2)")
else:
    consolidado = pd.concat(todas_anomalias, ignore_index=True)
    consolidado = consolidado.sort_values('abs_score', ascending=True).head(50)

    print("=" * 120)
    print("  RELATÓRIO — ANOMALIAS COM CONCORDÂNCIA AMPLA (anomalo=2)")
    print("  Ordenado por: quanto mais negativo o score, mais anômalo")
    print("=" * 120)
    print()

    for _, row in consolidado.iterrows():
        score_if = row['IF_score']
        score_svm = row['OCSVM_score']
        print(f"  🐐 Cargo: {row['cargo']} | Vínculo: {row['isn_vinculo']} | Período: {row['num_ano']}-{row['num_mes']:02d}")
        print(f"     IF score: {score_if:.6f} | SVM score: {score_svm:.6f}")
        print(f"     Rubricas com valor: {row['rubricas_com_valor']}")
        print()

    print("=" * 120)
    print(f"  Total de registros com anomalo=2: {len(consolidado)} (top 50 mais suspeitos)")
    print()

    # Resumo por cargo
    print("  RESUMO POR CARGO (anomalo=2)")
    print("  " + "-" * 50)
    por_cargo = consolidado.groupby('cargo').size().sort_values(ascending=False)
    for cargo, qtd in por_cargo.items():
        print(f"  {cargo:<10}: {qtd} anomalias")

EOF
