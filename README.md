# ConfereAI

Sistema de detecção de anomalias em folha de pagamento do funcionalismo público usando machine learning.

## Métodos

| Método | Descrição |
|--------|-----------|
| **Isolation Forest** | Detecta anomalias por isolamento em espaço n-dimensional |
| **One-Class SVM** | Aprende fronteira de normalidade |
| **Autoencoder** | Detecta reconstruções com alto erro |
| **YoY (Year-over-Year)** | Compara valor mensal com mesmo mês do ano anterior |
| **CAGR** | Remove evolução de longo prazo via taxa composta |

## Estrutura

```
confereai/
├── scripts/
│   ├── baseline_isolation_forest.py   # Baseline IF + OCSVM
│   ├── baseline_ajustado.py           # IF com ajuste CAGR
│   ├── baseline_yoy.py                # IF com normalização YoY
│   ├── analisar_crescimento.py        # Análise de CAGR por rubrica
│   └── treinar_autoencoder_paralelo.py
├── data/
│   └── baseline_results_yoy/          # Resultados por cargo
├── models/                            # Modelos treinados
└── plots/                             # Visualizações
```

## Uso

```bash
# Baseline simples
./run_baseline.sh

# Baseline com ajuste CAGR
./run_ajustado.sh

# Baseline com YoY
./run_yoy.sh
```

## Resultados

Resultados por cargo salvos em `data/baseline_results_{metodo}/anomalias_{metodo}_{cargo}.csv`

## Análise de CAGR

```bash
python scripts/analisar_crescimento.py
```

Calcula Compound Annual Growth Rate por rubrica dentro de cada cargo.

## Referências

- FN Payroll Audit (sistema similar baseado em regras fixas)
- Isolation Forest: Liu et al. (2008)
- Autoencoder para detecção de anomalias: Zhou & Paffenroth (2017)
