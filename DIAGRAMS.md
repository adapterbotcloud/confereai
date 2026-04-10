# ConfereAI — Diagramas

## Arquitetura Geral com Regras + ML

```mermaid
flowchart TD
    A[Dados Brutos<br/>historico_5_seplag.csv] --> B{Pré-processamento}
    
    B --> B1[Limpar dados<br/>dsc_rubrica + cod_situacao_funcional]
    B1 --> B2[Divisão Temporal 80/20]
    
    B2 --> C[Filtro por Regras<br/>regras_ativos.json]
    B2 --> D[ML: Isolation Forest<br/>Autoencoder]
    
    C --> C1[Regras FN Payroll Audit]
    D --> D1[Anomalias estatisticas]
    
    C1 --> E[Irregularidades por regra]
    D1 --> F[Anomalias por ML]
    
    E --> G[Consolidação]
    F --> G
    G --> H[Relatório final]
```

## Pipeline Completo — Dados → Resultado

```mermaid
flowchart LR
    subgraph Dados
        A[historico_5_seplag.csv<br/>163k registros<br/>2020-2025]
    end
    
    subgraph Camada 1 — Regras
        B[Admin Web<br/>regras.confereai.com]
        C[regras_ativos.json<br/>versionado git]
        D[Python: detectar_regras.py<br/>aplica regras nos dados]
    end
    
    subgraph Camada 2 — ML
        E[Filtro: remover<br/>irregularidades]
        F[Normalizar YoY<br/>StandardScaler]
        G[Isolation Forest<br/>contamination=0.05]
    end
    
    subgraph Resultado
        H[Irregularidades<br/>por regra]
        I[Anomalias<br/>por ML]
        J[Consolidado<br/>Geral]
    end
    
    A --> B --> C --> D
    D --> E
    E --> F --> G
    A --> D
    D --> H
    G --> I
    H --> J
    I --> J
```

## Admin de Regras — Interface

```mermaid
flowchart LR
    A[Auditor<br/>abre admin web] --> B[Interface HTML<br/>localhost:5001]
    B --> C{Regra existente?}
    C -->|Nova| D[Preenche formulario]
    C -->|Editar| E[Modifica regra]
    D --> F[Salvar]
    E --> F
    F --> G[API Flask<br/>POST /api/regras]
    G --> H[Atualiza<br/>regras_ativos.json]
    H --> I[Git commit<br/>automatico]
    I --> J[Pipeline ConfereAI<br/>lê JSON]
    J --> K[Testa regras<br/>contra dados]
    K --> L[Resultado]
```

## Regras — Tipos de Condição

```mermaid
flowchart TD
    A[Criação de Regra] --> B{Selecionar Situação Funcional}
    
    B -->|0| B1[Civil Ativo]
    B -->|2| B2[Civil Afastado c/ ônus]
    B -->|6| B3[Pensionista]
    B -->|7| B4[Pensão Alimento]
    
    B1 --> C{Rubrica CONTÉM?}
    B2 --> C
    B3 --> C
    B4 --> C
    
    C -->|VENCIMENTO| D1[ALERTA: rubrica errada]
    C -->|PROVENTO| D2[ALERTA: provento em ativo?]
    C -->|PENSAO| D3[OK: pensionista correto]
    C -->|GRAT TEMPO| D4[ALERTA: gratificacao]
    
    style D1 fill:#ef5350
    style D2 fill:#ffa726
    style D3 fill:#66bb6a
    style D4 fill:#ef5350
```

## Divisão Temporal 80/20

```mermaid
gantt
    title Corte Temporal dos Dados
    dateFormat  YYYY-MM
    axisFormat  %Y
    
    section Treino
    Dados 2020-01 a 2023-12    :done, t1, 2020-01, 2024-01
    
    section Teste
    Dados 2024-01 a 2025-01    :active, t2, 2024-01, 2025-01
```

## Comparação de Métodos

```mermaid
flowchart LR
    subgraph Regras
        A[FN Payroll Audit<br/>baseado em norma CGU]
    end
    
    subgraph ML
        B[Isolation Forest<br/>estatistico]
        C[Autoencoder<br/>reconstrução]
        D[YoY Year-over-Year<br/>mensal]
    end
    
    subgraph Saída
        E[Irregularidade<br/>por regra]
        F[Anomalia<br/>por ML]
        G[Consolidado]
    end
    
    A --> E
    B --> F
    C --> F
    D --> F
    E --> G
    F --> G
```

## Detecção por Cargo

```mermaid
flowchart TD
    A[Selecionar Cargo<br/>ex: Z120] --> B[Dados do Cargo]
    
    B --> C{Pivot Table<br/>isn_vinculo × isn_rubrica}
    
    C --> D{Rubricas suficientes?<br/>min 30 registros}
    
    D -->|Não| E[跳过 cargo]
    D -->|Sim| F[Divisão Temporal 80/20]
    
    F --> G[Dados Treino 80%]
    F --> H[Dados Teste 20%]
    
    G --> I[Filtro Regras<br/>remove irregulares]
    I --> J[Normalizar YoY<br/>StandardScaler]
    
    H --> J
    
    J --> K[Isolation Forest<br/>contamination=0.05]
    K --> L[Score anomalias]
    
    L --> M[Comparar treino vs teste]
    M --> N{Anomalias no teste<br/>muito maiores?}
    
    N -->|Sim| O[Investigar padrões<br/>temporais]
    N -->|Não| P[Modelo estável]
```

## Estrutura de Arquivos

```mermaid
filesystem
    
    .confereai/
    📁 regras/
    ├── 📄 regras_ativos.json    # Regras ativas (versionado git)
    ├── 📄 REGRAS.md             # Documentação
    └── 📁 admin/
        ├── 📄 app.py            # API Flask (:5001)
        └── 📁 static/
            └── 📄 index.html   # Admin web (dark theme)
    
    📁 data/
    ├── 📄 historico_5_seplag.csv
    ├── 📁 regras_resultados/
    │   ├── 📄 todas_violacoes.csv
    │   └── 📄 resumo_regras.json
    └── 📁 baseline_results_yoy/
    
    📁 scripts/
    ├── 📄 deteccao_regras.py
    └── 📄 baseline_yoy.py
```
