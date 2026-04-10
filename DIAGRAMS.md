# ConfereAI — Diagramas

## Arquitetura Geral

```mermaid
flowchart LR
    subgraph Dados
        A[historico_pagamento_15.csv]
    end
    
    subgraph Pré-processamento
        B[Consolidação de Rubricas]
        C[Normalização StandardScaler]
    end
    
    subgraph Métodos
        D[Isolation Forest]
        E[One-Class SVM]
        F[Autoencoder]
        G[YoY Year-over-Year]
        H[Ajuste CAGR]
    end
    
    subgraph Resultados
        I[Anomalias por Cargo]
        J[Métricas de Performance]
    end
    
    A --> B --> C
    C --> D
    C --> E
    C --> F
    C --> H --> G
    D --> I
    E --> I
    F --> I
    G --> I
    I --> J
```

## Pipeline de Detecção de Anomalias

```mermaid
flowchart TD
    A[Dados Brutos<br/>historico_pagamento_15.csv] --> B{Pré-processamento}
    
    B --> C[Consolidar rubricas por<br/>isn_vinculo + num_ano + num_mes]
    C --> D[Remover outliers extremos<br/>IQR 99.5%]
    D --> E[Divisão Temporal 80/20]
    
    E --> F[Dados Treino 80%]
    E --> G[Dados Teste 20%]
    
    F --> H[Normalizar<br/>StandardScaler]
    G --> H
    
    H --> I[Isolation Forest]
    H --> J[One-Class SVM]
    
    I --> K[Labels: -1=anômalo, 1=normal]
    J --> K
    
    K --> L[Comparar treino vs teste]
    L --> M{Anomalias no teste<br/>muito maiores?}
    
    M -->|Sim| N[Investigar padrões<br/>temporais]
    M -->|Não| O[Modelo estável]
    
    N --> P[Considerar YoY ou<br/>Ajuste CAGR]
```

## Divisão Temporal 80/20

```mermaid
gantt
    title Divisão Temporal dos Dados
    dateFormat  YYYY-MM
    axisFormat  %Y
    
    section Treino
    Dados 2007-2019    :done, t1, 2007-01, 2020-01
    
    section Teste
    Dados 2020-2024    :active, t2, 2020-01, 2025-01
```

## Comparação de Métodos

```mermaid
flowchart LR
    subgraph Métodos
        A[Sem Ajuste]
        B[Com CAGR]
        C[YoY]
    end
    
    subgraph Anomalias Detectadas
        A -->|~5%| D[Baseline]
        B -->|~3%| E[Deflacionado]
        C -->|~12%| F[Sensível]
    end
    
    subgraph Uso
        D -->|Uso geral| G[Baseline]
        E -->|Remover evolução<br/>de longo prazo| H[Anomalias reais]
        F -->|Capturar variações<br/>mensais| I[Análise fina]
    end
```

## Detecção de Anomalias por Cargo

```mermaid
flowchart TD
    A[Selecionar Cargo<br/>ex: P115] --> B[Dados do Cargo]
    
    B --> C{Pivot Table<br/>isn_vinculo × isn_rubrica}
    
    C --> D{Rubricas suficientes?<br/>min 30 registros}
    
    D -->|Não| E[跳过 cargo]
    D -->|Sim| F[Calcular CAGR por rubrica]
    
    F --> G{Usar método?}
    
    G -->|CAGR| H[Deflacionar valores<br/>para ano-base]
    G -->|YoY| I[valor_mês /<br/>valor_mesmo_mês_ano_anterior]
    G -->|Nenhum| J[Valores originais]
    
    H --> K[Normalizar StandardScaler]
    I --> K
    J --> K
    
    K --> L[Isolation Forest<br/>contamination=0.05]
    K --> M[One-Class SVM<br/>nu=0.05]
    
    L --> N[Score anomalias]
    M --> N
    
    N --> O[Salvar CSV<br/>anomalias_{cargo}.csv]
    O --> P[Próximo Cargo]
```

## CAGR — Compound Annual Growth Rate

```mermaid
flowchart LR
    A[Valor Inicial<br/>2007] -->|CAGR| B[Valor Final<br/>2024]
    
    B --> C["CAGR = (VF/VI)^(1/n) - 1"]
    
    D[VI: R$ 545] -->|17 anos| E[VF: R$ 1.590]
    D -->|CAGR| F[6.5% ao ano]
    
    G[VI: R$ 906] -->|17 anos| H[VF: R$ 3.077]
    G -->|CAGR| I[15.0% ao ano]
    
    style F fill:#90EE90
    style I fill:#FFB6C1
```

## YoY — Year-over-Year

```mermaid
sequenceDiagram
    participant Jan2023
    participant Jan2022
    participant Sistema
    
    Jan2023->>Sistema: Valor = R$ 150
    Jan2022->>Sistema: Valor = R$ 140
    
    Sistema->>Sistema: YoY = 150/140 = 1.07
    
    Note over Sistema: Aumento de 7% em 1 ano<br/>→ Normal (dentro do CAGR)
    
    participant Dez2023
    participant Dez2022
    participant Sistema2
    
    Dez2023->>Sistema2: Valor = R$ 5.000 (com 13º)
    Dez2022->>Sistema2: Valor = R$ 1.500 (sem 13º)
    
    Sistema2->>Sistema2: YoY = 5000/1500 = 3.33
    
    Note over Sistema2: Aumento de 233%<br/>→ FALSO POSITIVO (13º)
```

## Comparação Sem vs Com Ajuste

```mermaid
flowchart TD
    subgraph Sem Ajuste
        A1[Valor 2007: R$ 100] --> A2[Valor 2024: R$ 500]
        A2 --> A3[Isolation Forest]
        A3 --> A4[Anomalia detectada<br/>+400% parece suspeito]
    end
    
    subgraph Com CAGR
        B1[Valor 2007: R$ 100] --> B2[Deflacionado 2024: R$ 100]
        B2 --> B3[Isolation Forest]
        B3 --> B4[Normal<br/>sem anomalia]
    end
    
    subgraph Com YoY
        C1[Jan 2023: R$ 150] --> C2[Jan 2022: R$ 140]
        C2 --> C3[YoY = 1.07]
        C3 --> C4[Normal<br/>+7% dentro do esperado]
    end
```

## Autoencoder Architecture

```mermaid
flowchart LR
    subgraph Encoder
        A[Input: 15 rubricas] --> B[Dense: 64]
        B --> C[Dense: 32]
        C --> D[Latent: 8]
    end
    
    subgraph Decoder
        D --> E[Dense: 32]
        E --> F[Dense: 64]
        F --> G[Output: 15 rubricas]
    end
    
    G --> H[Erro de Reconstrução]
    H --> I{Erro > threshold?}
    
    I -->|Sim| J[ANOMALIA]
    I -->|Não| K[NORMAL]
```
