#!/usr/bin/env python3
"""
Admin de Regras do ConfereAI — FastAPI
Roda em localhost:5001

Autenticacao via token signed (HMAC-SHA256).
O token e enviado no header Authorization: Bearer <token>.
"""

import json
import secrets
from pathlib import Path
from datetime import datetime, timedelta

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

# ─── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="ConfereAI Admin", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Auth ──────────────────────────────────────────────────────────────────────
SECRET_KEY = secrets.token_hex(32)
TOKEN_MAX_AGE = 60 * 60 * 24  # 24h
serializer = URLSafeTimedSerializer(SECRET_KEY)

VALID_USER = "admin"
VALID_PASS = "123admin#"

# In-memory invalidation (set to token when logout)
invalidated_tokens: set = set()


def create_token() -> str:
    return serializer.dumps(VALID_USER)


def verify_token(token: str) -> bool:
    if token in invalidated_tokens:
        return False
    try:
        data = serializer.loads(token, max_age=TOKEN_MAX_AGE)
        return data == VALID_USER
    except (BadSignature, SignatureExpired):
        return False


def get_token(request: Request) -> str:
    """Extract token from Authorization: Bearer <token> header."""
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return ""


def require_auth(request: Request) -> None:
    """Dependency: raises 401 if not authenticated."""
    token = get_token(request)
    if not token or not verify_token(token):
        raise HTTPException(status_code=401, detail="Nao autenticado")


# ─── Files ─────────────────────────────────────────────────────────────────────
REGRAS_FILE = Path(__file__).parent.parent / "regras_ativos.json"
DATA_FILE = Path(__file__).parent.parent.parent / "data" / "historico_5_seplag.csv"


def carregar_regras():
    with open(REGRAS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def salvar_regras(data):
    with open(REGRAS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ─── Rotas Estaticas ───────────────────────────────────────────────────────────
@app.get("/")
async def index():
    return FileResponse("static/index.html")


# ─── Auth ─────────────────────────────────────────────────────────────────────
@app.post("/api/login")
async def login(body: dict):
    user = body.get("user", "")
    pw = body.get("pass", "")
    if user == VALID_USER and pw == VALID_PASS:
        token = create_token()
        return {"ok": True, "user": user, "token": token}
    return JSONResponse({"ok": False, "erro": "Credenciais invalidas"}, status_code=401)


@app.post("/api/logout")
async def logout(request: Request):
    token = get_token(request)
    if token:
        invalidated_tokens.add(token)
    return {"ok": True}


@app.get("/api/me")
async def me(request: Request):
    token = get_token(request)
    if token and verify_token(token):
        return {"user": VALID_USER, "authenticated": True}
    return {"authenticated": False}


# ─── API de Regras ─────────────────────────────────────────────────────────────
@app.get("/api/regras")
async def listar_regras(request: Request):
    require_auth(request)
    data = carregar_regras()
    return data


@app.post("/api/regras")
async def adicionar_regra(request: Request, body: dict):
    require_auth(request)
    data = carregar_regras()
    existentes = [int(r["id"][1:]) for r in data["regras"] if r["id"].startswith("R")]
    novo_id = max(existentes) + 1 if existentes else 1
    body["id"] = f"R{novo_id:03d}"
    data["regras"].append(body)
    data["_ultima_alteracao"] = "2026-04-10"
    salvar_regras(data)
    return {"ok": True, "regra": body}


@app.put("/api/regras/{regra_id}")
async def atualizar_regra(regra_id: str, request: Request, body: dict):
    require_auth(request)
    data = carregar_regras()
    for i, r in enumerate(data["regras"]):
        if r["id"] == regra_id:
            body["id"] = regra_id
            data["regras"][i] = body
            data["_ultima_alteracao"] = "2026-04-10"
            salvar_regras(data)
            return {"ok": True, "regra": body}
    return JSONResponse({"ok": False, "erro": "Regra nao encontrada"}, status_code=404)


@app.delete("/api/regras/{regra_id}")
async def remover_regra(regra_id: str, request: Request):
    require_auth(request)
    data = carregar_regras()
    antes = len(data["regras"])
    data["regras"] = [r for r in data["regras"] if r["id"] != regra_id]
    if len(data["regras"]) == antes:
        return JSONResponse({"ok": False, "erro": "Regra nao encontrada"}, status_code=404)
    data["_ultima_alteracao"] = "2026-04-10"
    salvar_regras(data)
    return {"ok": True}


# ─── Testar / Validar ──────────────────────────────────────────────────────────
@app.post("/api/regras/testar")
async def testar_regras(request: Request):
    require_auth(request)
    import pandas as pd

    try:
        df = pd.read_csv(DATA_FILE, delimiter=";", dtype=str, encoding="utf-8")
        for col in df.columns:
            df[col] = df[col].str.strip().str.strip('"')
        df["vlr_calculado"] = pd.to_numeric(df["vlr_calculado"], errors="coerce")
    except Exception as e:
        return JSONResponse({"ok": False, "erro": f"Erro ao carregar dados: {e}"}, status_code=500)

    regras_data = carregar_regras()
    todas_violacoes = []

    SITUACAO_MAP = {
        "0": "Civil Ativo", "1": "Militar Ativo",
        "2": "Civil Afastado c/ onus", "3": "Militar Afastado c/ onus",
        "4": "Civil Afastado", "5": "Militar Afastado",
        "6": "Pensionista", "7": "Pensao Alimento", "8": "Liminar",
    }
    df["dsc_situacao"] = df["cod_situacao_funcional"].map(SITUACAO_MAP).fillna(df["cod_situacao_funcional"])

    for regra in regras_data.get("regras", []):
        if regra.get("status") != "ativa":
            continue
        situacoes = regra.get("situacao_funcional", [])
        rubricas_contem = regra.get("rubrica_contem", [])
        rubricas_nao_contem = regra.get("rubrica_nao_contem", [])

        mask = df["cod_situacao_funcional"].isin(situacoes) if situacoes else pd.Series(True, index=df.index)

        if rubricas_contem:
            mask_rub = df["dsc_rubrica"].apply(
                lambda x: pd.notna(x) and any(t.upper() in str(x).upper() for t in rubricas_contem)
            )
            mask = mask & mask_rub

        if rubricas_nao_contem:
            mask_nao = df["dsc_rubrica"].apply(
                lambda x: pd.notna(x) and not any(t.upper() in str(x).upper() for t in rubricas_nao_contem)
            )
            mask = mask & mask_nao

        violacoes = df[mask]
        if len(violacoes) > 0:
            grupo = violacoes.groupby(["isn_vinculo", "dsc_rubrica", "dsc_situacao"])["vlr_calculado"].agg(["count", "sum"]).reset_index()
            grupo["regra_id"] = regra["id"]
            grupo["regra_nome"] = regra["nome"]
            grupo.columns = ["isn_vinculo", "dsc_rubrica", "dsc_situacao", "qtd", "vlr_total", "regra_id", "regra_nome"]
            todas_violacoes.append(grupo)

    if todas_violacoes:
        resultado = pd.concat(todas_violacoes, ignore_index=True)
        resultado = resultado.sort_values("vlr_total", ascending=False)
        return {
            "ok": True,
            "total_vinculos": int(resultado["isn_vinculo"].nunique()),
            "total_violacoes": int(len(resultado)),
            "valor_total": float(resultado["vlr_total"].sum()),
            "por_regra": resultado.groupby("regra_id").agg(
                qtd=("qtd", "sum"), vlr_total=("vlr_total", "sum"), regra_nome=("regra_nome", "first")
            ).to_dict("records"),
            "detalhes": resultado.head(50).to_dict("records"),
        }
    return {"ok": True, "total_vinculos": 0, "total_violacoes": 0, "valor_total": 0, "por_regra": [], "detalhes": []}


@app.get("/api/regras/validar")
async def validar_regras(request: Request):
    require_auth(request)
    regras_data = carregar_regras()
    erros = []
    required = ["id", "nome", "descricao", "situacao_funcional", "rubrica_contem", "rubrica_nao_contem", "acao", "severidade", "status"]
    for regra in regras_data.get("regras", []):
        for campo in required:
            if campo not in regra:
                erros.append(f"Regra {regra.get('id','?')} - campo '{campo}' faltando")
    if not erros:
        return {"ok": True, "valido": True, "mensagem": "Estrutura valida"}
    return {"ok": True, "valido": False, "erros": erros}


# ─── Dados Resumo ─────────────────────────────────────────────────────────────
@app.get("/api/dados/resumo")
async def dados_resumo(request: Request):
    require_auth(request)
    import pandas as pd
    try:
        df = pd.read_csv(DATA_FILE, delimiter=";", dtype=str, encoding="utf-8", nrows=1000)
        for col in df.columns:
            df[col] = df[col].str.strip().str.strip('"')
        situacoes = df["cod_situacao_funcional"].value_counts().to_dict()
        return {
            "ok": True,
            "amostra": len(df),
            "situacoes_funcionais": situacoes,
            "rubricas_unicas": int(df["dsc_rubrica"].nunique()),
            "cargos_unicos": int(df["cod_cargo"].nunique()),
        }
    except Exception as e:
        return JSONResponse({"ok": False, "erro": str(e)}, status_code=500)


# ─── ML Inline ────────────────────────────────────────────────────────────────
@app.get("/api/ml/cargos")
async def ml_cargos(request: Request):
    require_auth(request)
    import pandas as pd
    df = pd.read_csv(DATA_FILE, delimiter=";", dtype=str)
    for col in df.columns:
        df[col] = df[col].str.strip().str.strip('"')
    counts = df.groupby("cod_cargo").size().reset_index(name="n_registros")
    counts = counts.sort_values("n_registros", ascending=False)
    return {
        "cargos": [
            {"cargo": row["cod_cargo"], "n_registros": int(row["n_registros"]), "method": "yoy"}
            for _, row in counts.iterrows()
        ]
    }


@app.get("/api/ml/carregar_cache")
async def ml_carregar_cache(request: Request, method: str = "yoy", cargo: str = "P115"):
    require_auth(request)
    from pathlib import Path
    cache_file = Path(__file__).parent.parent.parent / "data" / "ml_cache" / f"{method}_{cargo}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return {"ok": True, "source": "cache", "data": json.load(f)}

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
    from ml_inline import start_job
    job_id = start_job(method, cargo)
    return {"ok": True, "source": "compute", "job_id": job_id, "status": "running"}


@app.get("/api/ml/comparar")
async def ml_comparar(request: Request, cargo: str = "P115"):
    require_auth(request)
    import pandas as pd
    from pathlib import Path

    results = []
    for method in ["yoy", "ajustado", "temporal"]:
        base = Path(__file__).parent.parent.parent / "data" / f"baseline_results_{method}"
        csv_file = base / f"anomalias_{method}_{cargo}.csv"
        if not csv_file.exists():
            continue
        try:
            df = pd.read_csv(csv_file, delimiter=";", encoding="utf-8")
            n_total = len(df)
            n_treino = int(n_total * 0.8)
            df_teste = df.iloc[n_treino:]
            pct_teste = round(df_teste["anomalo"].mean() * 100, 1) if "anomalo" in df_teste.columns else 0
            score_medio = round(df_teste["IF_score"].mean(), 4) if "IF_score" in df_teste.columns else 0
            results.append({
                "method": method,
                "label": {"yoy": "YoY", "ajustado": "CAGR", "temporal": "Sem Ajuste"}.get(method, method),
                "pct_anomalias_teste": pct_teste,
                "score_medio": score_medio,
            })
        except Exception:
            pass
    return {"cargo": cargo, "methods": results}


@app.post("/api/ml/rodar")
async def ml_rodar(request: Request):
    require_auth(request)
    body = await request.json()
    method = body.get("method", "yoy")
    cargo = body.get("cargo")
    if not cargo:
        return JSONResponse({"ok": False, "erro": "Cargo obrigatorio"}, status_code=400)

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
    from ml_inline import start_job
    job_id = start_job(method, cargo)
    return {"ok": True, "job_id": job_id, "method": method, "cargo": cargo}


@app.get("/api/ml/status/{job_id}")
async def ml_status(job_id: str, request: Request):
    require_auth(request)
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
    from ml_inline import get_job
    job = get_job(job_id)
    if job.get("status") == "unknown":
        return JSONResponse({"ok": False, "erro": "Job nao encontrado"}, status_code=404)
    return {"ok": True, **job}


@app.get("/api/ml/resultado/{job_id}")
async def ml_resultado(job_id: str, request: Request):
    require_auth(request)
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
    from ml_inline import get_job
    job = get_job(job_id)
    if job.get("status") == "unknown":
        return JSONResponse({"ok": False, "erro": "Job nao encontrado"}, status_code=404)
    if job.get("status") == "running":
        return {"ok": True, "status": "running", "job_id": job_id}
    if job.get("status") == "error":
        return JSONResponse({"ok": False, "status": "error", "erro": job.get("error")}, status_code=500)
    return {"ok": True, "status": "done", "result": job.get("result")}


# ─── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("  Admin de Regras - ConfereAI (FastAPI)")
    print("  http://localhost:5001")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=5001, log_level="warning")
