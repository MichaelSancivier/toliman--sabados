# -*- coding: utf-8 -*-
import io
import unicodedata
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Configuração da página
# -----------------------------
st.set_page_config(
    page_title="Conferência de Configuração por Placa",
    layout="wide",
)

st.title("🧭 Conferência de Configuração por Placa")
st.caption("Mostra as **placas** que estão **diferentes** da configuração solicitada pelo cliente (sem checar tempo de logoff).")

# -----------------------------
# Funções auxiliares
# -----------------------------
def _norm(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")
    for ch in [" ", "-", ".", "/", "\\", "|"]:
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s

def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    norm_map = {_norm(c): c for c in df.columns}
    for cand in candidates:
        k = _norm(cand)
        if k in norm_map:
            return norm_map[k]
    for k, original in norm_map.items():
        for cand in candidates:
            if _norm(cand) in k:
                return original
    return None

def to_bool(x):
    if pd.isna(x):
        return np.nan
    m = {
        "1": True, "0": False,
        "true": True, "false": False,
        "sim": True, "nao": False, "não": False,
        "on": True, "off": False,
        "ok": True, "nok": False,
        "ativado": True, "ativar": True, "ativo": True,
        "desativado": False, "desativar": False, "inativo": False,
    }
    xs = _norm(str(x))
    return m.get(xs, np.nan)

def to_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.map(to_bool)

def try_read_csv(file) -> pd.DataFrame:
    """Lê CSVs com separador ';' ou XLSX automaticamente."""
    try:
        file.seek(0)
    except Exception:
        pass
    raw = file.read()
    try:
        file.seek(0)
    except Exception:
        pass

    if len(raw) >= 2 and raw[:2] == b"PK":
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                file.seek(0)
                return pd.read_excel(file, engine="openpyxl")
        except Exception:
            pass

    encodings = ["utf-8", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            from io import StringIO
            txt = raw.decode(enc, errors="replace")
            buf = StringIO(txt)
            return pd.read_csv(buf, sep=";", engine="python", on_bad_lines="skip")
        except Exception:
            continue

    raise RuntimeError("Falha ao ler arquivo. Verifique se é CSV com ';' ou XLSX válido.")

def load_and_prepare(uploaded_files) -> Tuple[pd.DataFrame, List[str]]:
    logs, dfs = [], []
    for up in uploaded_files:
        if up is None:
            continue
        df = try_read_csv(up)
        df["fonte"] = getattr(up, "name", "arquivo")
        logs.append(f"✅ {getattr(up, 'name', 'arquivo')}  - {df.shape[0]} linhas, {df.shape[1]} colunas")
        dfs.append(df)
    if not dfs:
        return pd.DataFrame(), logs
    df_all = pd.concat(dfs, ignore_index=True)
    return df_all, logs

# -----------------------------
# Upload
# -----------------------------
st.sidebar.header("📥 Arquivos")
uploaded = st.sidebar.file_uploader(
    "Envie 1 ou mais CSVs (ou XLSX)", type=["csv", "xlsx"], accept_multiple_files=True
)

df, load_logs = load_and_prepare(uploaded)
for line in load_logs:
    st.sidebar.write(line)

if df.empty:
    st.info("Envie pelo menos um arquivo para começar.")
    st.stop()

# -----------------------------
# Identificação das colunas
# -----------------------------
col_cliente = find_col(df, ["cliente", "nome_cliente", "client", "customer"])
col_placa   = find_col(df, ["placa", "placa_veiculo", "placas", "license_plate", "veiculo", "vehicle"])
col_logoff  = find_col(df, ["logoff_ignicao", "logoff_ignição", "logoff"])
col_tempo   = find_col(df, ["app_tempo_direcao", "app_tempo_direção", "tempo_direcao_app", "app_tempodirecao", "tempo_direcao"])
col_bloq    = find_col(df, ["bloqueio_ignicao", "bloqueio_ignição", "ignicao_bloqueio", "bloqueioignicao", "bloq_ignicao"])

# -----------------------------
# Painel principal
# -----------------------------
left, right = st.columns([1,1])

with left:
    st.subheader("👤 Cliente")
    if col_cliente in df.columns:
        clientes = sorted([str(x) for x in df[col_cliente].dropna().unique()])
        selected_cliente = st.selectbox("Selecione o cliente", ["(Todos)"] + clientes, index=0, placeholder="Digite para filtrar…")
    else:
        selected_cliente = "(Todos)"
        st.info("Nenhuma coluna de cliente detectada; operando sobre **todas** as linhas.")

with right:
    st.subheader("🎯 Configuração desejada")
    target_logoff_enabled = st.selectbox("Logoff por ignição", ["Ativar", "Desativar"], index=0) == "Ativar"
    target_bloq = st.selectbox("Bloqueio por ignição", ["Ativar", "Desativar"], index=0) == "Ativar"
    target_app_tempo = st.selectbox("App Tempo de Direção", ["Ativar", "Desativar"], index=1) == "Ativar"

# -----------------------------
# Filtrar cliente
# -----------------------------
if selected_cliente != "(Todos)" and col_cliente in df.columns:
    df = df[df[col_cliente].astype(str) == selected_cliente].copy().reset_index(drop=True)

# -----------------------------
# Normalizar colunas e comparar
# -----------------------------
logoff_flag = to_bool_series(df[col_logoff]) if col_logoff in df.columns else pd.Series([np.nan]*len(df))
bloq_flag   = to_bool_series(df[col_bloq])   if col_bloq   in df.columns else pd.Series([np.nan]*len(df))
app_flag    = to_bool_series(df[col_tempo])  if col_tempo  in df.columns else pd.Series([np.nan]*len(df))

def equals_bool(a, b):
    if pd.isna(a):
        return np.nan
    return bool(a) == bool(b)

cmp_logoff = logoff_flag.map(lambda x: equals_bool(x, target_logoff_enabled))
cmp_bloq   = bloq_flag.map(lambda x: equals_bool(x, target_bloq))
cmp_app    = app_flag.map(lambda x: equals_bool(x, target_app_tempo))

ok_mask = (cmp_logoff & cmp_bloq & cmp_app).fillna(False)
diff_mask = ~ok_mask

# -----------------------------
# Resultados
# -----------------------------
cols = [c for c in [col_placa, col_cliente, col_logoff, col_bloq, col_tempo, "fonte"] if c in df.columns]
df_diff = df[diff_mask].copy()
if cols:
    df_diff = df_diff[cols]

# -----------------------------
# KPIs
# -----------------------------
k1, k2, k3 = st.columns(3)
k1.metric("Total de linhas avaliadas", f"{len(df):,}".replace(",", "."))
k2.metric("Placas divergentes (linhas)", f"{len(df_diff):,}".replace(",", "."))
if col_placa and (col_placa in df_diff.columns):
    k3.metric("Placas únicas divergentes", f"{df_diff[col_placa].nunique():,}".replace(",", "."))
else:
    k3.metric("Placas únicas divergentes", "—")

# -----------------------------
# Exibir tabela
# -----------------------------
st.subheader("Placas com configuração divergente")
st.dataframe(df_diff, use_container_width=True, height=460)

# -----------------------------
# Download CSV
# -----------------------------
def make_download(df: pd.DataFrame) -> Tuple[bytes, str]:
    out = io.StringIO()
    df.to_csv(out, index=False, sep=";")
    return out.getvalue().encode("utf-8"), f"placas_divergentes_{len(df)}_linhas.csv"

csv_bytes, fname = make_download(df_diff)
st.download_button("⬇️ Baixar lista de placas divergentes", data=csv_bytes, file_name=fname, mime="text/csv")

with st.expander("ℹ️ Observações"):
    st.markdown(
        "- Agora **sem mapa de configuração** — basta ajustar as 3 opções à direita.\n"
        "- Aceita **CSV com `;`** ou **Excel (.xlsx)**.\n"
        "- Clientes com colunas faltando são exibidos como divergentes."
    )
