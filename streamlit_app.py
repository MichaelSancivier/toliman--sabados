# -*- coding: utf-8 -*-
import io
import unicodedata
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Conferência de Configuração por Placa",
    layout="wide",
)

st.title("🧭 Conferência de Configuração por Placa")
st.caption("Mostra as **placas** que estão **diferentes** da configuração solicitada pelo cliente.")

# -----------------------------
# Helpers
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
    """Find first matching column (case/accents-insensitive). Returns actual df column name."""
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
        "y": True, "n": False,
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

def parse_seconds(val) -> Optional[float]:
    """Accepts 12, 12.5 (segundos) or 'HH:MM:SS' / 'MM:SS'."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    if s == "":
        return None
    try:
        f = float(s)
        return f
    except Exception:
        pass
    parts = s.split(":")
    try:
        parts = [int(p) for p in parts]
        if len(parts) == 3:
            h, m, sec = parts
            return h*3600 + m*60 + sec
        if len(parts) == 2:
            m, sec = parts
            return m*60 + sec
    except Exception:
        return None
    return None

def try_read_csv(file) -> pd.DataFrame:
    """
    Robust loader:
    - Detecta Excel (XLS/XLSX) e usa read_excel.
    - Tenta encodings: utf-8/latin-1/cp1252 e usa chardet se disponível.
    - Testa separadores: vírgula, ponto e vírgula, tab, pipe.
    - Fallback com engine='python' e on_bad_lines='skip'.
    """
    # pegar bytes do UploadedFile
    try:
        file.seek(0)
    except Exception:
        pass
    try:
        raw = file.read()
    finally:
        try:
            file.seek(0)
        except Exception:
            pass

    if raw is None:
        raise ValueError("Arquivo vazio.")

    # XLSX (ZIP: começa com 'PK')
    if len(raw) >= 2 and raw[:2] == b"PK":
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                file.seek(0)
                return pd.read_excel(file, engine="openpyxl")
        except Exception:
            try:
                file.seek(0)
            except Exception:
                pass
            # continua tentando CSV abaixo

    # encodings
    encodings = ["utf-8", "latin-1", "cp1252"]
    try:
        import chardet
        guess = chardet.detect(raw).get("encoding")
        if guess and guess.lower() not in [e.lower() for e in encodings]:
            encodings = [guess] + encodings
    except Exception:
        pass

    # separadores
    seps = [",", ";", "\t", "|"]
    for enc in encodings:
        for sep in seps:
            try:
                from io import StringIO
                txt = raw.decode(enc, errors="strict")
                buf = StringIO(txt)
                return pd.read_csv(buf, sep=sep, engine="python")
            except UnicodeDecodeError:
                continue
            except Exception:
                # tenta ignorar linhas ruins
                try:
                    from io import StringIO
                    txt = raw.decode(enc, errors="replace")
                    buf = StringIO(txt)
                    return pd.read_csv(buf, sep=sep, engine="python", on_bad_lines="skip")
                except Exception:
                    continue

    # último recurso
    try:
        file.seek(0)
        return pd.read_csv(file, engine="python", on_bad_lines="skip")
    except Exception as e:
        raise RuntimeError(f"Falha ao ler arquivo. Tente salvar como CSV UTF-8. Detalhes: {e}")

def load_and_prepare(uploaded_files) -> Tuple[pd.DataFrame, List[str]]:
    logs = []
    dfs = []
    for up in uploaded_files:
        if up is None:
            continue
        df = try_read_csv(up)
        df["fonte"] = getattr(up, "name", "arquivo")
        logs.append(f"✅ Carregado: {getattr(up, 'name', 'arquivo')}  - {df.shape[0]} linhas, {df.shape[1]} colunas")
        dfs.append(df)
    if not dfs:
        return pd.DataFrame(), logs
    df_all = pd.concat(dfs, ignore_index=True)
    return df_all, logs

# -----------------------------
# Sidebar - Uploads
# -----------------------------
st.sidebar.header("📥 Arquivos")
uploaded = st.sidebar.file_uploader("Envie 1 ou mais CSVs (ou XLSX)", type=["csv", "xlsx"], accept_multiple_files=True)
preset_file = st.sidebar.file_uploader(
    "Mapa de Configuração por Cliente (opcional)",
    type=["csv", "xlsx"],
    help="Colunas esperadas: cliente, logoff_enabled, logoff_seconds, bloqueio_ignicao, app_tempo_direcao"
)

df, load_logs = load_and_prepare(uploaded)
for line in load_logs:
    st.sidebar.write(line)

if df.empty:
    st.info("Envie pelo menos um arquivo para começar.")
    st.stop()

# -----------------------------
# Column detection
# -----------------------------
col_cliente = find_col(df, ["cliente", "nome_cliente", "client", "customer"])
col_placa   = find_col(df, ["placa", "placa_veiculo", "placas", "license_plate", "veiculo", "vehicle"])
col_logoff  = find_col(df, ["logoff_ignicao", "logoff_ignição", "logoff"])
col_logoff_s = find_col(df, ["logoff_tempo", "tempo_logoff", "logoff_delay", "logoff_segundos", "logoff_seconds"])
col_tempo   = find_col(df, ["app_tempo_direcao", "app_tempo_direção", "tempo_direcao_app", "app_tempodirecao", "tempo_direcao"])
col_bloq    = find_col(df, ["bloqueio_ignicao", "bloqueio_ignição", "ignicao_bloqueio", "bloqueioignicao", "bloq_ignicao"])

missing = [name for name, col in [
    ("Placa", col_placa),
    ("Logoff_Ignição (flag ou tempo)", col_logoff or col_logoff_s),
    ("Bloqueio_Ignição", col_bloq),
    ("App_Tempo_Direção", col_tempo),
] if col is None]

if missing:
    st.warning("Colunas não encontradas automaticamente: **{}**. Ajuste os nomes no arquivo se necessário.".format(", ".join(missing)))

# -----------------------------
# Painel principal (Cliente + Configuração Alvo)
# -----------------------------
left, right = st.columns([1,1])

with left:
    st.subheader("👤 Cliente")
    if col_cliente in df.columns:
        clientes = sorted([str(x) for x in df[col_cliente].dropna().unique()])
        selected_cliente = st.selectbox("Selecione o cliente", ["(Todos)"] + clientes)
    else:
        selected_cliente = "(Todos)"
        st.info("Nenhuma coluna de cliente detectada; operando sobre **todas** as linhas.")

# presets por cliente via CSV/XLSX opcional
preset_map = None
if preset_file is not None:
    try:
        # aceita csv e xlsx
        if getattr(preset_file, "name", "").lower().endswith(".xlsx"):
            preset_df = pd.read_excel(preset_file, engine="openpyxl")
        else:
            preset_df = try_read_csv(preset_file)

        req_cols = ["cliente", "logoff_enabled", "logoff_seconds", "bloqueio_ignicao", "app_tempo_direcao"]
        ok = all(any(_norm(c) == _norm(x) for c in preset_df.columns) for x in req_cols)
        if ok:
            colmap = {}
            for x in req_cols:
                for c in preset_df.columns:
                    if _norm(c) == _norm(x):
                        colmap[x] = c
                        break
            preset_df = preset_df.rename(columns=colmap)
            preset_map = preset_df.set_index("cliente").to_dict(orient="index")
        else:
            st.sidebar.warning("Mapa de configuração: colunas esperadas não encontradas. Esperado: " + ", ".join(req_cols))
    except Exception as e:
        st.sidebar.error(f"Erro ao ler mapa de configuração: {e}")

def _get_preset_for(cliente: str):
    if not preset_map or not cliente or cliente == "(Todos)":
        return None
    return preset_map.get(cliente)

with right:
    st.subheader("🎯 Configuração desejada")
    # defaults
    default_logoff_enabled = True
    default_logoff_seconds = 30
    default_bloq = True
    default_app_tempo = False

    # aplica preset do cliente se existir
    preset = _get_preset_for(selected_cliente)
    if preset:
        default_logoff_enabled = bool(to_bool(preset.get("logoff_enabled"))) if "logoff_enabled" in preset else default_logoff_enabled
        try:
            default_logoff_seconds = int(preset.get("logoff_seconds"))
        except Exception:
            pass
        default_bloq = bool(to_bool(preset.get("bloqueio_ignicao"))) if "bloqueio_ignicao" in preset else default_bloq
        default_app_tempo = bool(to_bool(preset.get("app_tempo_direcao"))) if "app_tempo_direcao" in preset else default_app_tempo
        st.caption("Preset carregado a partir do mapa de configuração.")

    target_logoff_enabled = st.selectbox("Logoff por ignição", ["Ativar", "Desativar"], index=0 if default_logoff_enabled else 1) == "Ativar"
    target_logoff_seconds  = st.number_input("Tempo do logoff (segundos)", min_value=0, value=int(default_logoff_seconds), step=5)
    tolerance_seconds = st.number_input("Tolerância do tempo (± seg.)", min_value=0, value=5, step=1)
    target_bloq = st.selectbox("Bloqueio por ignição", ["Ativar", "Desativar"], index=0 if default_bloq else 1) == "Ativar"
    target_app_tempo = st.selectbox("App Tempo de Direção", ["Ativar", "Desativar"], index=0 if default_app_tempo else 1) == "Ativar"

# -----------------------------
# Filtrar por cliente (se selecionado)
# -----------------------------
if selected_cliente != "(Todos)" and col_cliente in df.columns:
    df = df[df[col_cliente].astype(str) == selected_cliente].copy()

# -----------------------------
# Normalize/parse columns
# -----------------------------
if col_logoff in df.columns:
    logoff_flag = to_bool_series(df[col_logoff])
else:
    logoff_flag = pd.Series([np.nan]*len(df))

if col_logoff_s in df.columns:
    logoff_secs = df[col_logoff_s].map(parse_seconds)
else:
    if col_logoff in df.columns:
        tmp = df[col_logoff].map(parse_seconds)
        logoff_secs = tmp
        if logoff_flag.isna().any():
            guess_flag = (pd.to_numeric(tmp, errors="coerce") > 0).replace({False: np.nan})
            logoff_flag = logoff_flag.fillna(guess_flag)
    else:
        logoff_secs = pd.Series([np.nan]*len(df))

bloq_flag = to_bool_series(df[col_bloq]) if col_bloq in df.columns else pd.Series([np.nan]*len(df))

if col_tempo in df.columns:
    tempo_raw = df[col_tempo]
    tempo_as_bool = to_bool_series(tempo_raw)
    tempo_as_secs = tempo_raw.map(parse_seconds)
    app_tempo_flag = tempo_as_bool
    if app_tempo_flag.isna().any():
        inferred = (pd.to_numeric(tempo_as_secs, errors="coerce") > 0)
        app_tempo_flag = app_tempo_flag.fillna(inferred)
else:
    app_tempo_flag = pd.Series([np.nan]*len(df))

# -----------------------------
# Comparações
# -----------------------------
def equals_bool(a, b):
    if pd.isna(a):
        return np.nan
    return bool(a) == bool(b)

def equals_seconds(a, b, tol):
    if a is None or (isinstance(a, float) and np.isnan(a)):
        return np.nan
    try:
        return abs(float(a) - float(b)) <= float(tol)
    except Exception:
        return np.nan

desired = {
    "logoff_ativo": target_logoff_enabled,
    "logoff_tempo_s": target_logoff_seconds,
    "bloqueio_ignicao": target_bloq,
    "app_tempo_direcao_ativo": target_app_tempo,
}

cmp_logoff_flag = logoff_flag.map(lambda x: equals_bool(x, desired["logoff_ativo"]))
cmp_logoff_secs = logoff_secs.map(lambda x: equals_seconds(x, desired["logoff_tempo_s"], tolerance_seconds))
cmp_bloq = bloq_flag.map(lambda x: equals_bool(x, desired["bloqueio_ignicao"]))
cmp_app  = app_tempo_flag.map(lambda x: equals_bool(x, desired["app_tempo_direcao_ativo"]))

def row_ok(i):
    vals = []
    if not pd.isna(cmp_logoff_flag.iat[i]):
        vals.append(bool(cmp_logoff_flag.iat[i]))
    if not pd.isna(cmp_logoff_secs.iat[i]):
        vals.append(bool(cmp_logoff_secs.iat[i]))
    if not pd.isna(cmp_bloq.iat[i]):
        vals.append(bool(cmp_bloq.iat[i]))
    if not pd.isna(cmp_app.iat[i]):
        vals.append(bool(cmp_app.iat[i]))
    if not vals:
        return False
    return all(vals)

ok_mask = pd.Series([row_ok(i) for i in range(len(df))])
diff_mask = ~ok_mask

# -----------------------------
# Resultados: apenas placas divergentes
# -----------------------------
out_cols = []
if col_placa in df.columns: out_cols.append(col_placa)
if col_cliente in df.columns: out_cols.append(col_cliente)
for c in [col_logoff, col_logoff_s, col_bloq, col_tempo, "fonte"]:
    if c in df.columns:
        out_cols.append(c)

df_diff = df[diff_mask].copy()
if out_cols:
    df_diff = df_diff[out_cols]

df_diff["ok_logoff_flag"] = cmp_logoff_flag[diff_mask].values
df_diff["ok_logoff_tempo"] = cmp_logoff_secs[diff_mask].values
df_diff["ok_bloqueio"] = cmp_bloq[diff_mask].values
df_diff["ok_app_tempo"] = cmp_app[diff_mask].values

# -----------------------------
# KPIs
# -----------------------------
k1, k2, k3 = st.columns(3)
k1.metric("Total de linhas avaliadas", f"{len(df):,}".replace(",", "."))
k2.metric("Placas divergentes (linhas)", f"{len(df_diff):,}".replace(",", "."))
if col_placa in df_diff.columns:
    k3.metric("Placas únicas divergentes", f"{df_diff[col_placa].nunique():,}".replace(",", "."))

# -----------------------------
# Tabela
# -----------------------------
st.subheader("Placas com configuração divergente")
st.write("Cliente: **{}** | Configuração alvo: **Logoff** = {} ({}s), **Bloqueio** = {}, **App Tempo Direção** = {}.".format(
    selected_cliente,
    "Ativo" if desired["logoff_ativo"] else "Inativo",
    desired["logoff_tempo_s"],
    "Ativo" if desired["bloqueio_ignicao"] else "Inativo",
    "Ativo" if desired["app_tempo_direcao_ativo"] else "Inativo",
))

st.dataframe(df_diff, use_container_width=True, height=460)

# -----------------------------
# Download
# -----------------------------
def make_download(df: pd.DataFrame) -> Tuple[bytes, str]:
    out = io.StringIO()
    df.to_csv(out, index=False)
    b = out.getvalue().encode("utf-8")
    return b, f"placas_divergentes_{len(df)}_linhas.csv"

csv_bytes, fname = make_download(df_diff)
st.download_button("⬇️ Baixar lista de placas divergentes", data=csv_bytes, file_name=fname, mime="text/csv")

with st.expander("ℹ️ Observações"):
    st.markdown(
        "- **Painel de Cliente**: selecione um cliente ou use (Todos). Opcionalmente, envie um CSV/XLSX de **Mapa de Configuração** com presets por cliente.\n"
        "- **Configuração desejada**: ajuste os campos à direita. Se existir preset, ele é aplicado automaticamente.\n"
        "- Se o arquivo não tiver coluna de **tempo do logoff**, a verificação usa apenas o **flag** (ativado/desativado).\n"
        "- Duração aceita números (segundos) e formatos `HH:MM:SS` / `MM:SS`.\n"
        "- Linhas sem informação suficiente também aparecem como divergentes para facilitar revisão."
    )
