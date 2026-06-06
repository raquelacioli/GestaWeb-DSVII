# ESUSSifilis.py
# -*- coding: utf-8 -*-
"""
GestaWeb DS7 — e-SUS Monitoramento (DS VII) com login Firebase e bloqueio por UNIDADE
- Login (Firebase Email/Password, via API REST oficial)
- ADMIN único pode carregar até 69 planilhas (CSV/XLS/XLSX/ODS)
- Base é persistida em disco para todos os usuários (data/base.parquet ou CSV de fallback)
- Carrega credenciais do Firestore de um arquivo JSON separado
- Usuários comuns: somente visualização e apenas da(s) sua(s) UNIDADE(s)
- Detecta UNIDADE (prioriza coluna do arquivo; fallback por heurística)
- Mantém TODAS as colunas
- Filtro por UNIDADE (por palavra-chave; ignora o que vem após a vírgula)
- Relatórios AR/AS/AT/AU/AV/AN/AJ + gráficos
"""

import io
import re
import csv
import unicodedata
import warnings
import datetime
from io import StringIO
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from uuid import uuid4  # IDs seguros no Firestore

import pandas as pd
import streamlit as st
import altair as alt
import numpy as np
import requests

warnings.simplefilter("ignore", category=UserWarning)
st.set_page_config(page_title="GestaWeb DS7 — e-SUS", layout="wide")

# --- Dependências externas (checagem + instalador local opcional) ---
import importlib, sys, subprocess

# Pacotes listados (com versões seguras e sem pyrebase)
REQUIRED_PIP = [
    "streamlit>=1.36",
    "google-cloud-firestore==2.16.0",
    "google-auth==2.33.0",
    "google-api-core==2.19.1",
    "grpcio==1.65.5",
    "protobuf>=4.24,<5",
    "pandas>=2.0",
    "altair>=5.0",
    "openpyxl>=3.1",
    "sseclient-py",
    "requests_toolbelt",
    "requests",
]

def _check_imports():
    missing = []
    try:
        importlib.import_module("google.cloud.firestore")
    except Exception as e:
        missing.append(("google-cloud-firestore", "from google.cloud import firestore", repr(e)))

    try:
        importlib.import_module("google.oauth2.service_account")
    except Exception as e:
        missing.append(("google-auth", "from google.oauth2 import service_account", repr(e)))

    return missing

_missing = _check_imports()

if _missing:
    st.error("Dependências ausentes ou com erro no import:")
    st.write("\n".join([f"- **{pkg}** (falhou em `{hint}`) → {err}" for pkg, hint, err in _missing]))

    # Mostra o comando pip atualizado
    pip_cmd = (
        'pip install streamlit google-cloud-firestore google-auth '
        'google-api-core grpcio "protobuf>=4.24,<5" pandas altair openpyxl requests'
    )
    st.caption("Comando sugerido (ambiente local):")
    st.code(pip_cmd, language="bash")

    # Instalador local opcional
    if st.button("⚙️ Instalar dependências (ambiente local)"):
        try:
            for pkg in REQUIRED_PIP:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            st.success("Dependências instaladas com sucesso. Reiniciando o app…")
            st.rerun()
        except Exception as e:
            st.error(f"Falha ao instalar dependências: {e!r}")
    st.info("No Streamlit Cloud, use um arquivo requirements.txt na raiz do projeto.")
    st.stop()

# Se chegou aqui, os imports estão OK:
from google.cloud import firestore
from google.oauth2 import service_account

# =========================
# Credenciais / Conexões
# =========================
FIRESTORE_CREDENTIALS_PATH = "service_account.json"  # aponte para o seu JSON

# Chave Web do Firebase para autenticação REST (Login)
FIREBASE_API_KEY = "AIzaSyD7-HwufkeY97nP5I61DZJaeCTNY7ccvKo"

@st.cache_resource(show_spinner=False)
def get_firestore_client():
    try:
        if not Path(FIRESTORE_CREDENTIALS_PATH).is_file():
            st.error("ARQUIVO DE CREDENCIAIS NÃO ENCONTRADO!")
            st.info(f"Verifique o caminho em FIRESTORE_CREDENTIALS_PATH: {FIRESTORE_CREDENTIALS_PATH}")
            st.stop()
        creds = service_account.Credentials.from_service_account_file(FIRESTORE_CREDENTIALS_PATH)
        return firestore.Client(credentials=creds)
    except Exception as e:
        st.error(f"Falha ao conectar no Firestore: {e}")
        st.stop()

def firebase_sign_in(email: str, password: str) -> dict:
    """Faz login usando o endpoint oficial da API Identity Toolkit do Firebase Auth"""
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    response = requests.post(url, json=payload)
    data = response.json()
    
    if response.status_code == 200:
        return data
    else:
        error_message = data.get("error", {}).get("message", "Erro de autenticação desconhecido")
        raise Exception(error_message)

# =========================
# Banner de login (opcional)
# =========================
HERO_CANDIDATES = [
    r"C:\Users\raque\Desktop\Banco Sifilis\assets\gestaweb_ds7.jpg",
    "assets/gestaweb_ds7.jpg",
    "/mnt/data/gestaweb_ds7.jpg",
]
def _pick_hero_path() -> Optional[Path]:
    for p in HERO_CANDIDATES:
        path = Path(p)
        if path.exists():
            return path
    return None
def show_login_banner():
    hero = _pick_hero_path()
    if hero and hero.exists():
        st.image(str(hero), use_container_width=True)

# =========================
# Persistência local — CSV apenas
PERSIST_DIR = Path("data")
PERSIST_DIR.mkdir(exist_ok=True)
CSV_PATH = PERSIST_DIR / "base.csv"

def save_base_to_disk(df: pd.DataFrame) -> None:
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

def load_base_from_disk() -> Optional[pd.DataFrame]:
    if CSV_PATH.exists():
        return pd.read_csv(CSV_PATH, dtype=str)
    return None

# =========================
# Firestore helpers
# =========================
def clean_firestore_value(value):
    if pd.isna(value) or value is None:
        return None
    if isinstance(value, (np.int64, np.int32, np.int16)):
        return int(value)
    if isinstance(value, (np.float64, np.float32)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime.datetime, datetime.date)):
        return value.to_pydatetime() if isinstance(value, pd.Timestamp) else value
    return str(value)

def sanitize_doc_id(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.strip().lower()
    s = s.replace("/", "-").replace("\\", "-")
    s = re.sub(r"[\[\]#?]+", "-", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-\.]+", "-", s)
    s = s[:150].strip("-.")
    return s or f"doc_{uuid4().hex[:10]}"

# =========================
# Mapeamentos / critérios
# =========================
COLMAP = {
    "paciente": [
        "nome do paciente","paciente","nome",
        "nome do usuário","nome do usuario",
        "nome da usuaria","nome da usuária",
        "usuario","usuário"
    ],
}

COL_SPECS = {
    "AR":{"desc":"Exame de HIV no 1º trimestre","type":"text_nao",
          "names":["Exame de HIV no primeiro trimestre","hiv 1º trimestre","hiv 1 tri"]},
    "AS":{"desc":"Exame de Sífilis no 1º trimestre","type":"text_nao",
          "names":["Exame de Sífilis no primeiro trimestre","sífilis 1º trimestre","sifilis 1 tri"]},
    "AT":{"desc":"Exame de Hepatite B no 1º trimestre","type":"text_nao",
          "names":["Exame de Hepatite B no primeiro trimestre","hbv 1º trimestre","hepatite b 1 tri"]},
    "AU":{"desc":"Exame de Hepatite C no 1º trimestre","type":"text_nao",
          "names":["Exame de Hepatite C no primeiro trimestre","hcv 1º trimestre","hepatite c 1 tri"]},
    "AV":{"desc":"Exame de HIV no 3º trimestre","type":"text_nao",
          "names":["Exame de HIV no terceiro trimestre","hiv 3º trimestre","hiv 3 tri"]},
    "AN":{"desc":"Quantidade de atendimentos odontológicos no pré-natal","type":"num_lt","value":1,
          "names":["Quantidade de atendimentos odontológicos no pré-natal","odontol","atend odont"]},
    "AJ":{"desc":"IG (DUM) (semanas)","type":"num_gt","value":43,
          "names":["IG (DUM) (semanas)","idade gestacional (dum)","ig semanas"]},
}
LABELS = {
    "AR":"Exame de HIV no 1º trimestre = NÃO",
    "AS":"Exame de Sífilis no 1º trimestre = NÃO",
    "AT":"Exame de Hepatite B no 1º trimestre = NÃO",
    "AU":"Exame de Hepatite C no 1º trimestre = NÃO",
    "AV":"Exame de HIV no 3º trimestre = NÃO",
    "AN":"Quantidade de atendimentos odontológicos no pré-natal < 1",
    "AJ":"IG (DUM) (semanas) > 43",
}

# =========================
# ADMIN e acesso por e-mail -> PALAVRAS-CHAVE
# =========================
ADMIN_EMAILS = {"vigilanciaepidemiologicadsvii@gmail.com"}

EMAIL_TO_UNITS: Dict[str, List[str]] = {
    "brunomaiatbehansen@gmail.com": ["bruno maia"],
    "passarinhobaixotbehansen@gmail.com": ["passarinho baixo"],
    "passarinhoaltotbehansen@gmail.com": ["passarinho alto"],
    "altodoeucaliptotbehansen@gmail.com": ["alto do eucalipto"],
    "corregodoeucaliptotbehansen@gmail.com": ["corrego do eucalipto"],
    "corregodabicatbehansen@gmail.com": ["corrego da bica"],
    "alcidescodeceiratbehansen@gmail.com": ["alto jose bonifacio"],
    "morrodaconceicaotbehansen@gmail.com": ["morro da conceicao"],
    "mariomonteirotbehansen@gmail.com": ["mario monteiro"],
    "inarosaborgestbehansen@gmail.com": ["ina rosa"],
    "macaxeiratbehansen@gmail.com": ["macaxeira"],
    "mangabeiratbehansen@gmail.com": ["mangabeira"],
    "bolanaredetbehansen@gmail.com": ["bola na rede"],
    "domheldertbehansen@gmail.com": ["dom helder"],
    "heliomendoncatbehansen@gmail.com": ["helio mendonca"],
    "clementinofragatbehansen@gmail.com": ["clementino fraga"],
    "guabirabatbehansen@gmail.com": ["guabiraba"],
    "santaterezatbehansen@gmail.com": ["santa tereza"],
    "moacyrandregomestbehansen@gmail.com": ["moacyr"],
    "vilaboavistatbehansen@gmail.com": ["vila boa vista"],
    "corregodoeuclidestbehansen@gmail.com": ["maria rita"],
    "sitiodosmacacostbehansen01@gmail.com": ["sitios dos macacos","sitio dos macacos"],
    "irmadenisetbehansen@gmail.com": ["irma denise"],
    "diogenescavalcantitbehansen@gmail.com": ["diogenes cavalcanti"],
}

# =========================
# Helpers ETL
# =========================
def normalize(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s)
    return s

def find_first_matching_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_norm = {normalize(c): c for c in df.columns}
    for c in candidates:
        nc = normalize(c)
        if nc in cols_norm:
            return cols_norm[nc]
    for real in df.columns:
        rn = normalize(real)
        for cand in candidates:
            if normalize(cand) in rn:
                return real
    return None

def _read_csv_robusto(file_obj) -> pd.DataFrame:
    if hasattr(file_obj, "read"):
        pos = file_obj.tell() if hasattr(file_obj, "tell") else None
        data = file_obj.read()
        if pos is not None:
            try: file_obj.seek(pos)
            except Exception: pass
    else:
        with open(file_obj, "rb") as fh:
            data = fh.read()
    if isinstance(data, bytes):
        try: text = data.decode("utf-8","ignore")
        except Exception: text = data.decode("latin-1","ignore")
    else:
        text = str(data)
    buf = StringIO(text)
    try:
        return pd.read_csv(buf, sep=None, engine="python", header=None, dtype=str,
                           skip_blank_lines=True, quoting=csv.QUOTE_MINIMAL, on_bad_lines="skip")
    except Exception:
        pass
    for sep in [";","\t",",","|"]:
        buf.seek(0)
        try:
            return pd.read_csv(buf, sep=sep, engine="python", header=None, dtype=str,
                               skip_blank_lines=True, quoting=csv.QUOTE_MINIMAL, on_bad_lines="skip")
        except Exception:
            continue
    buf.seek(0)
    return pd.read_csv(buf, header=None, dtype=str, engine="python", on_bad_lines="skip")

def detect_unit_name(raw_df: pd.DataFrame) -> Optional[str]:
    try:
        b10 = raw_df.iloc[9, 1]
        if isinstance(b10, str) and b10.strip(): return b10.strip()
    except Exception: pass
    try:
        a3 = raw_df.iloc[2, 0]
        if isinstance(a3, str) and normalize(a3).startswith("unidade de saude"):
            return a3.strip()
    except Exception: pass
    for i in range(min(10, len(raw_df))):
        try:
            v = str(raw_df.iloc[i, 0])
            if re.search(r"(?i)unidade\s+de\s+saud(e|é)", v): return v.strip()
        except Exception: continue
    return None

def find_unit_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "UNIDADE","UNIDADE DE SAÚDE","UNIDADE DE SAUDE","UNIDADE (BA)","BA",
        "ESTABELECIMENTO","UNIDADE/ESTABELECIMENTO"
    ]
    cols_norm = {normalize(c): c for c in df.columns}
    for c in candidates:
        if normalize(c) in cols_norm: return cols_norm[normalize(c)]
    for real in df.columns:
        if "unidade" in normalize(real) or "estabelec" in normalize(real):
            return real
    return None

def read_any_table(file) -> Tuple[pd.DataFrame, Optional[str]]:
    name = getattr(file, "name", str(file)).lower()
    if name.endswith(".csv"):
        raw = _read_csv_robusto(file)
    elif name.endswith((".xlsx",".xls")):
        raw = pd.read_excel(file, header=None, dtype=str)
    elif name.endswith(".ods"):
        try: raw = pd.read_excel(file, engine="odf", header=None, dtype=str)
        except Exception: raw = pd.read_excel(file, header=None, dtype=str)
    else:
        raw = _read_csv_robusto(file)

    unidade_hint = detect_unit_name(raw)

    header_row = None
    for i in range(min(30, len(raw))):
        row = raw.iloc[i]
        text_count = sum(len(str(x).strip()) > 0 for x in row.dropna().astype(str))
        if text_count >= 4:
            header_row = i; break
    if header_row is None: header_row = 0

    df = raw.copy()
    df.columns = raw.iloc[header_row].fillna("")
    df = raw.iloc[header_row+1:].reset_index(drop=True)
    df.columns = [str(c).strip() if str(c).strip() != "" else f"col_{i}" for i,c in enumerate(df.columns)]

    col_paciente = find_first_matching_col(df, COLMAP["paciente"]) or df.columns[0]
    df = df[df[col_paciente].notna() & (df[col_paciente].astype(str).str.strip() != "")]

    col_unid = find_unit_column(df)
    if col_unid:
        df["UNIDADE"] = df[col_unid].astype(str).str.strip()
    else:
        df["UNIDADE"] = unidade_hint if unidade_hint else "(UNIDADE NAO DETECTADA)"
    return df, unidade_hint

def excel_letter_to_index(letter: str) -> int:
    if not letter or not letter.isalpha(): return -1
    letter = letter.upper(); idx = 0
    for ch in letter: idx = idx*26 + (ord(ch)-ord('A')+1)
    return idx - 1

def get_col_by_letter_or_name(df: pd.DataFrame, letter: str, names: List[str]) -> Optional[str]:
    j = excel_letter_to_index(letter)
    if 0 <= j < len(df.columns):
        col = df.columns[j]
        if str(col).strip() != "": return col
    names_n = [normalize(n) for n in names]
    for c in df.columns:
        cn = normalize(c)
        for nn in names_n:
            if cn == nn or nn in cn: return c
    return None

def series_is_nao(s: pd.Series) -> pd.Series:
    return s.astype(str).map(normalize).str.fullmatch(r"n[aã]o")

def to_numeric_series(s: pd.Series) -> pd.Series:
    ss = s.astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(ss, errors="coerce")

def build_requested_filters(df: pd.DataFrame) -> Tuple[Dict[str, pd.Series], Dict[str, str]]:
    masks, found_cols = {}, {}
    for code, spec in COL_SPECS.items():
        col = get_col_by_letter_or_name(df, code, spec.get("names", []))
        if col is None: continue
        found_cols[code] = col
        t = spec["type"]
        if t == "text_nao":
            masks[code] = series_is_nao(df[col])
        elif t == "num_lt":
            x = to_numeric_series(df[col]); masks[code] = x < spec.get("value", 0)
        elif t == "num_gt":
            x = to_numeric_series(df[col]); masks[code] = x > spec.get("value", 0)
    return masks, found_cols

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")

def single_sheet_xlsx(df: pd.DataFrame, sheet_name: str = "PLANILHA") -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    out.seek(0)
    return out.read()

# =========================
# Login (Firebase REST)
# =========================
def login_block():
    show_login_banner()
    st.title("🔐 Login — GestaWeb DS7")
    with st.form("login_form", clear_on_submit=False):
        email_in = st.text_input("E-mail institucional", value="", placeholder="nome@dominio")
        pwd_in   = st.text_input("Senha", value="", type="password")
        ok = st.form_submit_button("Entrar")
    if ok:
        try:
            # Login seguro via API REST do Firebase
            user = firebase_sign_in(email_in.strip(), pwd_in)
            st.session_state["auth_ok"]   = True
            st.session_state["user_email"]= email_in.strip().lower()
            st.session_state["id_token"]  = user.get("idToken")
            
            # Permissões
            if st.session_state["user_email"] in ADMIN_EMAILS:
                st.session_state["allowed_units"] = ["*"]
            else:
                st.session_state["allowed_units"] = EMAIL_TO_UNITS.get(st.session_state["user_email"], [])
            st.success("Login realizado com sucesso.")
            st.rerun()
        except Exception as e:
            st.error(f"Falha no login: {e}")
    st.stop()

# Gate de login
if not st.session_state.get("auth_ok"):
    login_block()

# Sidebar
with st.sidebar:
    st.success(f"Conectado: {st.session_state.get('user_email','')}")
    if st.button("Sair"):
        st.session_state.clear()
        st.rerun()

# =========================
# UI principal
# =========================
st.title("Unidades de Saúde DSVII — GestaWeb DS7 (e-SUS)")

user_email = (st.session_state.get("user_email") or "").lower()
allowed_units: List[str] = st.session_state.get("allowed_units", [])
is_admin = (user_email in ADMIN_EMAILS) or ("*" in allowed_units)

if is_admin:
    st.success("Modo ADMIN: você pode carregar planilhas e visualizar todas as unidades.")
else:
    st.info("Modo VISUALIZAÇÃO: você verá apenas os dados da(s) sua(s) unidade(s). Upload desabilitado.")

# Upload (ADMIN)
uploaded_files = None
if is_admin:
    uploaded_files = st.file_uploader(
        "Selecione as planilhas (até 69) — apenas ADMIN",
        type=["csv","xlsx","xls","ods"],
        accept_multiple_files=True,
    )
else:
    st.caption("Upload desabilitado para usuários não-admin.")

if is_admin and uploaded_files:
    if len(uploaded_files) > 69:
        st.warning("Você selecionou mais de 69 arquivos. Apenas os 69 primeiros serão processados.")
        uploaded_files = uploaded_files[:69]
    with st.spinner("Lendo e preparando as planilhas..."):
        dfs = []
        for f in uploaded_files:
            try:
                df, _ = read_any_table(f)
                dfs.append(df)
            except Exception as e:
                st.error(f"Erro ao ler {getattr(f,'name','arquivo')}: {e}")
        if not dfs:
            st.stop()
        consolidated_df = pd.concat(dfs, ignore_index=True)
        st.session_state["base_df"] = consolidated_df
        save_base_to_disk(consolidated_df)
        st.success("Base updated e salva em disco (disponível para todos).")

# Carrega base (sessão ou disco)
if "base_df" not in st.session_state:
    disk_df = load_base_from_disk()
    if disk_df is not None:
        st.session_state["base_df"] = disk_df
    else:
        st.warning("Nenhuma base carregada ainda. O ADMIN precisa carregar as planilhas.")
        st.stop()

base = st.session_state["base_df"].copy()

# Diagnóstico: ADMIN vê as UNIDADES únicas
if is_admin and "UNIDADE" in base.columns:
    with st.expander("Diagnóstico: UNIDADES únicas na base"):
        unidades_unicas = sorted(base["UNIDADE"].dropna().astype(str).str.strip().unique().tolist())
        st.write(unidades_unicas[:300])

# ---------- Filtro por UNIDADE (por palavra-chave) ----------
def _norm_view(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", s).strip().lower()

def _limpa_unidade(u: str) -> str:
    return str(u).split(",", 1)[0].strip()

def _matches_keywords(unit_text: str, allowed_keywords_norm: set[str]) -> bool:
    nu = _norm_view(_limpa_unidade(unit_text))
    return any(kw in nu for kw in allowed_keywords_norm)

if not is_admin:
    if not allowed_units:
        st.warning("Seu e-mail autenticou, mas não está associado a nenhuma UNIDADE. Solicite cadastro.")
        st.stop()
    if "UNIDADE" not in base.columns:
        st.warning("A base carregada não possui a coluna 'UNIDADE'.")
        st.stop()

    keywords_norm = {_norm_view(k) for k in allowed_units}
    base = base[ base["UNIDADE"].astype(str).map(lambda x: _matches_keywords(x, keywords_norm)) ]

    if base.empty:
        with st.expander("Diagnóstico do filtro (não-admin)"):
            st.write("Suas palavras-chave:", allowed_units)
            st.write("Normalizadas:", sorted(list(keywords_norm)))
            try:
                amostra = sorted({
                    _norm_view(_limpa_unidade(u))
                    for u in st.session_state["base_df"]["UNIDADE"].astype(str).fillna("")
                    if str(u).strip()
                })[:120]
                st.write("UNIDADES (limpas+normalizadas) presentes na base (amostra):", amostra)
            except Exception as e:
                st.write("Falha no diagnóstico:", e)
        st.warning("Não há registros compatíveis com a(s) sua(s) palavra(s)-chave na base.")
        st.stop()

    st.info("Visualizando somente a(s) família(s): " + ", ".join(sorted(set(allowed_units))))

# ===== Prévia =====
st.subheader("Pré-visualização da base (todas as colunas)")
st.dataframe(base.head(300), use_container_width=True, height=420)

# ===== Filtro global por UNIDADE =====
st.subheader("Filtro por UNIDADE")
units = sorted(base["UNIDADE"].dropna().astype(str).unique()) if "UNIDADE" in base.columns else []
default_sel = units[:1] if units else []
sel = st.multiselect("Selecione a(s) UNIDADE(s)", options=units, default=default_sel)

if "unit_filter_on" not in st.session_state:
    st.session_state["unit_filter_on"] = False

c1, c2, _ = st.columns([1,1,3])
with c1:
    if st.button("Aplicar filtro por UNIDADE"):
        st.session_state["unit_filter_on"] = True
with c2:
    if st.button("Limpar filtro"):
        st.session_state["unit_filter_on"] = False

base_filtrada = base.copy()
if st.session_state["unit_filter_on"] and sel:
    base_filtrada = base_filtrada[base_filtrada["UNIDADE"].isin(sel)]

st.write(f"Linhas após filtro por UNIDADE: {len(base_filtrada)}")
st.dataframe(base_filtrada.head(300), use_container_width=True, height=420)

df_fonte_relatorios = base_filtrada if (st.session_state["unit_filter_on"] and sel) else base

# ===== Relatórios específicos =====
st.subheader("Relatórios específicos (mantendo TODAS as colunas)")
masks, found_cols = build_requested_filters(df_fonte_relatorios)
if found_cols:
    st.caption("Colunas detectadas (código → nome real): " + ", ".join([f"{k}→{v}" for k,v in found_cols.items()]))
else:
    st.info("AR/AS/AT/AU/AV/AN/AJ não localizadas por letra/nome — verifique o cabeçalho.")

tables_spec: Dict[str, pd.DataFrame] = {}
for code in ["AR","AS","AT","AU","AV","AN","AJ"]:
    if code in masks:
        df_tab = df_fonte_relatorios[masks[code]].copy()
        tables_spec[code] = df_tab
        with st.expander(f"{LABELS[code]} — {len(df_tab)} linha(s)"):
            st.dataframe(df_tab.head(300), use_container_width=True, height=360)
            cdl1, cdl2 = st.columns(2)
            with cdl1:
                st.download_button(f"Baixar {code}.csv",
                                   data=df_to_csv_bytes(df_tab),
                                   file_name=f"{code}.csv",
                                   mime="text/csv",
                                   key=f"csv_{code}")
            with cdl2:
                st.download_button(f"Baixar {code}.xlsx",
                                   data=single_sheet_xlsx(df_tab, sheet_name=code),
                                   file_name=f"{code}.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   key=f"xlsx_{code}")
    else:
        st.info(f"Filtro {LABELS.get(code, code)}: coluna {code} não localizada.")

# ===== Análise por critério + gráfico por unidade =====
st.subheader("Análise por critério e Unidade de Saúde")

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", s).strip().lower()

def resolve_crit_key(selection_text: str) -> Optional[str]:
    sel_n = _norm(selection_text)
    for code, spec in COL_SPECS.items():
        if sel_n == _norm(spec.get("desc","")): return code
        for nm in spec.get("names", []):
            if sel_n == _norm(nm): return code
        if _norm(spec.get("desc","")) in sel_n: return code
        for nm in spec.get("names", []):
            if _norm(nm) in sel_n: return code
    return None

crit_options = sorted({ (spec.get("names") or [spec.get("desc", code)])[0] for code, spec in COL_SPECS.items() },
                      key=lambda x: _norm(x))

crit_selecionado = st.selectbox(
    "Escolha o critério",
    options=crit_options,
    index=0,
    help="Os critérios mapeiam para AR/AS/AT/AU/AV/AN/AJ conforme o cabeçalho das planilhas."
)

crit_key = resolve_crit_key(crit_selecionado)

if not crit_key:
    st.warning("Critério não reconhecido (sem mapeamento).")
elif crit_key not in masks:
    st.warning(f"O critério {crit_key} não foi localizado nas planilhas.")
else:
    df_crit = df_fonte_relatorios[masks[crit_key]].copy()
    unidades_crit = sorted(df_crit["UNIDADE"].dropna().astype(str).unique())
    sel_unid_crit = st.multiselect("Filtrar por Unidade de Saúde (apenas para este critério)",
                                   options=unidades_crit, default=unidades_crit)
    if sel_unid_crit:
        df_crit = df_crit[df_crit["UNIDADE"].isin(sel_unid_crit)]

    st.write(f"{LABELS[crit_key]} — total: {len(df_crit)} registro(s)")

    with st.expander("Ver linhas deste critério"):
        st.dataframe(df_crit.head(300), use_container_width=True, height=320)
        st.download_button("Baixar CSV (critério filtrado)",
                           data=df_to_csv_bytes(df_crit),
                           file_name=f"{crit_key}_criterio_filtrado.csv",
                           mime="text/csv")

    if df_crit.empty:
        st.info("Nenhum registro para este critério com o filtro selecionado.")
    else:
        grp = df_crit.groupby("UNIDADE").size().reset_index(name="quantidade")
        ordem_final = sorted(grp["UNIDADE"].tolist())
        grp["UNIDADE"] = pd.Categorical(grp["UNIDADE"], categories=ordem_final, ordered=True)
        grp = grp.sort_values("UNIDADE")
        st.caption(f"Gráfico — {LABELS[crit_key]} por Unidade de Saúde")
        chart = alt.Chart(grp).mark_bar().encode(
            x=alt.X("UNIDADE:O", sort=ordem_final, title="Unidade de Saúde"),
            y=alt.Y("quantidade:Q", title="Quantidade"),
            tooltip=[alt.Tooltip("UNIDADE:O", title="Unidade"),
                     alt.Tooltip("quantidade:Q", title="Quantidade")]
        )
        st.altair_chart(chart.properties(height=420, width="container"), use_container_width=True)

# ===== Resumo e comparativo por prefixo =====
TARGET_PREFIXES = [
    "ALTO DO EUCALIPTO","ALTO JOSE BONIFACIO","ESF MAIS BRUNO MAIA","ESF MAIS CORREGO DA BICA","BOLA NA REDE",
    "ALTO DA BRASILEIRA","CORREGO DO EUCALIPTO","ESF MAIS CORREGO JENIPAPO","ESF MAIS DOM HELDER","GUABIRABA",
    "INA ROSA","ESF MAIS MACAXEIRA","MANGABEIRA","ESF MAIS MARIA RITA","MARIO MONTEIRO","ESF MAIS MOACYR",
    "MORRO DA CONCEICAO","PASSARINHO ALTO","PASSARINHO BAIXO","ALTO JOSE DO PINHO","ALTO DO RESERVATORIO",
    "SANTA TEREZA","SITIOS DOS MACACOS"
]
def _norm_prefix(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", s).strip().lower()

TARGET_PREFIXES_N = [_norm_prefix(p) for p in TARGET_PREFIXES]

def _matches_any_prefix(u: str) -> bool:
    return any(_norm_prefix(u).startswith(pn) for pn in TARGET_PREFIXES_N)

def unit_to_bucket(u: str) -> Optional[str]:
    nu = _norm_prefix(u)
    for raw_prefix, norm_prefix in zip(TARGET_PREFIXES, TARGET_PREFIXES_N):
        if nu.startswith(norm_prefix): return raw_prefix
    return None

st.subheader("Resumo — grupo de unidades (nomes com mesmo início)")
if "UNIDADE" not in df_fonte_relatorios.columns:
    st.warning("Não há coluna UNIDADE para agregar no resumo.")
else:
    df_grupo = df_fonte_relatorios[
        df_fonte_relatorios["UNIDADE"].astype(str).apply(_matches_any_prefix)
    ].copy()

    if df_grupo.empty:
        st.info("Nenhuma linha pertence ao grupo de unidades (por prefixo).")
    else:
        codigos_para_resumo = ["AR","AS","AT","AU","AV","AN","AJ"]
        contagens = []
        idx_grupo = df_grupo.index
        for code in codigos_para_resumo:
            if code in masks:
                mask_total = masks[code]
                total = int(mask_total.loc[mask_total.index.intersection(idx_grupo)].sum())
            else:
                total = 0
            contagens.append({"codigo":code,"criterio":LABELS.get(code,code),"quantidade":total})
        df_resumo = pd.DataFrame(contagens).sort_values("codigo")

        with st.expander("Tabela de totais no grupo de unidades (por prefixo)"):
            st.dataframe(df_resumo[["criterio","quantidade"]], use_container_width=True, height=300)
            st.download_button("Baixar CSV (resumo do grupo por prefixo)",
                               data=df_to_csv_bytes(df_resumo[["criterio","quantidade"]]),
                               file_name="resumo_grupo_unidades_prefixo.csv",
                               mime="text/csv")

        st.caption("Gráfico — Totais por critério (grupo de unidades por prefixo)")
        chart_resumo = alt.Chart(df_resumo).mark_bar().encode(
            x=alt.X("criterio:N", sort=df_resumo["criterio"].tolist(), title="Critério"),
            y=alt.Y("quantidade:Q", title="Quantidade"),
            tooltip=[alt.Tooltip("criterio:N", title="Critério"),
                     alt.Tooltip("quantidade:Q", title="Quantidade")]
        )
        st.altair_chart(chart_resumo.properties(height=420, width="container"), use_container_width=True)

        st.subheader("Comparativo por unidade (agrupada por prefixo) — todos os critérios")
        df_all = df_fonte_relatorios.copy()
        df_all["bucket"] = df_all["UNIDADE"].astype(str).apply(unit_to_bucket)
        df_all = df_all[df_all["bucket"].notna()]

        if df_all.empty:
            st.info("Nenhuma linha pertence ao grupo de unidades (por prefixo).")
        else:
            linhas = []
            for code in codigos_para_resumo:
                if code in masks:
                    idx_ok = masks[code].index.intersection(df_all.index)
                    df_code = df_all.loc[idx_ok]
                    grp_b = df_code.groupby("bucket").size().reset_index(name="quantidade")
                else:
                    grp_b = pd.DataFrame({"bucket":sorted(df_all["bucket"].unique()),"quantidade":0})
                grp_b["codigo"] = code
                grp_b["criterio"] = LABELS.get(code, code)
                linhas.append(grp_b)

            df_barras = pd.concat(linhas, ignore_index=True) if linhas else pd.DataFrame(
                columns=["bucket","quantidade","codigo","criterio"]
            )
            df_barras["quantidade"] = pd.to_numeric(df_barras["quantidade"], errors="coerce").fillna(0)

            ordem_buckets = [b for b in TARGET_PREFIXES if b in df_barras["bucket"].unique().tolist()]
            restantes = [b for b in df_barras["bucket"].unique().tolist() if b not in ordem_buckets]
            ordem_buckets = ordem_buckets + sorted(restantes)

            with st.expander("Tabela — contagens por unidade (consolidada por prefixo) e critério"):
                st.dataframe(df_barras, use_container_width=True, height=360)
                st.download_button("Baixar CSV (unidade consolidada × critério)",
                                   data=df_to_csv_bytes(df_barras),
                                   file_name="comparativo_unidades_consolidadas.csv",
                                   mime="text/csv")

            chart_buckets = alt.Chart(df_barras).mark_bar().encode(
                x=alt.X("bucket:O", sort=ordem_buckets, title="Unidade (agrupada por prefixo)"),
                y=alt.Y("quantidade:Q", title="Quantidade"),
                color=alt.Color("criterio:N", title="Critério"),
                xOffset=alt.XOffset("criterio:N"),  
                tooltip=[
                    alt.Tooltip("bucket:O",     title="Unidade"),
                    alt.Tooltip("criterio:N",   title="Critério"),
                    alt.Tooltip("quantidade:Q", title="Quantidade"),
                ],
            )
            st.altair_chart(
                chart_buckets.properties(height=440, width="container"),
                use_container_width=True
            )
