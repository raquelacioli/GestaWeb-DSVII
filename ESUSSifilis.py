# ESUSSifilis.py
# -*- coding: utf-8 -*-
"""
GestaWeb DS7 — e-SUS Monitoramento (DS VII) com login Firebase e bloqueio por UNIDADE
- Login (Firebase Email/Password, via Pyrebase)
- ADMIN único pode carregar até 69 planilhas (CSV/XLS/XLSX/ODS)
- Base é persistida em disco para todos os usuários (data/base.parquet ou CSV de fallback)
- <<< CORRIGIDO: Carrega credenciais do Firestore de um arquivo JSON separado >>>
- Usuários comuns: somente visualização e apenas da(s) sua(s) UNIDADE(s)
- Detecta UNIDADE (prioriza coluna do arquivo; fallback por heurística)
- Mantém TODAS as colunas
- Filtro por UNIDADE
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

import pandas as pd
import streamlit as st
import altair as alt
import numpy as np

warnings.simplefilter("ignore", category=UserWarning)
st.set_page_config(page_title="GestaWeb DS7 — e-SUS", layout="wide")

# <<< INÍCIO DA SEÇÃO CORRIGIDA >>>
# --- Importações necessárias ---
try:
    from google.cloud import firestore
    from google.oauth2 import service_account
    import pyrebase
except ImportError:
    st.error("Dependências ausentes! Por favor, instale: pip install pyrebase4 google-cloud-firestore")
    st.stop()

# =========================
# Configuração das Credenciais e Conexões
# =========================

# >>> IMPORTANTE: Coloque o nome (ou caminho) do seu arquivo de credenciais do Firestore aqui <<<
FIRESTORE_CREDENTIALS_PATH = "service_account.json"  # ALTERE SE O NOME DO SEU ARQUIVO FOR DIFERENTE

# Configuração do Pyrebase (usado apenas para o login de email/senha)
FIREBASE_CONFIG = {
    "apiKey": "AIzaSyD7-HwufkeY97nP5I61DZJaeCTNY7ccvKo",
    "authDomain": "gestawebds7raquelacioli.firebaseapp.com",
    "projectId": "gestawebds7raquelacioli",
    "storageBucket": "gestawebds7raquelacioli.appspot.com",
    "messagingSenderId": "536966961533",
    "appId": "1:536966961533:web:1f4b0bfe25f60799c31414",
    "databaseURL": "https://gestawebds7raquelacioli-default-rtdb.firebaseio.com"
}

# --- Inicialização do Pyrebase (para Login) ---
@st.cache_resource(show_spinner=False)
def firebase_init():
    try:
        fb = pyrebase.initialize_app(FIREBASE_CONFIG)
        return fb
    except Exception as e:
        st.error(f"Falha ao inicializar Pyrebase (para login): {e}")
        st.stop()

# --- Inicialização do Firestore (para o Banco de Dados) ---
@st.cache_resource(show_spinner=False)
def get_firestore_client():
    """Inicializa o cliente do Firestore usando um arquivo de credenciais."""
    try:
        # Verifica se o arquivo de credenciais existe no caminho especificado
        if not Path(FIRESTORE_CREDENTIALS_PATH).is_file():
            st.error(f"ARQUIVO DE CREDENCIAIS NÃO ENCONTRADO!")
            st.error(f"Verifique se o arquivo '{FIRESTORE_CREDENTIALS_PATH}' está na pasta correta.")
            st.info("O caminho do arquivo é definido na variável 'FIRESTORE_CREDENTIALS_PATH' no topo do script.")
            st.stop()
        
        creds = service_account.Credentials.from_service_account_file(FIRESTORE_CREDENTIALS_PATH)
        db = firestore.Client(credentials=creds)
        return db
    except Exception as e:
        st.error(f"Falha ao conectar com o Firestore usando o arquivo de credenciais: {e}")
        st.stop()
# <<< FIM DA SEÇÃO CORRIGIDA >>>


def firebase_sign_in(firebase, email: str, password: str):
    auth = firebase.auth()
    return auth.sign_in_with_email_and_password(email, password)

# =========================
# Banner no login (opcional)
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
# Persistência da base (disco)
# =========================
PERSIST_DIR = Path("data")
PERSIST_DIR.mkdir(exist_ok=True)
PERSIST_PATH = PERSIST_DIR / "base.parquet"

def save_base_to_disk(df: pd.DataFrame) -> None:
    try:
        df.to_parquet(PERSIST_PATH, index=False)
    except Exception:
        df.to_csv(PERSIST_PATH.with_suffix(".csv"), index=False, encoding="utf-8-sig")

def load_base_from_disk() -> Optional[pd.DataFrame]:
    if PERSIST_PATH.exists():
        return pd.read_parquet(PERSIST_PATH)
    csv_path = PERSIST_PATH.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path, dtype=str)
    return None

# =========================
# Persistência no Firestore
# =========================
def clean_firestore_value(value):
    """Converte valores do Pandas para tipos compatíveis com Firestore."""
    if pd.isna(value) or value is None:
        return None
    if isinstance(value, (np.int64, np.int32, np.int16)):
        return int(value)
    if isinstance(value, (np.float64, np.float32)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime.datetime, datetime.date)):
        return value.to_pydatetime() if isinstance(value, pd.Timestamp) else value
    return str(value)

def save_to_firestore(df: pd.DataFrame, collection_name: str = "gestantes_sifilis"):
    """
    Converte um DataFrame e salva cada linha como um documento no Firestore.
    """
    try:
        db = get_firestore_client()
        st.info(f"Iniciando salvamento de {len(df)} registros no Firestore na coleção '{collection_name}'...")

        sanitized_columns = {col: re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in df.columns}
        df_sanitized = df.rename(columns=sanitized_columns)
        
        col_paciente_sanitized = find_first_matching_col(df_sanitized, COLMAP["paciente"]) or df_sanitized.columns[0]
        unidade_col_sanitized = "UNIDADE" if "UNIDADE" in df_sanitized.columns else None

        progress_bar = st.progress(0)
        total_rows = len(df_sanitized)
        
        batch = db.batch()
        batch_count = 0
        
        for i, row in df_sanitized.iterrows():
            doc_data = {col: clean_firestore_value(val) for col, val in row.items()}
            
            paciente_nome = normalize(str(doc_data.get(col_paciente_sanitized, ""))).replace(" ", "_")
            unidade = normalize(str(doc_data.get(unidade_col_sanitized, "sem_unidade"))).replace(" ", "_")
            doc_id = f"{unidade}_{paciente_nome}_{i}"
            
            doc_data["_timestamp_upload"] = firestore.SERVER_TIMESTAMP
            
            doc_ref = db.collection(collection_name).document(doc_id)
            batch.set(doc_ref, doc_data)
            batch_count += 1
            
            if batch_count >= 499:
                batch.commit()
                batch = db.batch()
                batch_count = 0

            progress_bar.progress((i + 1) / total_rows)
            
        if batch_count > 0:
            batch.commit()
            
        progress_bar.empty()
        st.success(f"Todos os {total_rows} registros foram salvos/atualizados no Firestore com sucesso!")

    except Exception as e:
        st.error(f"Ocorreu um erro ao salvar os dados no Firestore: {e}")
        st.exception(e)

# =========================
# Mapeamentos de coluna e critérios
# =========================
COLMAP = {
    "paciente": [
        "nome do paciente", "paciente", "nome",
        "nome do usuário", "nome do usuario",
        "nome da usuaria", "nome da usuária",
        "usuario", "usuário"
    ],
}

COL_SPECS = {
    "AR": {"desc": "Exame de HIV no 1º trimestre", "type": "text_nao",
           "names": ["Exame de HIV no primeiro trimestre", "hiv 1º trimestre", "hiv 1 tri"]},
    "AS": {"desc": "Exame de Sífilis no 1º trimestre", "type": "text_nao",
           "names": ["Exame de Sífilis no primeiro trimestre", "sífilis 1º trimestre", "sifilis 1 tri"]},
    "AT": {"desc": "Exame de Hepatite B no 1º trimestre", "type": "text_nao",
           "names": ["Exame de Hepatite B no primeiro trimestre", "hbv 1º trimestre", "hepatite b 1 tri"]},
    "AU": {"desc": "Exame de Hepatite C no 1º trimestre", "type": "text_nao",
           "names": ["Exame de Hepatite C no primeiro trimestre", "hcv 1º trimestre", "hepatite c 1 tri"]},
    "AV": {"desc": "Exame de HIV no 3º trimestre", "type": "text_nao",
           "names": ["Exame de HIV no terceiro trimestre", "hiv 3º trimestre", "hiv 3 tri"]},
    "AN": {"desc": "Quantidade de atendimentos odontológicos no pré-natal", "type": "num_lt", "value": 1,
           "names": ["Quantidade de atendimentos odontológicos no pré-natal", "odontol", "atend odont"]},
    "AJ": {"desc": "IG (DUM) (semanas)", "type": "num_gt", "value": 43,
           "names": ["IG (DUM) (semanas)", "idade gestacional (dum)", "ig semanas"]},
}
CODE_ALIAS = {"NA": "AN"}

LABELS = {
    "AR": "Exame de HIV no 1º trimestre = NÃO",
    "AS": "Exame de Sífilis no 1º trimestre = NÃO",
    "AT": "Exame de Hepatite B no 1º trimestre = NÃO",
    "AU": "Exame de Hepatite C no 1º trimestre = NÃO",
    "AV": "Exame de HIV no 3º trimestre = NÃO",
    "AN": "Quantidade de atendimentos odontológicos no pré-natal < 1",
    "AJ": "IG (DUM) (semanas) > 43",
}

# =========================
# Admin e mapeamento e-mail -> UNIDADE(s)
# =========================
ADMIN_EMAILS = {"vigilanciaepidemiologicadsvii@gmail.com"}

EMAIL_TO_UNITS: Dict[str, List[str]] = {
    "brunomaiatbehansen@gmail.com": ["Bruno Maia"],
    "alcidescodeceiratbehansen@gmail.com": ["Alto José Bonifacio"],
    "irmadenisetbehansen@gmail.com": ["Irmã Denise"],
    "altodoeucaliptotbehansen@gmail.com": ["Alto do Eucalipto"],
    "domheldertbehansen@gmail.com": ["Dom Helder"],
    "corregodabicatbehansen@gmail.com": ["Córrego da Bica"],
    "corregodoeucaliptotbehansen@gmail.com": ["Corrego do Eucalipto"],
    "heliomendoncatbehansen@gmail.com": ["Hélio Mendonça"],
    "clementinofragatbehansen@gmail.com": ["Clementino Fraga"],
    "guabirabatbehansen@gmail.com": ["Guabiraba"],
    "inarosaborgestbehansen@gmail.com": ["Iná Rosa Borges"],
    "passarinhoaltotbehansen@gmail.com": ["Passarinho Alto"],
    "passarinhobaixotbehansen@gmail.com": ["Passarinho Baixo"],
    "macaxeiratbehansen@gmail.com": ["Macaxeira"],
    "mangabeiratbehansen@gmail.com": ["Mangabeira"],
    "mariomonteirotbehansen@gmail.com": ["Mário Monteiro"],
    "bolanaredetbehansen@gmail.com": ["Bola na Rede"],
    "morrodaconceicaotbehansen@gmail.com": ["Morro da Conceição"],
    "santaterezatbehansen@gmail.com": ["Santa Tereza"],
    "moacyrandregomestbehansen@gmail.com": ["Moacyr André Gomes"],
    "vilaboavistatbehansen@gmail.com": ["Vila Boa vista"],
    "corregodoeuclidestbehansen@gmail.com": ["Maria Rita"],
    "sitiodosmacacostbehansen01@gmail.com": ["Sítio dos Macacos"],
    "diogenescavalcantitbehansen@gmail.com": ["Diógenes Cavalcanti"],
}

# =========================
# Helpers de normalização/ETL
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
            try:
                file_obj.seek(pos)
            except Exception:
                pass
    else:
        with open(file_obj, "rb") as fh:
            data = fh.read()

    if isinstance(data, bytes):
        try:
            text = data.decode("utf-8", "ignore")
        except Exception:
            text = data.decode("latin-1", "ignore")
    else:
        text = str(data)

    buf = StringIO(text)
    try:
        return pd.read_csv(buf, sep=None, engine="python", header=None, dtype=str,
                           skip_blank_lines=True, quoting=csv.QUOTE_MINIMAL, on_bad_lines="skip")
    except Exception:
        pass

    for sep in [";", "\t", ",", "|"]:
        buf.seek(0)
        try:
            return pd.read_csv(buf, sep=sep, engine="python", header=None, dtype=str,
                               skip_blank_lines=True, quoting=csv.QUOTE_MINIMAL, on_bad_lines="skip")
        except Exception:
            continue

    buf.seek(0)
    return pd.read_csv(buf, header=None, dtype=str, engine="python", on_bad_lines="skip")

def detect_unit_name(raw_df: pd.DataFrame) -> Optional[str]:
    # Heurísticas quando não há coluna de UNIDADE no arquivo
    try:
        b10 = raw_df.iloc[9, 1]
        if isinstance(b10, str) and b10.strip():
            return b10.strip()
    except Exception:
        pass
    try:
        a3 = raw_df.iloc[2, 0]
        if isinstance(a3, str) and normalize(a3).startswith("unidade de saude"):
            return a3.strip()
    except Exception:
        pass
    for i in range(min(10, len(raw_df))):
        try:
            v = str(raw_df.iloc[i, 0])
            if re.search(r"(?i)unidade\s+de\s+saud(e|é)", v):
                return v.strip()
        except Exception:
            continue
    return None

def find_unit_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "UNIDADE", "UNIDADE DE SAÚDE", "UNIDADE DE SAUDE", "UNIDADE (BA)", "BA",
        "ESTABELECIMENTO", "UNIDADE/ESTABELECIMENTO"
    ]
    cols_norm = {normalize(c): c for c in df.columns}
    for c in candidates:
        nc = normalize(c)
        if nc in cols_norm:
            return cols_norm[nc]
    for real in df.columns:
        rn = normalize(real)
        if "unidade" in rn or "estabelec" in rn:
            return real
    return None

def read_any_table(file) -> Tuple[pd.DataFrame, Optional[str]]:
    name = getattr(file, "name", str(file)).lower()
    if name.endswith(".csv"):
        raw = _read_csv_robusto(file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        raw = pd.read_excel(file, header=None, dtype=str)
    elif name.endswith(".ods"):
        try:
            raw = pd.read_excel(file, engine="odf", header=None, dtype=str)
        except Exception:
            raw = pd.read_excel(file, header=None, dtype=str)
    else:
        raw = _read_csv_robusto(file)

    unidade_hint = detect_unit_name(raw)

    # Detecta linha de cabeçalho (heurística)
    header_row = None
    for i in range(min(30, len(raw))):
        row = raw.iloc[i]
        non_na = row.dropna().astype(str)
        text_count = sum(len(str(x).strip()) > 0 for x in non_na)
        if text_count >= 4:
            header_row = i
            break
    if header_row is None:
        header_row = 0

    # Reconstrói DF
    df = raw.copy()
    df.columns = raw.iloc[header_row].fillna("")
    df = raw.iloc[header_row + 1:].reset_index(drop=True)
    df.columns = [str(c).strip() if str(c).strip() != "" else f"col_{i}" for i, c in enumerate(df.columns)]

    # Mantém linhas com paciente
    col_paciente = find_first_matching_col(df, COLMAP["paciente"]) or df.columns[0]
    df = df[df[col_paciente].notna() & (df[col_paciente].astype(str).str.strip() != "")]

    # UNIDADE: prioriza coluna do arquivo; se não houver, usa heurística
    col_unid = find_unit_column(df)
    if col_unid:
        df["UNIDADE"] = df[col_unid].astype(str).str.strip()
    else:
        df["UNIDADE"] = unidade_hint if unidade_hint else "(UNIDADE NAO DETECTADA)"
    return df, unidade_hint

def excel_letter_to_index(letter: str) -> int:
    if not letter or not letter.isalpha():
        return -1
    letter = letter.upper()
    idx = 0
    for ch in letter:
        idx = idx * 26 + (ord(ch) - ord('A') + 1)
    return idx - 1

def get_col_by_letter_or_name(df: pd.DataFrame, letter: str, names: List[str]) -> Optional[str]:
    j = excel_letter_to_index(letter)
    if 0 <= j < len(df.columns):
        col = df.columns[j]
        if str(col).strip() != "":
            return col
    names_n = [normalize(n) for n in names]
    for c in df.columns:
        cn = normalize(c)
        for nn in names_n:
            if cn == nn or nn in cn:
                return c
    return None

def series_is_nao(s: pd.Series) -> pd.Series:
    sn = s.astype(str).map(normalize)
    return sn.str.fullmatch(r"n[aã]o")

def to_numeric_series(s: pd.Series) -> pd.Series:
    ss = s.astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(ss, errors="coerce")

def build_requested_filters(df: pd.DataFrame) -> Tuple[Dict[str, pd.Series], Dict[str, str]]:
    masks, found_cols = {}, {}
    for code, spec in COL_SPECS.items():
        col = get_col_by_letter_or_name(df, code, spec.get("names", []))
        if col is None:
            continue
        found_cols[code] = col
        t = spec["type"]
        if t == "text_nao":
            masks[code] = series_is_nao(df[col])
        elif t == "num_lt":
            v = spec.get("value", 0)
            x = to_numeric_series(df[col])
            masks[code] = x < v
        elif t == "num_gt":
            v = spec.get("value", 0)
            x = to_numeric_series(df[col])
            masks[code] = x > v
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
# Login (Firebase)
# =========================
def login_block():
    show_login_banner()
    st.title("🔐 Login — GestaWeb DS7 (Firebase)")
    with st.form("login_form", clear_on_submit=False):
        email_in = st.text_input("E-mail institucional", value="", placeholder="nome@dominio")
        pwd_in   = st.text_input("Senha", value="", type="password")
        ok = st.form_submit_button("Entrar")
    if ok:
        try:
            fb = firebase_init()
            user = firebase_sign_in(fb, email_in.strip(), pwd_in)
            st.session_state["auth_ok"] = True
            st.session_state["user_email"] = email_in.strip().lower()
            st.session_state["id_token"] = user.get("idToken")

            # Unidades permitidas a este e-mail
            if st.session_state["user_email"] in ADMIN_EMAILS:
                st.session_state["allowed_units"] = ["*"]  # admin vê tudo
            else:
                allowed = EMAIL_TO_UNITS.get(st.session_state["user_email"], [])
                st.session_state["allowed_units"] = allowed

            st.success("Login realizado.")
            st.rerun()
        except Exception as e:
            st.error(f"Falha no login: {e}")
    st.stop()

# =========================
# Gate de login
# =========================
if not st.session_state.get("auth_ok"):
    login_block()

# =========================
# Barra lateral (logout)
# =========================
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

# ===== Carregamento de dados =====
uploaded_files = None
if is_admin:
    uploaded_files = st.file_uploader(
        "Selecione as planilhas (até 69) — apenas ADMIN",
        type=["csv", "xlsx", "xls", "ods"],
        accept_multiple_files=True,
    )
else:
    st.caption("Upload desabilitado para usuários não-admin.")

# Se admin enviar, atualiza a base na sessão e salva no disco
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
                st.error(f"Erro ao ler {getattr(f, 'name', 'arquivo')}: {e}")
        if not dfs:
            st.stop()
        
        # Consolida os dados
        consolidated_df = pd.concat(dfs, ignore_index=True)
        st.session_state["base_df"] = consolidated_df
        
        # 1. Persistência em disco (para as próximas sessões/usuários)
        save_base_to_disk(consolidated_df)
        st.success("Base atualizada e salva em disco (disponível para todos os usuários).")
        
        # 2. Persistência no Firestore
        save_to_firestore(consolidated_df.copy(), collection_name="gestantes")


# Recupera base da sessão ou do disco
if "base_df" not in st.session_state:
    disk_df = load_base_from_disk()
    if disk_df is not None:
        st.session_state["base_df"] = disk_df
    else:
        st.warning("Nenhuma base carregada ainda. O ADMIN precisa carregar as planilhas.")
        st.stop()

base = st.session_state["base_df"].copy()

# Enforce lock para usuários não-admin (comparação tolerante a acentos/maiúsculas)
def _norm_view(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.strip().lower()

if not is_admin:
    if not allowed_units:
        st.warning("Seu e-mail autenticou, mas não está associado a nenhuma UNIDADE. Solicite cadastro.")
        st.stop()
    allowed_norm = {_norm_view(u) for u in allowed_units}
    base = base[base["UNIDADE"].astype(str).map(lambda x: _norm_view(x) in allowed_norm)]
    if base.empty:
        st.warning("Não há registros para a(s) sua(s) unidade(s) na base carregada.")
        st.stop()
    st.info("Visualizando somente a(s) sua(s) unidade(s): " + ", ".join(sorted(set(allowed_units))))

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

c1, c2, _ = st.columns([1, 1, 3])
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

# Fonte para relatórios (respeita filtro quando ligado)
df_fonte_relatorios = base_filtrada if (st.session_state["unit_filter_on"] and sel) else base

# ===== Relatórios específicos =====
st.subheader("Relatórios específicos (mantendo TODAS as colunas)")
masks, found_cols = build_requested_filters(df_fonte_relatorios)

if found_cols:
    st.caption("Colunas detectadas (código → nome real): " +
               ", ".join([f"{k}→{v}" for k, v in found_cols.items()]))
else:
    st.info("AR/AS/AT/AU/AV/AN/AJ não localizadas por letra nem por nome — verifique o cabeçalho.")

tables_spec: Dict[str, pd.DataFrame] = {}
for code in ["AR", "AS", "AT", "AU", "AV", "AN", "AJ"]:
    if code in masks:
        df_tab = df_fonte_relatorios[masks[code]].copy()
        tables_spec[code] = df_tab
        with st.expander(f"{LABELS[code]} — {len(df_tab)} linha(s)"):
            st.dataframe(df_tab.head(300), use_container_width=True, height=360)
            cdl1, cdl2 = st.columns(2)
            with cdl1:
                st.download_button(
                    f"Baixar {code}.csv",
                    data=df_to_csv_bytes(df_tab),
                    file_name=f"{code}.csv",
                    mime="text/csv",
                    key=f"csv_{code}"
                )
            with cdl2:
                st.download_button(
                    f"Baixar {code}.xlsx",
                    data=single_sheet_xlsx(df_tab, sheet_name=code),
                    file_name=f"{code}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"xlsx_{code}"
                )
    else:
        st.info(f"Filtro {LABELS.get(code, code)}: coluna {code} não localizada.")

# ===== Análise por critério + gráfico por unidade =====

st.subheader("Análise por critério e Unidade de Saúde")

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", s).strip().lower()

def resolve_crit_key(selection_text: str) -> Optional[str]:
    """Mapeia o texto mostrado no selectbox para o código (AR/AS/.../AJ)."""
    sel_n = _norm(selection_text)
    for code, spec in COL_SPECS.items():
        # 1) casa exatamente com a descrição oficial
        if sel_n == _norm(spec.get("desc", "")):
            return code
        # 2) casa com qualquer sinônimo declarado em names
        for nm in spec.get("names", []):
            if sel_n == _norm(nm):
                return code
        # 3) fallback: contém (para tolerar pequenas variações)
        if _norm(spec.get("desc", "")) in sel_n:
            return code
        for nm in spec.get("names", []):
            if _norm(nm) in sel_n:
                return code
    return None

# Mostre as opções amigáveis a partir dos próprios specs (primeiro name + desc)
crit_options = []
for code, spec in COL_SPECS.items():
    # use o primeiro sinônimo, senão a descrição
    label = (spec.get("names") or [spec.get("desc", code)])[0]
    crit_options.append(label)
# Garanta ordem consistente
crit_options = sorted(set(crit_options), key=lambda x: _norm(x))

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
    sel_unid_crit = st.multiselect(
        "Filtrar por Unidade de Saúde (apenas para este critério)",
        options=unidades_crit,
        default=unidades_crit,
    )
    if sel_unid_crit:
        df_crit = df_crit[df_crit["UNIDADE"].isin(sel_unid_crit)]

    st.write(f"{LABELS[crit_key]} — total: {len(df_crit)} registro(s)")

    with st.expander("Ver linhas deste critério"):
        st.dataframe(df_crit.head(300), use_container_width=True, height=320)
        st.download_button(
            "Baixar CSV (critério filtrado)",
            data=df_to_csv_bytes(df_crit),
            file_name=f"{crit_key}_criterio_filtrado.csv",
            mime="text/csv",
        )

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
            tooltip=[
                alt.Tooltip("UNIDADE:O", title="Unidade"),
                alt.Tooltip("quantidade:Q", title="Quantidade"),
            ],
        )
        st.altair_chart(chart.properties(height=420, width="container"), use_container_width=True)


# ===== Resumo e comparativo por prefixo (opcional) =====
TARGET_PREFIXES = [
    "ALTO DO EUCALIPTO","ALTO JOSE BONIFACIO","ESF MAIS BRUNO MAIA","ESF MAIS CORREGO DA BICA","BOLA NA REDE",
    "ALTO DA BRASILEIRA","CORREGO DO EUCALIPTO","ESF MAIS CORREGO JENIPAPO","ESF MAIS DOM HELDER","GUABIRABA",
    "INA ROSA","ESF MAIS MACAXEIRA","MANGABEIRA","ESF MAIS MARIA RITA","MARIO MONTEIRO","ESF MAIS MOACYR",
    "MORRO DA CONCEICAO","PASSARINHO ALTO","PASSARINHO BAIXO","ALTO JOSE DO PINHO","ALTO DO RESERVATORIO",
    "SANTA TEREZA","SITIOS DOS MACACOS",
]
def _norm_prefix(s: str) -> str:
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", s).strip().lower()
TARGET_PREFIXES_N = [_norm_prefix(p) for p in TARGET_PREFIXES]
def _matches_any_prefix(u: str) -> bool:
    nu = _norm_prefix(u)
    return any(nu.startswith(pn) for pn in TARGET_PREFIXES_N)
def unit_to_bucket(u: str) -> Optional[str]:
    nu = _norm_prefix(u)
    for raw_prefix, norm_prefix in zip(TARGET_PREFIXES, TARGET_PREFIXES_N):
        if nu.startswith(norm_prefix):
            return raw_prefix
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
        codigos_para_resumo = ["AR", "AS", "AT", "AU", "AV", "AN", "AJ"]
        contagens = []
        idx_grupo = df_grupo.index
        for code in codigos_para_resumo:
            if code in masks:
                mask_total = masks[code]
                total = int(mask_total.loc[mask_total.index.intersection(idx_grupo)].sum())
            else:
                total = 0
            contagens.append({"codigo": code, "criterio": LABELS.get(code, code), "quantidade": total})
        df_resumo = pd.DataFrame(contagens).sort_values("codigo")

        with st.expander("Tabela de totais no grupo de unidades (por prefixo)"):
            st.dataframe(df_resumo[["criterio", "quantidade"]], use_container_width=True, height=300)
            st.download_button(
                "Baixar CSV (resumo do grupo por prefixo)",
                data=df_to_csv_bytes(df_resumo[["criterio", "quantidade"]]),
                file_name="resumo_grupo_unidades_prefixo.csv",
                mime="text/csv",
            )

        st.caption("Gráfico — Totais por critério (grupo de unidades por prefixo)")
        chart_resumo = alt.Chart(df_resumo).mark_bar().encode(
            x=alt.X("criterio:N", sort=df_resumo["criterio"].tolist(), title="Critério"),
            y=alt.Y("quantidade:Q", title="Quantidade"),
            tooltip=[
                alt.Tooltip("criterio:N", title="Critério"),
                alt.Tooltip("quantidade:Q", title="Quantidade"),
            ],
        )
        st.altair_chart(chart_resumo.properties(height=420, width="container"), use_container_width=True)

        # ===== Comparativo — por unidade consolidada (prefixo) =====
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
                    grp_b = pd.DataFrame({"bucket": sorted(df_all["bucket"].unique()), "quantidade": 0})
                grp_b["codigo"] = code
                grp_b["criterio"] = LABELS.get(code, code)
                linhas.append(grp_b)

            df_barras = pd.concat(linhas, ignore_index=True) if linhas else pd.DataFrame(
                columns=["bucket", "quantidade", "codigo", "criterio"]
            )

            ordem_buckets = [b for b in TARGET_PREFIXES if b in df_barras["bucket"].unique().tolist()]
            restantes = [b for b in df_barras["bucket"].unique().tolist() if b not in ordem_buckets]
            ordem_buckets = ordem_buckets + sorted(restantes)

            with st.expander("Tabela — contagens por unidade (consolidada por prefixo) e critério"):
                st.dataframe(df_barras, use_container_width=True, height=360)
                st.download_button(
                    "Baixar CSV (unidade consolidada × critério)",
                    data=df_to_csv_bytes(df_barras),
                    file_name="comparativo_unidades_consolidadas.csv",
                    mime="text/csv",
                )

            chart_buckets = alt.Chart(df_barras).mark_bar().encode(
                x=alt.X("bucket:O", sort=ordem_buckets, title="Unidade (agrupada por prefixo)"),
                y=alt.Y("quantidade:Q", title="Quantidade"),
                color=alt.Color("criterio:N", title="Critério"),
                xOffset=alt.X("criterio:N"),
                tooltip=[
                    alt.Tooltip("bucket:O", title="Unidade"),
                    alt.Tooltip("criterio:N", title="Critério"),
                    alt.Tooltip("quantidade:Q", title="Quantidade"),
                ],
            )
            st.altair_chart(chart_buckets.properties(height=440, width="container"), use_container_width=True)