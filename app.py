import streamlit as st
import os, re, glob, zipfile
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
import gdown # Used for downloading data if not present
import google.generativeai as genai
import json # Used in generate_sql, although current notebook implementation uses resp.text

# === Global Constants from Notebook ===
SCHEMA_STR = """customers(cust_id, nama, tarif, wilayah)
usage(cust_id, bulan, kwh, tagihan, status_bayar)

Relasi:
- usage.cust_id -> customers.cust_id
Catatan: kolom 'bulan' berformat 'YYYY-MM' (mis. '2026-01').
         status_bayar berisi 'lunas' atau 'tertunggak'."""

DIALEK = 'PostgreSQL'
TERLARANG = ('drop','delete','update','insert','alter','truncate','create','replace',
             'grant','revoke','merge','into','attach','detach','pragma','vacuum','copy','dblink')
POLA_BAHAYA = (r'\binformation_schema\b', r'\bpg_catalog\b', r'\bpg_\w+\b',
               r'\bsqlite_master\b', r'\bload_extension\b', r'\blo_import\b', r'\blo_export\b')
_FUNGSI_FROM = r'\b(?:extract|substring|trim|position|overlay)\s*\([^)]*\)'
TABEL_OK = {'customers','usage'}
TEAL = '#0E8388'

# --- LLM Setup Configuration ---
# In Streamlit, it's best practice to get API keys from st.secrets or environment variables.
PROVIDER = 'mock' # Default to mock if no API key is found
GEMINI_MODEL = 'gemini-1.5-flash' # Using the original model name from notebook's TODO 1

# Try to get GOOGLE_API_KEY from st.secrets first for Streamlit Cloud deployment
if st.secrets.get('GOOGLE_API_KEY'):
    PROVIDER = 'gemini'

USE_MOCK = (PROVIDER == 'mock')

# --- Functions (adapted from notebook) ---

@st.cache_resource
def setup_database(db_url):
    """
    Sets up the PostgreSQL engine and loads data. This function is cached
    to avoid re-running on every Streamlit rerun.
    """
    st.info("Initializing database connection and loading data...")
    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1")) # Test connection
            st.success(f"PostgreSQL connected to {db_url.split('@')[-1]}")

        CSV_FILES = ['customers.csv', 'usage.csv']
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)

        # Assuming the ZIP download was already handled in the Colab setup
        # If running this Streamlit app independently, you might need to
        # re-implement the gdown logic or ensure files are pre-staged.
        # For this example, we assume data/customers.csv and data/usage.csv exist.
        
        # Check if files exist, if not, attempt to download with default IDs
        # These default IDs are from the initial notebook setup for Use Case B
        if not os.path.exists(os.path.join(data_dir, "dataset.zip")):
            DRIVE_ZIP_URL = "https://drive.google.com/uc?id=1KozrOgW1ChAKK2zHM1o-r_BPt08-zKR2"
            try:
                gdown.download(DRIVE_ZIP_URL, os.path.join(data_dir, "dataset.zip"), quiet=True)
                with zipfile.ZipFile(os.path.join(data_dir, "dataset.zip")) as z:
                    z.extractall(data_dir)
                st.info("Dataset ZIP downloaded and extracted.")
            except Exception as dl_e:
                st.warning(f"Could not download dataset ZIP: {dl_e}. Assuming CSVs exist or will be manually placed.")

        for fn in CSV_FILES:
            table = fn[:-4]
            filepath = os.path.join(data_dir, fn)

            if not os.path.exists(filepath):
                 st.error(f"File {filepath} not found. Please ensure data files are available in the 'data' directory.")
                 continue # Skip this table if file not found

            df = pd.read_csv(filepath)
            df.to_sql(table, engine, if_exists="replace", index=False)
            st.success(f"Tabel '{table}' dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")

        return engine
    except Exception as e:
        st.error(f"Failed to setup database: {e}. Please ensure PostgreSQL is running and accessible and DB_URL is correct.")
        return None

def build_prompt(question: str) -> str:
    prompt = f"""Anda ahli SQL {DIALEK}
              skema db: {SCHEMA_STR}
              Buat SATU query SELECT (JOIN bila perlu). Balas HANYA query SQL.
              Pertanyaan: {question}
              """
    return prompt

def generate_sql(resp):
    # This function expects a string (resp.text from LLM) or a dict (for mock)
    if isinstance(resp, dict): # For mock responses that might pass a dict
        resp = resp.get('sql', json.dumps(resp))
    teks = str(resp).strip()
    m = re.search(r'```(?:sql)?\s*(.+?)```', teks, re.S)
    if m: teks = m.group(1).strip()
    m = re.search(r'(select\b.+)', teks, re.I | re.S)
    if m: teks = m.group(1)
    return teks.rstrip(';').strip()

def _strip_komentar(sql):
    sql = re.sub(r'/\*.*?\*/', ' ', sql, flags=re.S)
    return re.sub(r'--[^\n]*', ' ', sql)

def validasi_sql(sql, batas=200, batas_maks=1000):
    t = _strip_komentar(sql).strip().rstrip(';').strip(); low = t.lower()
    if not (low.startswith('select') or low.startswith('with')): raise ValueError('Hanya SELECT/WITH')
    if ';' in t: raise ValueError('Multi-statement')
    for k in TERLARANG:
        if re.search(rf'\b{k}\b', low): raise ValueError(f'Terlarang: {k}')
    for pola in POLA_BAHAYA:
        m = re.search(pola, low)
        if m: raise ValueError(f'Objek terlarang: {m.group()}')
    low_tab = re.sub(_FUNGSI_FROM, ' ', low)
    asing = set(re.findall(r'(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)', low_tab)) - TABEL_OK
    if asing: raise ValueError(f'Tabel tak dikenal: {asing}')
    m = re.search(r'\blimit\s+(\d+)', low)
    if m:
        if int(m.group(1)) > batas_maks: t = re.sub(r'\blimit\s+\d+', f'LIMIT {batas_maks}', t, flags=re.I)
    else:
        t += f' LIMIT {batas}'
    return t

def run_sql(sql: str, engine_obj) -> pd.DataFrame:
    with engine_obj.connect() as conn:
        return pd.read_sql(text(sql), conn)

def visualize(df, pertanyaan='', jenis=None):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None # No data to visualize

    # Determine chart type based on columns and question
    if len(df.columns) < 2:
        return None # Need at least 2 columns for meaningful chart

    x_col = df.columns[0]
    y_col = df.columns[-1]

    # Try to infer type if not explicitly given
    if jenis is None:
        p = pertanyaan.lower()
        x_col_lower = str(x_col).lower()

        if 'pie' in p or 'komposisi' in p or 'proporsi' in p:
            jenis = 'pie'
        elif 'periode' in x_col_lower or 'bulan' in p or 'tren' in p:
            jenis = 'line'
        else:
            jenis = 'bar'

    fig, ax = plt.subplots(figsize=(7, 4))
    try:
        if jenis == 'line':
            ax.plot(df[x_col].astype(str), df[y_col], marker='o', color=TEAL)
            ax.set_ylabel(str(y_col))
            plt.xticks(rotation=30, ha='right')
        elif jenis == 'pie':
            if df[y_col].sum() == 0:
                return None # Avoid division by zero
            ax.pie(df[y_col], labels=df[x_col].astype(str), autopct='%1.0f%%',
                   colors=plt.cm.Greens([0.4,0.55,0.7,0.85,0.6,0.45]))
        else: # Default to bar chart
            ax.bar(df[x_col].astype(str), df[y_col], color=TEAL)
            ax.set_ylabel(str(y_col))
            plt.xticks(rotation=30, ha='right')
        ax.set_title(pertanyaan or f'{y_col} per {x_col}')
        plt.tight_layout()
        return fig
    except Exception as e:
        st.warning(f"Could not generate visualization: {e}")
        plt.close(fig) # Close figure if an error occurs to prevent memory leak
        return None

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("⚡️ ASYIK - Asisten Yang kamu Inginkan")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Get DB_URL from st.secrets
DB_URL = st.secrets.get('DB_URL', 'postgresql://postgres:postgres@localhost:5432/miniproject')
if not DB_URL:
    st.error("DB_URL not found in Streamlit secrets or environment variables. Please provide your PostgreSQL connection string.")
    st.stop()

# Setup database (cached function)
if "db_engine" not in st.session_state:
    st.session_state.db_engine = setup_database(DB_URL)
    if st.session_state.db_engine is None:
        st.error("Database setup failed. Please check the console for errors and ensure DB_URL is correct.")
        st.stop() # Stop the app if DB setup fails

# Initialize LLM model
if "llm_model" not in st.session_state:
    if USE_MOCK:
        st.session_state.llm_model = "mock" # Simple string for mock model
    elif PROVIDER == 'gemini':
        gemini_api_key = st.secrets.get('GOOGLE_API_KEY')
        if not gemini_api_key:
            st.error("Google API Key (GOOGLE_API_KEY) not found in Streamlit secrets or environment variables. Please set it.")
            st.stop()
        try:
            genai.configure(api_key=gemini_api_key)
            st.session_state.llm_model = genai.GenerativeModel(GEMINI_MODEL)
        except Exception as e:
            st.error(f"Failed to initialize Gemini model: {e}")
            st.stop()
    else:
        st.error(f"LLM provider '{PROVIDER}' not supported or API key missing.")
        st.stop()
llm_model = st.session_state.llm_model

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["type"] == "text":
            st.markdown(message["content"])
        elif message["type"] == "df":
            st.dataframe(message["content"])
        elif message["type"] == "sql":
            st.markdown("**Generated SQL:**")
            st.code(message["content"], language="sql")
        elif message["type"] == "plot":
            st.pyplot(message["content"])
            plt.close(message["content"]) # Close the figure to free memory

# React to user input
if prompt := st.chat_input("Ada pertanyaan lain?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "type": "text", "content": prompt})

    with st.chat_message("assistant"):
        status_message = st.empty()
        status_message.info("Menganalisis pertanyaan...")

        sql_generated = ""
        df_result = pd.DataFrame()
        plot_fig = None

        try:
            attempts = 0
            while attempts < 2:
                try:
                    llm_prompt = build_prompt(prompt)
                    if USE_MOCK:
                        if "total konsumsi kWh per wilayah pada bulan Januari (2026-01)" in prompt:
                            sql_generated = "SELECT c.wilayah, SUM(u.kwh) AS total_kwh FROM customers c JOIN usage u ON c.cust_id = u.cust_id WHERE u.bulan = '2026-01' GROUP BY c.wilayah ORDER BY total_kwh DESC"
                        elif "10 pelanggan dengan total tunggakan tertinggi" in prompt:
                            sql_generated = "SELECT c.nama, SUM(u.tagihan) AS total_tunggakan FROM customers c JOIN usage u ON c.cust_id = u.cust_id WHERE u.status_bayar = 'tertunggak' GROUP BY c.nama ORDER BY total_tunggakan DESC LIMIT 10"
                        elif "tren total tagihan selama 6 bulan terakhir" in prompt:
                            sql_generated = "SELECT bulan, SUM(tagihan) AS total_tagihan FROM usage GROUP BY bulan ORDER BY bulan DESC LIMIT 6"
                        else:
                            sql_generated = f"SELECT 'Mock SQL result for: {prompt}' AS result;"
                    else:
                        resp = llm_model.generate_content(llm_prompt)
                        sql_generated = generate_sql(resp.text)

                    validated_sql = validasi_sql(sql_generated)
                    sql_generated = validated_sql
                    break
                except ValueError as e:
                    status_message.warning(f"Validasi SQL gagal (Percobaan {attempts+1}): {e}")
                    if attempts == 1:
                        raise
                    attempts += 1
                except Exception as e:
                    status_message.error(f"Gagal generate SQL (Percobaan {attempts+1}): {e}")
                    if attempts == 1:
                        raise
                    attempts += 1

            if not sql_generated:
                raise Exception("Could not generate valid SQL after retries.")

            status_message.info("SQL yang dihasilkan:")
            st.code(sql_generated, language="sql")
            st.session_state.messages.append({"role": "assistant", "type": "sql", "content": sql_generated})

            status_message.info("Mengeksekusi SQL...")
            df_result = run_sql(sql_generated, st.session_state.db_engine)
            status_message.empty()
            st.dataframe(df_result)
            st.session_state.messages.append({"role": "assistant", "type": "df", "content": df_result})

            plot_fig = visualize(df_result, prompt)
            if plot_fig:
                st.pyplot(plot_fig)
                st.session_state.messages.append({"role": "assistant", "type": "plot", "content": plot_fig})
                plt.close(plot_fig)
            else:
                st.info("Tidak ada visualisasi yang sesuai untuk data ini.")

        except Exception as e:
            status_message.error(f"⚠️ Terjadi kesalahan: {e}\nSilakan periksa kembali pertanyaan atau konfigurasi.")
            st.session_state.messages.append({"role": "assistant", "type": "text", "content": f"⚠️ Terjadi kesalahan: {e}\nSilakan periksa kembali pertanyaan atau konfigurasi."})
