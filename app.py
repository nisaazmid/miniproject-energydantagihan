import io
import re

import matplotlib.pyplot as plt
import pandas as pd
import sqlglot
import streamlit as st
from google import genai
from google.genai import types
from sqlalchemy import create_engine, text
from sqlglot import exp


# ============================================================
# Konfigurasi aplikasi
# ============================================================

st.set_page_config(
    page_title="ASYIK",
    page_icon="⚡",
    layout="wide",
)

SCHEMA_STR = """
costumers(cust_id, nama, tarif, wilayah)
usage(cust_id, bulan, kwh, tagihan, status_bayar)

Relasi:
- usage.cust_id -> costumers.cust_id

Catatan:
- bulan berformat YYYY-MM, misalnya 2026-01
- status_bayar berisi lunas atau tertunggak
""".strip()

ALLOWED_TABLES = {"costumers", "usage"}
DEFAULT_LIMIT = 200
MAX_LIMIT = 1000
QUERY_TIMEOUT_MS = 8_000
TEAL = "#0E8388"


# ============================================================
# Secrets
# ============================================================

def load_secrets() -> tuple[str, str, str]:
    """Ambil seluruh konfigurasi hanya dari Streamlit Secrets."""
    try:
        db_url = st.secrets["DB_URL"]
        google_api_key = st.secrets["GOOGLE_API_KEY"]
        gemini_model = st.secrets.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
    except (KeyError, FileNotFoundError) as exc:
        st.error(
            "Secrets belum lengkap. Tambahkan DB_URL dan GOOGLE_API_KEY "
            "pada Streamlit App Settings → Secrets."
        )
        st.stop()
        raise RuntimeError("Streamlit secrets tidak lengkap") from exc

    return str(db_url), str(google_api_key), str(gemini_model)


DB_URL, GOOGLE_API_KEY, GEMINI_MODEL = load_secrets()


# ============================================================
# Resource initialization
# ============================================================

@st.cache_resource
def get_database_engine(db_url: str):
    """Buat koneksi SQLAlchemy ke PostgreSQL Supabase."""
    engine = create_engine(
        db_url,
        pool_pre_ping=True,
        pool_recycle=300,
        connect_args={"sslmode": "require"},
    )

    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))

    return engine


@st.cache_resource
def get_gemini_client(api_key: str):
    """Buat Gemini client satu kali untuk seluruh sesi aplikasi."""
    return genai.Client(api_key=api_key)


try:
    db_engine = get_database_engine(DB_URL)
except Exception as exc:
    st.error(f"Gagal terhubung ke database Supabase: {exc}")
    st.stop()

gemini_client = get_gemini_client(GOOGLE_API_KEY)


# ============================================================
# SQL generation and validation
# ============================================================

def build_prompt(question: str) -> str:
    return f"""
Anda adalah pembuat query PostgreSQL.

Skema database:
{SCHEMA_STR}

Aturan:
1. Buat tepat satu query read-only.
2. Query utama harus berupa SELECT; WITH/CTE boleh digunakan bila diperlukan.
3. Hanya gunakan tabel costumers dan usage.
4. Gunakan JOIN berdasarkan relasi yang tersedia.
5. Jangan gunakan markdown, penjelasan, atau code fence.
6. Balas hanya dengan query SQL.

Pertanyaan pengguna:
{question}
""".strip()


def extract_sql(response_text: str) -> str:
    """Ambil SQL dari respons model dan buang code fence bila masih muncul."""
    if not response_text or not response_text.strip():
        raise ValueError("Gemini tidak mengembalikan query SQL.")

    sql = response_text.strip()

    fenced_match = re.search(
        r"```(?:sql)?\s*(.*?)```",
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced_match:
        sql = fenced_match.group(1).strip()

    query_match = re.search(
        r"\b(?:WITH|SELECT)\b.*",
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not query_match:
        raise ValueError("Respons Gemini tidak mengandung query SELECT/WITH.")

    return query_match.group(0).strip().rstrip(";").strip()


def generate_sql(question: str) -> str:
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=build_prompt(question),
        config=types.GenerateContentConfig(
            temperature=0,
        ),
    )
    return extract_sql(response.text or "")


def _get_limit_value(limit_expression: exp.Limit | None) -> int | None:
    if limit_expression is None:
        return None

    value_expression = limit_expression.expression
    if not isinstance(value_expression, exp.Literal):
        return None

    try:
        return int(value_expression.this)
    except (TypeError, ValueError):
        return None


def validate_sql(
    sql: str,
    default_limit: int = DEFAULT_LIMIT,
    max_limit: int = MAX_LIMIT,
) -> str:
    """
    Parse SQL sebagai PostgreSQL, pastikan hanya query SELECT,
    batasi tabel, dan pasang batas jumlah baris.
    """
    try:
        statements = sqlglot.parse(sql, read="postgres")
    except sqlglot.errors.ParseError as exc:
        raise ValueError(f"SQL tidak valid: {exc}") from exc

    if len(statements) != 1:
        raise ValueError("Hanya satu statement SQL yang diperbolehkan.")

    query = statements[0]

    # WITH ... SELECT tetap diparse sebagai Select.
    if not isinstance(query, exp.Select):
        raise ValueError("Hanya query SELECT/WITH yang diperbolehkan.")

    forbidden_types = (
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Create,
        exp.Drop,
        exp.Alter,
        exp.Command,
        exp.Copy,
        exp.Merge,
        exp.TruncateTable,
    )

    if any(isinstance(node, forbidden_types) for node in query.walk()):
        raise ValueError("Query mengandung operasi yang tidak diperbolehkan.")

    cte_names = {
        cte.alias_or_name.lower()
        for cte in query.find_all(exp.CTE)
        if cte.alias_or_name
    }

    referenced_tables = {
        table.name.lower()
        for table in query.find_all(exp.Table)
        if table.name and table.name.lower() not in cte_names
    }

    unknown_tables = referenced_tables - ALLOWED_TABLES
    if unknown_tables:
        raise ValueError(
            "Tabel tidak diperbolehkan: "
            + ", ".join(sorted(unknown_tables))
        )

    current_limit = _get_limit_value(query.args.get("limit"))

    if current_limit is None:
        query = query.limit(default_limit, copy=False)
    elif current_limit > max_limit:
        query = query.limit(max_limit, copy=False)

    return query.sql(dialect="postgres")


def run_sql(sql: str) -> pd.DataFrame:
    """
    Jalankan query dalam transaksi read-only dengan statement timeout.
    """
    with db_engine.connect() as connection:
        transaction = connection.begin()

        try:
            connection.execute(text("SET TRANSACTION READ ONLY"))
            connection.execute(
                text(f"SET LOCAL statement_timeout = '{QUERY_TIMEOUT_MS}ms'")
            )
            result = pd.read_sql_query(text(sql), connection)
            transaction.commit()
            return result
        except Exception:
            transaction.rollback()
            raise


# ============================================================
# Visualization
# ============================================================

def choose_chart_type(
    dataframe: pd.DataFrame,
    question: str,
    x_column: str,
) -> str:
    question_lower = question.lower()
    x_lower = x_column.lower()

    if any(keyword in question_lower for keyword in ("pie", "komposisi", "proporsi")):
        return "pie"

    time_keywords = ("bulan", "tanggal", "periode", "tahun", "date", "month", "year")
    if any(keyword in x_lower for keyword in time_keywords):
        return "line"

    if any(keyword in question_lower for keyword in ("tren", "trend")):
        return "line"

    return "bar"


def create_visualization(
    dataframe: pd.DataFrame,
    question: str,
):
    if dataframe.empty or len(dataframe.columns) < 2:
        return None

    numeric_columns = list(
        dataframe.select_dtypes(include="number").columns
    )
    if not numeric_columns:
        return None

    y_column = numeric_columns[-1]
    x_candidates = [
        column for column in dataframe.columns if column != y_column
    ]
    if not x_candidates:
        return None

    x_column = x_candidates[0]
    chart_type = choose_chart_type(
        dataframe=dataframe,
        question=question,
        x_column=str(x_column),
    )

    chart_data = dataframe[[x_column, y_column]].dropna().copy()
    if chart_data.empty:
        return None

    if chart_type == "line":
        chart_data = chart_data.sort_values(by=x_column)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    try:
        if chart_type == "line":
            ax.plot(
                chart_data[x_column].astype(str),
                chart_data[y_column],
                marker="o",
                color=TEAL,
            )
            ax.set_ylabel(str(y_column))
            ax.tick_params(axis="x", rotation=30)

        elif chart_type == "pie":
            values = pd.to_numeric(
                chart_data[y_column],
                errors="coerce",
            ).fillna(0)

            if values.sum() <= 0:
                plt.close(fig)
                return None

            ax.pie(
                values,
                labels=chart_data[x_column].astype(str),
                autopct="%1.0f%%",
            )

        else:
            ax.bar(
                chart_data[x_column].astype(str),
                chart_data[y_column],
                color=TEAL,
            )
            ax.set_ylabel(str(y_column))
            ax.tick_params(axis="x", rotation=30)

        ax.set_title(question or f"{y_column} per {x_column}")
        fig.tight_layout()
        return fig

    except Exception:
        plt.close(fig)
        raise


def figure_to_png(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=150,
        bbox_inches="tight",
    )
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# Chat rendering
# ============================================================

def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
            return

        if message.get("error"):
            st.error(message["error"])
            return

        if message.get("sql"):
            st.markdown("**Generated SQL**")
            st.code(message["sql"], language="sql")

        dataframe = message.get("dataframe")
        if isinstance(dataframe, pd.DataFrame):
            st.dataframe(
                dataframe,
                use_container_width=True,
                hide_index=True,
            )

        chart_png = message.get("chart_png")
        if chart_png:
            st.image(chart_png, use_container_width=True)
        elif message.get("show_no_chart"):
            st.info("Tidak ada visualisasi yang sesuai untuk hasil ini.")


# ============================================================
# Streamlit interface
# ============================================================

st.title("⚡ ASYIK - Asisten Yang Kamu Inginkan")

if "messages" not in st.session_state:
    st.session_state.messages = []

for saved_message in st.session_state.messages:
    render_message(saved_message)

if user_question := st.chat_input("Ada pertanyaan lain?"):
    user_message = {
        "role": "user",
        "content": user_question,
    }
    st.session_state.messages.append(user_message)
    render_message(user_message)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Menganalisis pertanyaan..."):
                generated_sql = generate_sql(user_question)
                validated_sql = validate_sql(generated_sql)

                st.markdown("**Generated SQL**")
                st.code(validated_sql, language="sql")

                dataframe = run_sql(validated_sql)
                st.dataframe(
                    dataframe,
                    use_container_width=True,
                    hide_index=True,
                )

                fig = create_visualization(
                    dataframe=dataframe,
                    question=user_question,
                )

                chart_png = None
                if fig is not None:
                    st.pyplot(fig, use_container_width=True)
                    chart_png = figure_to_png(fig)
                    plt.close(fig)
                else:
                    st.info("Tidak ada visualisasi yang sesuai untuk hasil ini.")

                assistant_message = {
                    "role": "assistant",
                    "sql": validated_sql,
                    "dataframe": dataframe,
                    "chart_png": chart_png,
                    "show_no_chart": fig is None,
                }

        except Exception as exc:
            error_message = (
                f"Terjadi kesalahan: {exc}\n\n"
                "Periksa pertanyaan, konfigurasi Secrets, dan koneksi database."
            )
            st.error(error_message)
            assistant_message = {
                "role": "assistant",
                "error": error_message,
            }

    st.session_state.messages.append(assistant_message)
