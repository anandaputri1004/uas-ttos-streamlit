import re
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datasets import load_dataset
from nltk.stem.porter import PorterStemmer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from wordcloud import WordCloud

# ----------------------------------------------------------------------------
# KONFIGURASI HALAMAN
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Analisis Sentimen Review Restoran",
    page_icon="🍽️",
    layout="wide",
)

# ----------------------------------------------------------------------------
# PREPROCESSING TEKS (sama seperti pipeline pada notebook)
# ----------------------------------------------------------------------------
STEMMER = PorterStemmer()

CONTRACTIONS = {
    "didn't": "did not", "won't": "will not", "isn't": "is not",
    "wasn't": "was not", "aren't": "are not", "doesn't": "does not",
    "don't": "do not", "can't": "can not", "couldn't": "could not",
    "wouldn't": "would not", "shouldn't": "should not",
}

NEGATION_WORDS = {"not", "no", "never", "nor", "none", "cannot"}
CUSTOM_STOPWORDS = ENGLISH_STOP_WORDS - NEGATION_WORDS


def expand_contractions(text: str) -> str:
    for k, v in CONTRACTIONS.items():
        text = text.replace(k, v)
    return text


def preprocess(text: str) -> str:
    text = expand_contractions(text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in CUSTOM_STOPWORDS and len(t) > 1]
    tokens = [STEMMER.stem(t) for t in tokens]
    return " ".join(tokens)


# ----------------------------------------------------------------------------
# LOAD DATA & TRAIN MODEL (di-cache supaya tidak diulang setiap interaksi)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Mengunduh & memuat dataset...")
def load_data() -> pd.DataFrame:
    ds = load_dataset("KarthikaRajagopal/Restaurant_Reviews.tsv")
    df = ds["train"].to_pandas()
    df = df.drop_duplicates(subset="Review").reset_index(drop=True)
    df["review_final"] = df["Review"].apply(preprocess)
    return df


@st.cache_resource(show_spinner="Melatih model Naive Bayes...")
def train_model(df: pd.DataFrame):
    X = df["review_final"]
    y = df["Liked"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=["Negatif (0)", "Positif (1)"]
    )

    return vectorizer, model, acc, report


df = load_data()
vectorizer, model, acc, report = train_model(df)


# ----------------------------------------------------------------------------
# RAG (Retrieval-Augmented Generation) — retrieval + penyusunan jawaban
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model embedding...")
def load_embedding_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


@st.cache_data(show_spinner="Menghitung embedding seluruh dokumen review...")
def compute_doc_embeddings(reviews: list) -> np.ndarray:
    embed_model = load_embedding_model()
    return embed_model.encode(reviews, show_progress_bar=False)


def retrieve(query: str, knowledge_base: list, doc_embeddings: np.ndarray,
             liked_labels: list, top_k: int = 5):
    embed_model = load_embedding_model()
    query_embedding = embed_model.encode([query])
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "doc_id": int(idx),
            "text": knowledge_base[idx],
            "similarity": float(similarities[idx]),
            "liked": int(liked_labels[idx]),
        })
    return results


def compose_answer(query: str, retrieved_docs: list) -> str:
    """Menyusun jawaban (generation) berdasarkan dokumen hasil retrieval.
    Menggunakan ringkasan berbasis statistik + kutipan dokumen teratas,
    bukan model LLM generatif bebas (agar ringan & stabil untuk deployment)."""
    n = len(retrieved_docs)
    n_pos = sum(1 for d in retrieved_docs if d["liked"] == 1)
    n_neg = n - n_pos

    if n_pos > n_neg:
        kesimpulan = f"cenderung **positif** ({n_pos} dari {n} review relevan bernada positif)"
    elif n_neg > n_pos:
        kesimpulan = f"cenderung **negatif** ({n_neg} dari {n} review relevan bernada negatif)"
    else:
        kesimpulan = f"**campuran** ({n_pos} positif, {n_neg} negatif dari {n} review relevan)"

    top_doc = retrieved_docs[0]
    jawaban = (
        f"Berdasarkan {n} review pelanggan yang paling relevan dengan pertanyaan Anda, "
        f"pendapat pelanggan {kesimpulan}. "
        f"Review paling relevan (similarity {top_doc['similarity']:.2f}) menyatakan: "
        f"\"{top_doc['text']}\""
    )
    return jawaban


# ----------------------------------------------------------------------------
# SIDEBAR NAVIGASI
# ----------------------------------------------------------------------------
st.sidebar.title("🍽️ Menu")
page = st.sidebar.radio(
    "Pilih halaman",
    ["🔮 Prediksi Sentimen", "🔎 Tanya Jawab (RAG)", "📊 Eksplorasi Data (EDA)", "ℹ️ Tentang Proyek"],
)

st.sidebar.markdown("---")
st.sidebar.metric("Akurasi Model (test set)", f"{acc:.2%}")
st.sidebar.caption("Model: TF-IDF + Multinomial Naive Bayes")

# ----------------------------------------------------------------------------
# HALAMAN 1: PREDIKSI SENTIMEN
# ----------------------------------------------------------------------------
if page == "🔮 Prediksi Sentimen":
    st.title("🔮 Prediksi Sentimen Review Restoran")
    st.write(
        "Masukkan review pelanggan restoran (Bahasa Inggris), lalu sistem akan "
        "memprediksi apakah review tersebut **Positif** atau **Negatif**."
    )

    review_input = st.text_area(
        "Tulis review di sini:",
        placeholder="Example: The food was delicious and the staff was very friendly!",
        height=120,
    )

    if st.button("Prediksi Sentimen", type="primary"):
        if not review_input.strip():
            st.warning("Silakan masukkan teks review terlebih dahulu.")
        else:
            cleaned = preprocess(review_input)
            vect = vectorizer.transform([cleaned])
            pred = model.predict(vect)[0]
            proba = model.predict_proba(vect)[0]

            col1, col2 = st.columns(2)
            with col1:
                if pred == 1:
                    st.success(f"### 🟢 Sentimen: POSITIF")
                else:
                    st.error(f"### 🔴 Sentimen: NEGATIF")
                st.write(f"Keyakinan model: **{proba.max():.2%}**")

            with col2:
                st.write("**Probabilitas per kelas:**")
                st.bar_chart(
                    pd.DataFrame(
                        {"Probabilitas": proba},
                        index=["Negatif", "Positif"],
                    )
                )

            with st.expander("Lihat hasil preprocessing teks"):
                st.write("**Teks setelah dibersihkan & di-stem:**")
                st.code(cleaned)

    st.markdown("---")
    st.subheader("Coba beberapa contoh cepat")
    examples = [
        "The food was absolutely amazing and the service was great!",
        "Honestly it didn't taste that fresh.",
        "Worst experience ever, the staff was so rude.",
        "Great ambiance, will definitely come back again.",
    ]
    cols = st.columns(len(examples))
    for c, ex in zip(cols, examples):
        if c.button(ex, use_container_width=True):
            st.session_state["quick_example"] = ex
            st.rerun()

    if "quick_example" in st.session_state:
        st.info(f"Contoh dipilih: _{st.session_state['quick_example']}_")
        cleaned = preprocess(st.session_state["quick_example"])
        vect = vectorizer.transform([cleaned])
        pred = model.predict(vect)[0]
        label = "🟢 POSITIF" if pred == 1 else "🔴 NEGATIF"
        st.write(f"Prediksi: **{label}**")
        del st.session_state["quick_example"]

# ----------------------------------------------------------------------------
# HALAMAN 2: TANYA JAWAB (RAG)
# ----------------------------------------------------------------------------
elif page == "🔎 Tanya Jawab (RAG)":
    st.title("🔎 Tanya Jawab Berbasis Review (RAG)")
    st.write(
        "Ajukan pertanyaan (Bahasa Indonesia atau Inggris) tentang review pelanggan restoran. "
        "Sistem akan **mencari review paling relevan** (retrieval), **menyusun jawaban** "
        "berdasarkan review tersebut (generation), dan **menampilkan dokumen referensinya**."
    )

    knowledge_base = df["Review"].tolist()
    liked_labels = df["Liked"].tolist()
    doc_embeddings = compute_doc_embeddings(knowledge_base)

    query = st.text_input(
        "Pertanyaan Anda:",
        placeholder="Contoh: Bagaimana pendapat pelanggan tentang pelayanan di restoran?",
    )
    top_k = st.slider("Jumlah dokumen yang diambil (top-k)", min_value=3, max_value=10, value=5)

    if st.button("Cari Jawaban", type="primary"):
        if not query.strip():
            st.warning("Silakan masukkan pertanyaan terlebih dahulu.")
        else:
            retrieved_docs = retrieve(query, knowledge_base, doc_embeddings, liked_labels, top_k)
            jawaban = compose_answer(query, retrieved_docs)

            st.subheader("💬 Jawaban")
            st.info(jawaban)

            st.subheader("📄 Referensi Dokumen yang Digunakan")
            for i, ref in enumerate(retrieved_docs, 1):
                sentimen = "🟢 Positif" if ref["liked"] == 1 else "🔴 Negatif"
                st.write(f"**{i}.** [{ref['similarity']:.3f}] {sentimen} — {ref['text']}")

    st.markdown("---")
    st.subheader("Coba beberapa contoh pertanyaan")
    example_questions = [
        "Bagaimana kualitas makanan di restoran ini?",
        "Is the food here expensive?",
        "What are the most common complaints from customers?",
        "Bagaimana suasana atau ambiance di tempat ini?",
        "Apakah restoran ini direkomendasikan untuk dikunjungi?",
    ]
    for eq in example_questions:
        st.code(eq, language=None)

# ----------------------------------------------------------------------------
# HALAMAN 3: EKSPLORASI DATA
# ----------------------------------------------------------------------------
elif page == "📊 Eksplorasi Data (EDA)":
    st.title("📊 Eksplorasi Data Review Restoran")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Jumlah data (setelah hapus duplikat)", len(df))
    with col2:
        st.metric("Jumlah review positif", int((df["Liked"] == 1).sum()))

    st.subheader("Distribusi Label")
    st.bar_chart(df["Liked"].value_counts().rename({0: "Negatif", 1: "Positif"}))

    st.subheader("Kata Paling Sering Muncul")
    tab_all, tab_pos, tab_neg = st.tabs(["Semua Review", "Review Positif", "Review Negatif"])

    def top_words_chart(text_series, n=15):
        words = " ".join(text_series).split()
        freq = Counter(words).most_common(n)
        words_, counts_ = zip(*freq)
        return pd.DataFrame({"Frekuensi": counts_}, index=words_)

    with tab_all:
        st.bar_chart(top_words_chart(df["review_final"]))
    with tab_pos:
        st.bar_chart(top_words_chart(df[df["Liked"] == 1]["review_final"]))
    with tab_neg:
        st.bar_chart(top_words_chart(df[df["Liked"] == 0]["review_final"]))

    st.subheader("Word Cloud")
    wc_col1, wc_col2 = st.columns(2)
    with wc_col1:
        st.caption("Review Positif")
        pos_text = " ".join(df[df["Liked"] == 1]["review_final"])
        wc_pos = WordCloud(width=600, height=350, background_color="white",
                            colormap="Greens", stopwords=set()).generate(pos_text)
        fig, ax = plt.subplots()
        ax.imshow(wc_pos, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
    with wc_col2:
        st.caption("Review Negatif")
        neg_text = " ".join(df[df["Liked"] == 0]["review_final"])
        wc_neg = WordCloud(width=600, height=350, background_color="white",
                            colormap="Reds", stopwords=set()).generate(neg_text)
        fig2, ax2 = plt.subplots()
        ax2.imshow(wc_neg, interpolation="bilinear")
        ax2.axis("off")
        st.pyplot(fig2)

    with st.expander("Lihat Classification Report model"):
        st.text(report)

    with st.expander("Lihat contoh data mentah"):
        st.dataframe(df[["Review", "Liked", "review_final"]].head(20))

# ----------------------------------------------------------------------------
# HALAMAN 4: TENTANG PROYEK
# ----------------------------------------------------------------------------
else:
    st.title("ℹ️ Tentang Proyek")
    st.markdown(
        """
        Aplikasi ini adalah **prototipe** hasil pengerjaan UAS Text Mining/TTOS,
        yang mendemonstrasikan alur *end-to-end*:

        1. **Preprocessing teks**: case folding, pembersihan karakter non-huruf,
           tokenisasi, penghapusan stopword (dengan pengecualian kata negasi),
           dan stemming (Porter Stemmer).
        2. **Ekstraksi fitur**: TF-IDF (Term Frequency – Inverse Document Frequency).
        3. **Klasifikasi sentimen**: Multinomial Naive Bayes.
        4. **Retrieval-Augmented Generation (RAG)**: pencarian review paling relevan
           dengan `sentence-transformers` (retrieval), penyusunan jawaban dari
           review yang ditemukan (generation), dan tampilan referensi dokumen.
        5. **Eksplorasi data**: distribusi label, kata paling sering muncul, word cloud.

        **Dataset**: `KarthikaRajagopal/Restaurant_Reviews.tsv` dari HuggingFace Datasets.

        Dibangun dengan **Streamlit** sebagai antarmuka interaktif pengganti Gradio
        pada notebook asli, agar dapat di-deploy sebagai link prototipe publik.
        """
    )
