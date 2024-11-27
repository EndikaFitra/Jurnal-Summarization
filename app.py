import os
import tempfile
import nltk
from langchain.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
import streamlit as st
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from langdetect import detect
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download NLTK resources for tokenization and stop words
nltk.download('punkt', quiet=True)
nltk.download("stopwords", quiet=True)

# Helper Functions
def count_sentences(text):
    """Count total number of sentences in a text."""
    sentences = nltk.sent_tokenize(text)
    return len(sentences)

def count_words(text):
    """Count total number of words in a text."""
    words = text.split()
    return len(words)

def count_characters(text):
    """Count total number of characters in a text."""
    return len(text)

def detect_language(text):
    """Detect language of the text."""
    return detect(text)

def extract_keywords(text):
    """Extract keywords based on frequency analysis."""
    stop_words = set(stopwords.words('indonesian'))
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    fdist = FreqDist(filtered_words)
    return fdist.most_common(10)  # Top 10 frequent words

def preprocess_text(text):
    """Remove headers, footers, and unnecessary spaces from the text."""
    lines = text.split('\n')
    cleaned_lines = [line for line in lines if not line.strip().startswith("Page") and len(line.strip()) > 0]
    return ' '.join(cleaned_lines)

def generate_wordcloud(text):
    """Generate word cloud from text."""
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

def generate_response(doc_path):
    """Generate response from the entire PDF document."""
    # Load PDF using PyPDFLoader
    loader = PyPDFLoader(doc_path)
    docs = loader.load()

    # Combine all pages into one text
    original_text = ' '.join([doc.page_content for doc in docs])
    original_text = preprocess_text(original_text)

    # Count total sentences in original document
    original_sentence_count = count_sentences(original_text)

    # Detect language
    language = detect_language(original_text)

    # Initialize the LLM (Groq) model
    llm = ChatGroq(model="llama3-8b-8192", api_key="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")

    # Define prompt template in Indonesian
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Buatlah ringkasan komprehensif dalam bahasa Indonesia dengan kriteria berikut:
        1. Gunakan bahasa Indonesia yang baik dan benar
        2. Fokus pada poin-poin kunci dan ide utama dokumen
        3. Sajikan informasi secara sistematis dan mudah dipahami
        4. Sertakan informasi penting seperti:
           - Judul dokumen/jurnal
           - Nama penulis/peneliti
           - Metodologi utama (jika ada)
           - Temuan atau kesimpulan kunci
        5. Gunakan struktur paragraf yang jelas dan informatif

        Dokumen untuk diringkas:\\n\\n{context}"""),
        ("human", "Hasilkan ringkasan lengkap dalam bahasa Indonesia")
    ])

    # Create chain for document processing
    chain = create_stuff_documents_chain(llm, prompt)
    result = chain.invoke({"context": docs})

    # Count sentences in summary
    summary_sentence_count = count_sentences(result)

    return {
        'summary': result,
        'original_text': original_text,
        'original_sentence_count': original_sentence_count,
        'summary_sentence_count': summary_sentence_count,
        'language': language
    }

# Streamlit UI
st.set_page_config(page_title="Aplikasi Peringkas Jurnal", page_icon="📄")

# Sidebar menu with dropdown
st.sidebar.title("Menu")
menu = st.sidebar.selectbox("Navigasi", ["Beranda", "Tentang Aplikasi", "Cara Penggunaan"])

# Main content based on selected menu
if menu == "Beranda":
    st.title("📄 Aplikasi Peringkas Jurnal")
    st.markdown("""
    Selamat datang di aplikasi **Peringkas Jurnal PDF**! Unggah dokumen Anda dan dapatkan ringkasan lengkap yang mencakup poin-poin utama, keyword penting, dan visualisasi data.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Pilih file PDF", type=["pdf"])
    
    # Button to submit and process
    if st.button("Proses"):
        if uploaded_file is not None:
            try:
                # Save uploaded PDF to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_pdf_path = temp_file.name
                
                # Generate response using the uploaded file
                result = generate_response(temp_pdf_path)
                
                # Display the result
                st.subheader("Hasil Ringkasan")
                st.write(result['summary'])
                
                # Display document statistics
                st.subheader("Statistik Dokumen")
                st.write(f"Jumlah Kalimat Dokumen Asli: {result['original_sentence_count']}")
                st.write(f"Jumlah Kalimat Ringkasan: {result['summary_sentence_count']}")
                st.write(f"Jumlah Kata Dokumen Asli: {count_words(result['original_text'])}")
                st.write(f"Jumlah Karakter Dokumen Asli: {count_characters(result['original_text'])}")
                st.write(f"Bahasa Deteksi: {result['language']}")

                # Extract and display keywords
                st.subheader("Keyword Utama")
                keywords = extract_keywords(result['original_text'])
                st.write([kw[0] for kw in keywords])

                # Generate and display word cloud
                st.subheader("Word Cloud")
                generate_wordcloud(result['original_text'])
                
                # Download summary
                st.subheader("Unduh Ringkasan")
                st.download_button(
                    label="Unduh Ringkasan sebagai TXT",
                    data=result['summary'],
                    file_name="ringkasan.txt",
                    mime="text/plain"
                )
                
            except Exception as e:
                st.error(f"Kesalahan: {e}")
        else:
            st.error("Silakan unggah file PDF.")

elif menu == "Tentang Aplikasi":
    st.title("📘 Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini dirancang untuk memudahkan pengguna dalam meringkas dokumen PDF, khususnya jurnal akademik. Dengan menggunakan teknologi kecerdasan buatan, aplikasi ini mampu:
    
    - Menghasilkan ringkasan dari dokumen asli dengan fokus pada poin-poin utama.
    - Menyediakan keyword utama dan visualisasi data melalui Word Cloud.
    - Memberikan statistik terkait jumlah kalimat, kata, dan karakter pada dokumen asli.
    
    Gunakan aplikasi ini untuk mempercepat proses pemahaman jurnal atau dokumen akademik lainnya!
    """)

elif menu == "Cara Penggunaan":
    st.title("📗 Cara Penggunaan")
    st.markdown("""
    Ikuti langkah-langkah berikut untuk menggunakan aplikasi:
    
    1. **Unggah File PDF**:
       - Pilih file PDF dari komputer Anda menggunakan fitur unggah di halaman utama.
    2. **Proses Dokumen**:
       - Klik tombol "Proses" untuk memulai proses pembuatan ringkasan.
    3. **Lihat Hasil**:
       - Aplikasi akan menampilkan ringkasan dokumen, keyword utama, dan visualisasi Word Cloud.
    4. **Unduh Ringkasan**:
       - Anda dapat mengunduh ringkasan yang dihasilkan dalam format TXT.
    """)
