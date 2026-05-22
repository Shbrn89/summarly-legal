from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import TfidfVectorizer
from rouge_score import rouge_scorer
import PyPDF2
import torch
import re
import math

app = Flask(__name__)

tokenizer = None
model = None

MODEL_NAME = "facebook/bart-large-cnn"


def load_bart_model():
    global tokenizer, model

    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        model.eval()

    return tokenizer, model


def clean_text(text):
    if text is None:
        return ""

    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)

    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([.,!?;:])([A-Za-z])", r"\1 \2", text)

    replacements = {
        "underyour": "under your",
        "underYour": "under your",
        "terminateYour": "terminate your",
        "restrict,or": "restrict, or",
        "accountif": "account if",
        "servicesif": "services if",
        "areresponsible": "are responsible",
        "areResponsible": "are responsible",
        "youare": "you are",
        "Youare": "You are",
        "Youraccount": "Your account",
        "youraccount": "your account",
        "theplatform": "the platform",
        "Theplatform": "The platform"
    }

    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)

    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def post_process_summary(summary):
    if summary is None:
        return ""

    summary = re.sub(r"([a-z])([A-Z])", r"\1 \2", summary)
    summary = re.sub(r"\s+([.,!?;:])", r"\1", summary)
    summary = re.sub(r"([.,!?;:])([A-Za-z])", r"\1 \2", summary)

    replacements = {
        "underyour": "under your",
        "underYour": "under your",
        "terminateYour": "terminate your",
        "restrict,or": "restrict, or",
        "accountif": "account if",
        "servicesif": "services if",
        "areresponsible": "are responsible",
        "areResponsible": "are responsible",
        "youare": "you are",
        "Youare": "You are",
        "Youraccount": "Your account",
        "youraccount": "your account",
        "theplatform": "the platform",
        "Theplatform": "The platform"
    }

    for wrong, correct in replacements.items():
        summary = summary.replace(wrong, correct)

    summary = re.sub(r"\s+", " ", summary)
    summary = summary.strip()

    if summary and summary[-1] not in ".!?":
        last_period = max(
            summary.rfind("."),
            summary.rfind("!"),
            summary.rfind("?")
        )

        if last_period != -1:
            summary = summary[:last_period + 1]

    return summary


def split_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences


def extract_pdf_text(file):
    reader = PyPDF2.PdfReader(file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + " "

    return text


def detect_language_simple(text):
    lower_text = text.lower()

    indonesian_words = [
        "yang", "dan", "dengan", "adalah", "untuk", "dalam", "pada",
        "karena", "sebagai", "tersebut", "pengguna", "dokumen",
        "penelitian", "ringkasan", "maka", "atau", "dapat", "tidak",
        "akan", "ini", "itu", "oleh", "juga", "bahwa", "menjadi"
    ]

    english_words = [
        "the", "and", "with", "is", "are", "for", "to", "in", "on",
        "by", "this", "that", "you", "your", "we", "our", "may",
        "service", "data", "account", "information", "privacy",
        "terms", "conditions"
    ]

    indonesian_score = 0
    english_score = 0

    words = re.findall(r"\b[a-zA-Z]+\b", lower_text)

    for word in words:
        if word in indonesian_words:
            indonesian_score += 1

        if word in english_words:
            english_score += 1

    if indonesian_score > english_score:
        return "indonesian"

    if english_score > indonesian_score:
        return "english"

    return "mixed"


def chunk_text(text, max_words=700):
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)

    return chunks


def get_summary_length(word_count):
    if word_count < 80:
        return 35, 10

    if word_count < 150:
        return 50, 15

    if word_count < 250:
        return 75, 20

    if word_count < 400:
        return 100, 30

    if word_count < 700:
        return 140, 45

    return 180, 60


def normalize_for_compare(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def calculate_copy_ratio(original_text, summary_text):
    original_sentences = split_sentences(original_text)
    summary_sentences = split_sentences(summary_text)

    if len(summary_sentences) == 0:
        return 0

    copied_count = 0

    normalized_original_sentences = [
        normalize_for_compare(sentence)
        for sentence in original_sentences
    ]

    for summary_sentence in summary_sentences:
        normalized_summary = normalize_for_compare(summary_sentence)

        for original_sentence in normalized_original_sentences:
            if len(normalized_summary.split()) >= 8 and normalized_summary in original_sentence:
                copied_count += 1
                break

            if len(original_sentence.split()) >= 8 and original_sentence in normalized_summary:
                copied_count += 1
                break

    return copied_count / len(summary_sentences)


def generate_bart_summary(tokenizer, model, chunk, aggressive=False):
    word_count = len(chunk.split())
    max_summary_length, min_summary_length = get_summary_length(word_count)

    if aggressive:
        max_summary_length = max(35, int(max_summary_length * 0.70))
        min_summary_length = max(12, int(min_summary_length * 0.60))
        num_beams = 6
        length_penalty = 3.0
        no_repeat_ngram_size = 3
        encoder_no_repeat_ngram_size = 4
        repetition_penalty = 1.35
    else:
        num_beams = 4
        length_penalty = 2.5
        no_repeat_ngram_size = 3
        encoder_no_repeat_ngram_size = 4
        repetition_penalty = 1.2

    inputs = tokenizer(
        chunk,
        max_length=1024,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        summary_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_summary_length,
            min_length=min_summary_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            early_stopping=True
        )

    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    summary = post_process_summary(summary)

    return summary


def bart_summarize(text):
    language = detect_language_simple(text)

    warning = ""

    if language == "indonesian":
        warning = (
            "Warning: Teks terdeteksi dominan bahasa Indonesia. "
            "Model BART-Large-CNN lebih cocok untuk teks bahasa Inggris. "
            "Untuk teks Indonesia, hasil yang lebih stabil biasanya menggunakan TF-IDF."
        )

    word_count = len(text.split())

    if word_count < 35:
        return {
            "summary": (
                "Teks terlalu pendek untuk BART. "
                "Gunakan teks Terms and Conditions yang lebih panjang agar BART dapat membuat ringkasan yang lebih baik."
            ),
            "warning": warning
        }

    tokenizer, model = load_bart_model()

    chunks = chunk_text(text, max_words=700)
    summaries = []

    for chunk in chunks:
        if len(chunk.split()) < 35:
            continue

        summary = generate_bart_summary(
            tokenizer=tokenizer,
            model=model,
            chunk=chunk,
            aggressive=False
        )

        copy_ratio = calculate_copy_ratio(chunk, summary)

        if copy_ratio >= 0.60:
            summary = generate_bart_summary(
                tokenizer=tokenizer,
                model=model,
                chunk=chunk,
                aggressive=True
            )

        summary = post_process_summary(summary)
        summaries.append(summary)

    final_summary = " ".join(summaries)
    final_summary = post_process_summary(final_summary)

    if not final_summary:
        return {
            "summary": (
                "BART gagal membuat ringkasan. "
                "Coba gunakan teks Terms and Conditions bahasa Inggris yang lebih panjang."
            ),
            "warning": warning
        }

    if len(final_summary.split()) > 160:
        final_summary = generate_bart_summary(
            tokenizer=tokenizer,
            model=model,
            chunk=final_summary,
            aggressive=True
        )

    final_summary = post_process_summary(final_summary)

    return {
        "summary": final_summary,
        "warning": warning
    }


def tfidf_summarize(text):
    sentences = split_sentences(text)

    if len(sentences) == 0:
        return {
            "summary": "Tidak ada kalimat yang bisa diringkas.",
            "warning": ""
        }

    if len(sentences) <= 3:
        return {
            "summary": " ".join(sentences),
            "warning": "Teks terlalu pendek, jadi TF-IDF mengembalikan hampir seluruh kalimat."
        }

    indonesian_stopwords = [
        "yang", "dan", "di", "ke", "dari", "untuk", "dengan", "pada",
        "adalah", "ini", "itu", "atau", "sebagai", "dalam", "akan",
        "dapat", "karena", "oleh", "juga", "tidak", "lebih", "agar",
        "serta", "yaitu", "tersebut", "maka", "sehingga", "hal",
        "para", "bagi", "suatu", "telah", "menjadi", "secara",
        "pengguna", "dokumen"
    ]

    english_stopwords = [
        "the", "and", "is", "in", "to", "of", "for", "with", "on",
        "this", "that", "as", "by", "are", "be", "or", "an", "a",
        "from", "it", "we", "you", "your", "our", "may", "can",
        "will", "shall", "have", "has", "had", "not", "if", "any"
    ]

    stopwords = indonesian_stopwords + english_stopwords

    total_sentences = len(sentences)
    top_n = max(3, math.ceil(total_sentences * 0.25))
    top_n = min(top_n, 7)

    vectorizer = TfidfVectorizer(
        stop_words=stopwords,
        lowercase=True,
        ngram_range=(1, 2)
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        return {
            "summary": " ".join(sentences[:top_n]),
            "warning": "TF-IDF tidak menemukan fitur kata yang cukup, sehingga mengambil kalimat awal sebagai fallback."
        }

    scores = tfidf_matrix.sum(axis=1).A1

    ranked_sentences = sorted(
        [
            (score, index, sentence)
            for index, (score, sentence) in enumerate(zip(scores, sentences))
        ],
        reverse=True
    )

    selected_sentences = sorted(
        ranked_sentences[:top_n],
        key=lambda x: x[1]
    )

    summary = " ".join(
        [sentence for score, index, sentence in selected_sentences]
    )

    summary = post_process_summary(summary)

    return {
        "summary": summary,
        "warning": (
            "TF-IDF adalah extractive summarization. "
            "Hasilnya mengambil kalimat asli yang dianggap paling penting, bukan menulis ulang kalimat."
        )
    }


def detect_legal_risks(text):
    lower_text = text.lower()

    risk_keywords = {
        "Data Privacy": [
            "personal data",
            "privacy",
            "third party",
            "collect information",
            "user data",
            "share your data",
            "data pribadi",
            "pihak ketiga",
            "mengumpulkan data",
            "informasi pribadi"
        ],
        "Hidden Cost": [
            "fee",
            "payment",
            "subscription",
            "billing",
            "automatic renewal",
            "charges",
            "biaya",
            "pembayaran",
            "langganan",
            "perpanjangan otomatis"
        ],
        "Account Termination": [
            "terminate",
            "suspend",
            "ban",
            "disable account",
            "remove account",
            "menghapus akun",
            "menangguhkan",
            "pemutusan akun",
            "menonaktifkan akun"
        ],
        "User Rights": [
            "license",
            "ownership",
            "intellectual property",
            "content rights",
            "rights",
            "lisensi",
            "kepemilikan",
            "hak pengguna",
            "hak cipta"
        ],
        "Legal Responsibility": [
            "liability",
            "damages",
            "indemnify",
            "responsibility",
            "legal claim",
            "tanggung jawab",
            "kerugian",
            "klaim hukum"
        ]
    }

    detected = []

    for category, keywords in risk_keywords.items():
        found = []

        for keyword in keywords:
            if keyword in lower_text:
                found.append(keyword)

        if found:
            detected.append({
                "category": category,
                "keywords": found
            })

    return detected


def calculate_rouge(reference, generated):
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )

    scores = scorer.score(reference, generated)

    result = {
        "rouge1": {
            "precision": round(scores["rouge1"].precision, 4),
            "recall": round(scores["rouge1"].recall, 4),
            "f1": round(scores["rouge1"].fmeasure, 4)
        },
        "rouge2": {
            "precision": round(scores["rouge2"].precision, 4),
            "recall": round(scores["rouge2"].recall, 4),
            "f1": round(scores["rouge2"].fmeasure, 4)
        },
        "rougeL": {
            "precision": round(scores["rougeL"].precision, 4),
            "recall": round(scores["rougeL"].recall, 4),
            "f1": round(scores["rougeL"].fmeasure, 4)
        }
    }

    return result


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/summarize", methods=["POST"])
def summarize():
    try:
        input_type = request.form.get("inputType")
        method = request.form.get("method")
        text = ""

        if input_type == "text":
            text = request.form.get("text", "")

        elif input_type == "file":
            uploaded_file = request.files.get("file")

            if uploaded_file is None:
                return jsonify({
                    "success": False,
                    "message": "File belum diupload."
                })

            filename = uploaded_file.filename.lower()

            if filename.endswith(".pdf"):
                text = extract_pdf_text(uploaded_file)

            elif filename.endswith(".txt"):
                text = uploaded_file.read().decode("utf-8")

            else:
                return jsonify({
                    "success": False,
                    "message": "Format file tidak didukung. Gunakan file PDF atau TXT."
                })

        text = clean_text(text)

        if not text:
            return jsonify({
                "success": False,
                "message": "Teks kosong. Masukkan teks atau upload dokumen terlebih dahulu."
            })

        language = detect_language_simple(text)

        if method == "bart":
            result = bart_summarize(text)

        elif method == "tfidf":
            result = tfidf_summarize(text)

        else:
            return jsonify({
                "success": False,
                "message": "Metode summarization tidak valid."
            })

        summary = result["summary"]
        warning = result["warning"]

        risks = detect_legal_risks(text)

        return jsonify({
            "success": True,
            "method": method,
            "language": language,
            "modelName": MODEL_NAME if method == "bart" else "TF-IDF",
            "originalWordCount": len(text.split()),
            "summaryWordCount": len(summary.split()),
            "summary": summary,
            "warning": warning,
            "risks": risks
        })

    except Exception as error:
        return jsonify({
            "success": False,
            "message": str(error)
        })


@app.route("/rouge", methods=["POST"])
def rouge():
    try:
        data = request.get_json()

        reference = data.get("reference", "")
        generated = data.get("generated", "")

        if not reference or not generated:
            return jsonify({
                "success": False,
                "message": "Reference summary dan generated summary wajib diisi."
            })

        scores = calculate_rouge(reference, generated)

        return jsonify({
            "success": True,
            "scores": scores
        })

    except Exception as error:
        return jsonify({
            "success": False,
            "message": str(error)
        })


if __name__ == "__main__":
    app.run(debug=False)