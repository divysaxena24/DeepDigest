import re
import numpy as np
import tensorflow as tf
from pypdf import PdfReader
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from app.dl_models import AttentionSalienceScorerTF

class TFDeepLearningDocPipeline:
    def __init__(self):
        # 1. Semantic Embedder (Dense representation)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        # 2. Initialize Custom TensorFlow Attention Salience Model
        self.salience_model = AttentionSalienceScorerTF(embed_dim=384, hidden_dim=128)
        dummy = tf.zeros((1, 5, 384), dtype=tf.float32)
        self.salience_model(dummy)

        # 3. Transformer Abstractive Summarizer
        self.summarizer = pipeline('summarization', model='sshleifer/distilbart-cnn-12-6')

        # 4. Transformer Extractive QA Head
        self.qa_head = pipeline('question-answering', model='deepset/roberta-base-squad2')

    @staticmethod
    def extract_and_clean_pdf(file_bytes) -> str:
        reader = PdfReader(file_bytes)
        text = ' '.join([page.extract_text() or '' for page in reader.pages])
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        text = re.sub(r'[\r\n\t]+', ' ', text)
        return re.sub(r' +', ' ', text).strip()

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 150, overlap: int = 30):
        words = text.split()
        if not words:
            return []
        chunks = []
        step = chunk_size - overlap
        for i in range(0, len(words), step):
            sub = words[i:i + chunk_size]
            chunks.append({'chunk_id': len(chunks), 'text': ' '.join(sub)})
            if i + chunk_size >= len(words):
                break
        return chunks

    def process_and_summarize(self, text: str) -> dict:
        chunks = self.chunk_text(text, chunk_size=150, overlap=30)
        if not chunks:
            return {"summary": "Empty document.", "attention_weights": [], "top_chunk_idx": 0}

        chunk_texts = [c['text'] for c in chunks]
        raw_embs = self.embedder.encode(chunk_texts, convert_to_tensor=False)
        tf_embs = tf.expand_dims(tf.convert_to_tensor(raw_embs, dtype=tf.float32), axis=0)

        _, attn_weights = self.salience_model(tf_embs, training=False)
        attn_dist = attn_weights.numpy()[0].tolist()
        top_idx = int(np.argmax(attn_dist))

        summary_res = self.summarizer(
            chunk_texts[top_idx],
            max_length=120,
            min_length=30,
            do_sample=False
        )

        return {
            "summary": summary_res[0]["summary_text"],
            "top_chunk_idx": top_idx,
            "attention_distribution": [round(float(w), 4) for w in attn_dist]
        }

    def answer_query(self, question: str, text: str, top_k: int = 2) -> dict:
        chunks = self.chunk_text(text, chunk_size=150, overlap=30)
        if not chunks:
            return {"answer": "No content to search.", "confidence": 0.0, "retrieved_context": ""}

        chunk_texts = [c['text'] for c in chunks]
        doc_emb = self.embedder.encode(chunk_texts, normalize_embeddings=True)
        q_emb = self.embedder.encode([question], normalize_embeddings=True)

        sims = np.dot(doc_emb, q_emb.T).flatten()
        top_indices = np.argsort(sims)[::-1][:top_k]
        context = ' '.join([chunk_texts[idx] for idx in top_indices])

        qa_res = self.qa_head(question=question, context=context)
        return {
            "answer": qa_res["answer"],
            "confidence": round(float(qa_res["score"]), 4),
            "retrieved_context": context[:250] + '...'
        }