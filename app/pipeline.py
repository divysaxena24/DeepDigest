import numpy as np
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from app.dl_models import AttentionSalienceScorerTF
import pypdf
import io
import re

class TFDeepLearningDocPipeline:
    def __init__(self):

# 1. Custom TensorFlow Bi-GRU + Bahdanau Attention model
        self.salience_model = AttentionSalienceScorerTF(vocab_size=5000, embed_dim=64, gru_units=64)
        dummy = tf.zeros((1, 20), dtype=tf.int32)
        self.salience_model(dummy)  # Build weights
        
        # 2. Dense semantic embeddings
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # 3. Abstractive summarization with DistilBART (explicit PyTorch)
        self.summarizer = pipeline(
            'summarization',
            model='sshleifer/distilbart-cnn-12-6',
            framework='pt'
        )

        # 4. Contextual extractive QA with RoBERTa (explicit PyTorch)
        self.qa_model = pipeline(
            'question-answering',
            model='deepset/roberta-base-squad2',
            framework='pt'
        )

        self.current_chunks = []
        self.chunk_embeddings = None

    def ingest_pdf(self, file_bytes: bytes) -> dict:
        pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        full_text = ""
        for page in pdf_reader.pages:
            t = page.extract_text()
            if t:
                full_text += t + " "

        return self.ingest_text(full_text)

    def ingest_text(self, text: str) -> dict:
        clean = re.sub(r'\s+', ' ', text).strip()
        words = clean.split(' ')
        chunk_size = 150
        overlap = 30
        step = chunk_size - overlap

        chunks = []
        for i in range(0, len(words), step):
            c = " ".join(words[i:i + chunk_size])
            if c:
                chunks.append(c)

        if not chunks and clean:
            chunks = [clean]

        self.current_chunks = chunks
        if chunks:
            self.chunk_embeddings = self.embedder.encode(chunks, convert_to_numpy=True)
        else:
            self.chunk_embeddings = None

        return {"chunks_count": len(chunks), "total_words": len(words)}

    def compute_salience_and_summarize(self) -> dict:
        if not self.current_chunks:
            raise ValueError("No document ingested yet.")

        # Simulate token index representations for each chunk
        chunk_tokens = []
        for c in self.current_chunks:
            words = c.lower().split()[:20]
            tokens = [hash(w) % 4999 + 1 for w in words]
            if len(tokens) < 20:
                tokens = tokens + [0] * (20 - len(tokens))
            chunk_tokens.append(tokens)

        x = tf.constant(chunk_tokens, dtype=tf.int32)
        _, weights = self.salience_model(x)

        avg_weights = np.mean(weights.numpy(), axis=1).flatten()
        if np.sum(avg_weights) > 0:
            norm_weights = avg_weights / np.sum(avg_weights)
        else:
            norm_weights = np.ones(len(self.current_chunks)) / len(self.current_chunks)

        top_chunk_idx = int(np.argmax(norm_weights))
        top_chunk_text = self.current_chunks[top_chunk_idx]

        summary_res = self.summarizer(top_chunk_text, max_length=130, min_length=30, do_sample=False)
        summary = summary_res[0]['summary_text']

        return {
            "salient_chunk_index": top_chunk_idx,
            "attention_distribution": [float(w) for w in norm_weights],
            "summary": summary
        }

    def answer_question(self, question: str) -> dict:
        if not self.current_chunks or self.chunk_embeddings is None:
            raise ValueError("No document ingested yet.")

        q_emb = self.embedder.encode([question], convert_to_numpy=True)

        norms = np.linalg.norm(self.chunk_embeddings, axis=1) * np.linalg.norm(q_emb)
        norms[norms == 0] = 1e-10
        similarities = np.dot(self.chunk_embeddings, q_emb.T).flatten() / norms

        top_idx = int(np.argmax(similarities))
        context = self.current_chunks[top_idx]

        qa_res = self.qa_model(question=question, context=context)

        return {
            "answer": qa_res['answer'],
            "confidence": float(qa_res['score']),
            "context": context
        }