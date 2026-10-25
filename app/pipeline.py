import numpy as np
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from app.dl_models import AttentionSalienceScorerTF
from app.vector_store import FAISSVectorStore
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

        # 3. Persistent FAISS Vector Store (384-dim matching all-MiniLM-L6-v2)
        self.vector_store = FAISSVectorStore(dimension=384)

        # 4. Abstractive summarization with DistilBART (explicit PyTorch)
        self.summarizer = pipeline(
            'summarization',
            model='sshleifer/distilbart-cnn-12-6',
            framework='pt'
        )

        # 5. Contextual extractive QA with RoBERTa (explicit PyTorch)
        self.qa_model = pipeline(
            'question-answering',
            model='deepset/roberta-base-squad2',
            framework='pt'
        )

        self.current_chunks = self.vector_store.chunks

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
        
        # Reset and index in persistent FAISS
        self.vector_store.reset()
        if chunks:
            embeddings = self.embedder.encode(
                chunks, 
                batch_size=32, 
                show_progress_bar=False, 
                convert_to_numpy=True
            )
            self.vector_store.add_chunks(chunks, embeddings)
            self.vector_store.save()

        return {"chunks_count": len(chunks), "total_words": len(words)}

    def compute_salience_and_summarize(self) -> dict:
        if not self.current_chunks:
            raise ValueError("No document ingested yet.")

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

        # Dynamic max_length prevents warnings on short test chunks
        token_count = max(len(top_chunk_text.split()), 10)
        max_len = min(130, max(25, int(token_count * 1.2)))
        min_len = min(10, max_len - 5)

        summary_res = self.summarizer(top_chunk_text, max_length=max_len, min_length=min_len, do_sample=False)
        summary = summary_res[0]['summary_text']

        return {
            "salient_chunk_index": top_chunk_idx,
            "top_chunk_idx": top_chunk_idx,
            "attention_distribution": [float(w) for w in norm_weights],
            "summary": summary
        }

    def process_and_summarize(self, text: str = None) -> dict:
        if text and (not self.current_chunks or text != " ".join(self.current_chunks)):
            self.ingest_text(text)
        return self.compute_salience_and_summarize()

    def answer_question(self, question: str) -> dict:
        if not self.current_chunks or self.vector_store.total_chunks == 0:
            raise ValueError("No document ingested yet.")

        q_emb = self.embedder.encode(question, convert_to_numpy=True)
        search_results = self.vector_store.search(q_emb, top_k=1)

        if not search_results:
            raise ValueError("Context retrieval failed.")

        context, sim_score = search_results[0]
        qa_res = self.qa_model(question=question, context=context)

        return {
            "answer": qa_res['answer'],
            "confidence": float(qa_res['score']),
            "retrieval_similarity": float(sim_score),
            "context": context,
            "retrieved_context": context
        }

    def answer_query(self, question: str, text: str = None) -> dict:
        if text and (not self.current_chunks or self.vector_store.total_chunks == 0):
            self.ingest_text(text)
        return self.answer_question(question)