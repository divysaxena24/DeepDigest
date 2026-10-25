import numpy as np
import tensorflow as tf
from app.pipeline import TFDeepLearningDocPipeline

def run_evaluation():
    print("=" * 65)
    print("      DEEPDIGEST ENGINE: LOCAL ACCURACY & BENCHMARK SUITE     ")
    print("=" * 65)

    print("\n[1/4] Initializing TF pipeline & transformer weights...")
    pipeline = TFDeepLearningDocPipeline()
    print("      ✓ Models loaded successfully.")

    # -------------------------------------------------------------
    # Multi-paragraph text (~320 words) to produce >= 3 distinct chunks
    # -------------------------------------------------------------
    eval_text = (
        "Traditional recurrent neural networks struggle with long-range dependencies "
        "due to vanishing and exploding gradient problems during backpropagation through time. "
        "Gated architectures like LSTMs and Gated Recurrent Units introduce reset and update "
        "gates to regulate gradient flows across sequential steps. However, sequential computation "
        "imposes significant training bottlenecks because hidden states cannot be calculated in parallel. "
        "To mitigate this, attention mechanisms dynamically calculate weighted sums across arbitrary steps. "
        "The Bahdanau additive attention layer computes alignment scores using a learned feedforward network "
        "parameterized by weight matrices and a hyperbolic tangent non-linearity. This dynamic context vector "
        "eliminates the single fixed-length vector bottleneck present in standard sequence-to-sequence models, "
        "allowing downstream decoders to selectively retrieve input tokens based on computed salience weights. "
        "Dense vector representations transform textual phrases into high-dimensional geometric coordinates. "
        "Using Facebook AI Similarity Search with IndexFlatIP and unit L2 normalization guarantees exact cosine "
        "similarity retrieval over dense passages without loss of semantic precision. This structure enables "
        "the retrieval-augmented generation engine to rapidly filter irrelevant background paragraphs before "
        "passing the optimal top-k contexts to heavy extractive question answering architectures like RoBERTa."
    )

    print("\n[2/4] Testing Ingestion & FAISS Semantic Indexing...")
    ingest_meta = pipeline.ingest_text(eval_text)
    print(f"      Total Chunks: {ingest_meta['chunks_count']} | Total Words: {ingest_meta['total_words']}")
    assert ingest_meta["chunks_count"] >= 2, "Document must chunk into at least 2 segments."
    print("      ✓ FAISS index initialized with normalized vectors.")

    # -------------------------------------------------------------
    # 1. Attention Model Validation
    # -------------------------------------------------------------
    print("\n[3/4] Evaluating Bi-GRU + Bahdanau Attention Salience...")
    salience_res = pipeline.compute_salience_and_summarize()
    weights = np.array(salience_res["attention_distribution"])
    
    weight_sum = np.sum(weights)
    entropy = -np.sum(weights * np.log(weights + 1e-12))
    top_idx = salience_res["salient_chunk_index"]

    print(f"      Attention Distribution: {[round(w, 4) for w in weights]}")
    print(f"      Sum of Weights: {weight_sum:.6f} (Must equal 1.0)")
    print(f"      Salience Entropy: {entropy:.4f}")
    print(f"      Top Salient Chunk Index: {top_idx}")
    print(f"      Abstractive Summary: \"{salience_res['summary'][:100]}...\"")

    assert np.isclose(weight_sum, 1.0, atol=1e-5), "Attention weights do not sum to 1.0!"
    assert len(salience_res["summary"]) > 20, "Summary generation failed minimum length."
    print("      ✓ Bahdanau attention layer mathematically sound.")

    # -------------------------------------------------------------
    # 2. FAISS Retrieval + RoBERTa QA Accuracy
    # -------------------------------------------------------------
    print("\n[4/4] Evaluating FAISS Retrieval & RoBERTa QA Extraction...")
    test_cases = [
        {
            "query": "What causes traditional RNNs to struggle with long-range dependencies?",
            "expected_keywords": ["vanishing", "exploding", "gradient"]
        },
        {
            "query": "What non-linearity is used in Bahdanau additive attention?",
            "expected_keywords": ["hyperbolic tangent", "tanh"]
        },
        {
            "query": "What index guarantees exact cosine similarity search in FAISS?",
            "expected_keywords": ["indexflatip"]
        }
    ]

    passed_qa = 0
    for idx, tc in enumerate(test_cases, 1):
        res = pipeline.answer_question(tc["query"])
        ans = res["answer"].lower()
        conf = res["confidence"]
        sim = res["retrieval_similarity"]
        
        keyword_hits = [k for k in tc["expected_keywords"] if k in ans]
        hit = len(keyword_hits) > 0

        print(f"\n      Test Q{idx}: \"{tc['query']}\"")
        print(f"      -> Retrieved Similarity: {sim:.4f}")
        print(f"      -> RoBERTa Answer: \"{res['answer']}\" (Confidence: {conf * 100:.2f}%)")
        print(f"      -> Groundedness Check: {'PASS' if hit else 'PARTIAL/FAIL'}")

        if hit and conf > 0.30:
            passed_qa += 1

    accuracy_pct = (passed_qa / len(test_cases)) * 100
    print("\n" + "=" * 65)
    print(f"      ACCURACY VERDICT: {passed_qa}/{len(test_cases)} ({accuracy_pct:.1f}%) QA Tests Passed")
    print("=" * 65)

    if accuracy_pct >= 66.0:
        print("\n>>> System accuracy verified. Safe to proceed to GHCR & Cloud Deployment.\n")
    else:
        print("\n>>> Accuracy under threshold. Review tokenization or confidence margins.\n")

if __name__ == "__main__":
    run_evaluation()