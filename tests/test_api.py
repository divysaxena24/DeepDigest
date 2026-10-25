import pytest
from fastapi.testclient import TestClient
from app.main import app, DOC_STORE

client = TestClient(app)

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

def test_summarize_without_doc():
    DOC_STORE["text"] = ""
    res = client.post("/summarize")
    assert res.status_code == 400
    assert "No active document" in res.json()["detail"]

def test_ask_flow():
    # Inject text into DOC_STORE directly for isolated testing
    DOC_STORE["text"] = (
        "The Attention mechanism enables neural networks to focus dynamically on key features. "
        "Transformers rely heavily on multi-head self-attention mechanisms to capture complex document patterns."
    )
    DOC_STORE["filename"] = "test_attention_doc.pdf"

    # Test Q&A endpoint
    res = client.post("/ask", data={"question": "What enables neural networks to focus dynamically?"})
    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert "confidence" in data
    assert len(data["answer"]) > 0

def test_summarize_flow():
    DOC_STORE["text"] = (
        "Deep learning architectures process sequential data through recurrence and attention mechanisms. "
        "By calculating salience weights across individual document segments, systems can isolate critical "
        "thematic contexts for downstream abstractive generation models."
    )
    DOC_STORE["filename"] = "test_salience.pdf"

    res = client.post("/summarize")
    assert res.status_code == 200
    data = res.json()
    assert "summary" in data
    assert "salient_chunk_index" in data
    assert "attention_distribution" in data
    assert len(data["attention_distribution"]) > 0