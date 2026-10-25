import io
import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from app.pipeline import TFDeepLearningDocPipeline

app = FastAPI(
    title="DeepDigest API",
    description="Document Intelligence Engine with Custom Attention Salience Scorer and Transformers",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DOC_STORE = {
    "text": "",
    "filename": ""
}

pipeline_instance = None

def get_pipeline():
    global pipeline_instance
    if pipeline_instance is None:
        pipeline_instance = TFDeepLearningDocPipeline()
    return pipeline_instance

@app.on_event("startup")
def load_pipeline():
    get_pipeline()

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "DeepDigest", "version": "1.0.0"}

@app.get("/")
def serve_index():
    index_file = os.path.join("frontend", "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "DeepDigest API running. Visit /docs for OpenAPI specifications."}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    pipe = get_pipeline()
    contents = await file.read()
    extracted_text = pipe.extract_and_clean_pdf(io.BytesIO(contents))
    
    if not extracted_text:
        raise HTTPException(status_code=400, detail="Unable to extract text from the PDF.")
    
    DOC_STORE["text"] = extracted_text
    DOC_STORE["filename"] = file.filename
    word_count = len(extracted_text.split())

    return {
        "filename": file.filename,
        "word_count": word_count,
        "message": f"Successfully parsed {file.filename} ({word_count} words)."
    }

@app.post("/ingest_text")
def ingest_text(text: str = Form(...), filename: str = Form("direct_text.txt")):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    DOC_STORE["text"] = text.strip()
    DOC_STORE["filename"] = filename
    return {"filename": filename, "word_count": len(text.split()), "status": "stored"}

@app.post("/summarize")
def summarize_document():
    if not DOC_STORE["text"]:
        raise HTTPException(status_code=400, detail="No active document found. Upload a PDF first.")
    
    pipe = get_pipeline()
    result = pipe.process_and_summarize(DOC_STORE["text"])
    return {
        "filename": DOC_STORE["filename"],
        "summary": result["summary"],
        "salient_chunk_index": result["top_chunk_idx"],
        "attention_distribution": result["attention_distribution"]
    }

@app.post("/ask")
def ask_question(question: str = Form(...)):
    if not DOC_STORE["text"]:
        raise HTTPException(status_code=400, detail="No active document found. Upload a PDF first.")
    
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    pipe = get_pipeline()
    result = pipe.answer_query(question.strip(), DOC_STORE["text"])
    return {
        "question": question,
        "answer": result["answer"],
        "confidence": result["confidence"],
        "context": result["retrieved_context"]
    }