import os
import subprocess
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from huggingface_hub import snapshot_download
from utils.chunking import chunk_text
from utils.bias_checker import bias_check

# Optional: ONNX runtime for optimized BART
try:
    from onnxruntime import InferenceSession
except ImportError:
    InferenceSession = None

router = APIRouter()

# === Configuration ===
# Hugging Face Hub repository IDs for each model
MODEL_REPOS = {
    "pegasus": "eboafour1/pegasus",
    "bart":    "eboafour1/bart",
    "bertsum": "eboafour1/bertsum"
}
# Map user choice to model key
LENGTH_MAP = {
    "detailed": "pegasus",
    "medium":   "bart",
    "short":    "bertsum"
}

class SummarizeRequest(BaseModel):
    text: str = Field(..., description="The text or document content to summarize")
    summary_length: str = Field(
        "medium",
        description="Choose 'detailed' (Pegasus), 'medium' (BART), or 'short' (BERTSum)"
    )

@router.post("/")
def summarize(req: SummarizeRequest):
    # 1) Determine the model key from user input
    length_key = req.summary_length.lower()
    if length_key not in LENGTH_MAP:
        raise HTTPException(400, "Invalid summary_length. Use 'detailed','medium', or 'short'.")
    model_key = LENGTH_MAP[length_key]

    # 2) Download or load model from HF Hub
    hf_token = os.getenv("HF_TOKEN")
    repo_id = MODEL_REPOS[model_key]
    model_path = snapshot_download(repo_id, use_auth_token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 3) Split input into chunks for manageable inference
    chunks = chunk_text(req.text, max_words=800)
    MAX_PARTS = 10  # hierarchical summarization threshold

    # === Pegasus (Detailed) ===
    if model_key == "pegasus":
        # Load quantized Pegasus (8-bit) for speed
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path, load_in_8bit=True, device_map='auto'
        )
        abstractive = pipeline("summarization", model=model, tokenizer=tokenizer)

        # First-pass: draft summaries for each chunk
        drafts = [
            abstractive(chunk, max_length=200, min_length=50)[0]['summary_text']
            for chunk in chunks
        ]
        draft_text = "\n".join(drafts)

        # Second-pass: extract key points with BERTSum
        bert_path = snapshot_download(MODEL_REPOS['bertsum'], use_auth_token=hf_token)
        bert_tok = AutoTokenizer.from_pretrained(bert_path)
        bert_mod = AutoModelForSeq2SeqLM.from_pretrained(bert_path)
        extractive = pipeline("summarization", model=bert_mod, tokenizer=bert_tok)
        key_points = extractive(req.text, max_length=80, min_length=40)[0]['summary_text']

        combined = draft_text + "\n" + key_points
        # Hierarchical: re-summarize if too many parts
        if len(drafts) > MAX_PARTS:
            combined = abstractive(combined, max_length=150, min_length=50)[0]['summary_text']

        return {"summary": bias_check(combined)}

    # === BART (Medium) ===
    elif model_key == "bart":
        # Attempt ONNX optimization
        onnx_file = os.path.join(model_path, "model.onnx")
        if InferenceSession and not os.path.exists(onnx_file):
            subprocess.run([
                "python", "-m", "transformers.onnx",
                f"--model={model_path}", onnx_file
            ], check=True)

        if InferenceSession and os.path.exists(onnx_file):
            session = InferenceSession(onnx_file)
            def summarize_chunk(txt):
                inputs = tokenizer(txt, return_tensors='np', truncation=True)
                ort_inputs = {session.get_inputs()[0].name: inputs['input_ids'].numpy()}
                ort_outs = session.run(None, ort_inputs)
                return tokenizer.decode(ort_outs[0][0], skip_special_tokens=True)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            pipe = pipeline("summarization", model=model, tokenizer=tokenizer)
            summarize_chunk = lambda t: pipe(t, max_length=150, min_length=60)[0]['summary_text']

        # Summarize each chunk and debias
        parts = [bias_check(summarize_chunk(chunk)) for chunk in chunks]
        summary_text = "\n".join(parts)
        # Hierarchical summarization
        if len(parts) > MAX_PARTS:
            summary_text = summarize_chunk(summary_text)

        return {"summary": summary_text}

    # === BERTSum (Short) ===
    else:
        # Extractive summarization
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        pipe = pipeline("summarization", model=model, tokenizer=tokenizer)
        extracts = [
            pipe(chunk, max_length=80, min_length=20)[0]['summary_text']
            for chunk in chunks
        ]
        combined_ext = "\n".join(extracts)
        # Hierarchical if needed
        if len(extracts) > MAX_PARTS:
            combined_ext = pipe(combined_ext, max_length=100, min_length=40)[0]['summary_text']

        # Final pass: rephrase with Pegasus
        peg_path = snapshot_download(MODEL_REPOS['pegasus'], use_auth_token=hf_token)
        peg_tok  = AutoTokenizer.from_pretrained(peg_path)
        peg_mod  = AutoModelForSeq2SeqLM.from_pretrained(
            peg_path, load_in_8bit=True, device_map='auto'
        )
        rephraser = pipeline("summarization", model=peg_mod, tokenizer=peg_tok)
        final_summary = rephraser(combined_ext, max_length=100, min_length=40)[0]['summary_text']

        return {"summary": bias_check(final_summary)}

@router.post("/file")
async def summarize_file(
    file: UploadFile = File(...),
    summary_length: str = Form('medium')
):
    # Save uploaded file to temporary directory
    tmp_dir = "/tmp/bookbot"
    os.makedirs(tmp_dir, exist_ok=True)
    file_path = os.path.join(tmp_dir, file.filename)
    with open(file_path, 'wb') as f:
        f.write(await file.read())

    ext = file.filename.rsplit('.', 1)[-1].lower()
    # Convert MOBI files
    if ext == 'mobi':
        txt_path = file_path.replace('.mobi', '.txt')
        subprocess.run(['ebook-convert', file_path, txt_path], check=True)
        text = open(txt_path, 'r', encoding='utf-8').read()
    else:
        text = open(file_path, 'r', encoding='utf-8').read()

    # Reuse the summarize logic
    req = SummarizeRequest(text=text, summary_length=summary_length)
    return summarize(req)
