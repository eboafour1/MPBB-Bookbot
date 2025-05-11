import os
import subprocess
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    pipeline,
)
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
MODEL_REPOS = {
    "pegasus": "eboafour1/pegasus",
    "bart":    "eboafour1/bart",
    "bertsum": "eboafour1/bertsum"
}
LENGTH_MAP = {"detailed": "pegasus", "medium": "bart", "short": "bertsum"}

class SummarizeRequest(BaseModel):
    text: str = Field(..., description="Text to summarize")
    summary_length: str = Field(
        "medium",
        description="Choose 'detailed' (Pegasus), 'medium' (BART), or 'short' (BERTSum)"
    )

@router.post("/")
def summarize(req: SummarizeRequest):
    # Determine which model to use
    length_key = req.summary_length.lower()
    if length_key not in LENGTH_MAP:
        raise HTTPException(400, "Invalid summary_length. Use 'detailed', 'medium', or 'short'.")
    model_key = LENGTH_MAP[length_key]

    # Download model from HF Hub (cached locally)
    hf_token = os.getenv("HF_TOKEN")
    repo_id = MODEL_REPOS[model_key]
    model_path = snapshot_download(repo_id, use_auth_token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Split text into word-based chunks
    chunks = chunk_text(req.text, max_words=800)
    MAX_PARTS = 10

    # === Pegasus (Detailed) ===
    if model_key == "pegasus":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path, load_in_8bit=True, device_map='auto'
        )
        abstractive = pipeline("summarization", model=model, tokenizer=tokenizer)

        drafts = [
            abstractive(chunk, max_length=200, min_length=50)[0]['summary_text']
            for chunk in chunks
        ]
        draft_text = "\n".join(drafts)

        # Extractive pass with BERTSum
        bert_path = snapshot_download(MODEL_REPOS['bertsum'], use_auth_token=hf_token)
        bert_tok = AutoTokenizer.from_pretrained(bert_path)
        bert_mod = AutoModelForSeq2SeqLM.from_pretrained(bert_path)
        extractive = pipeline("summarization", model=bert_mod, tokenizer=bert_tok)
        key_points = extractive(req.text, max_length=80, min_length=40)[0]['summary_text']

        combined = draft_text + "\n" + key_points
        if len(drafts) > MAX_PARTS:
            combined = abstractive(combined, max_length=150, min_length=50)[0]['summary_text']

        return {"summary": bias_check(combined)}

    # === BART (Medium) ===
    elif model_key == "bart":
        onnx_file = os.path.join(model_path, "model.onnx")
        if InferenceSession and not os.path.exists(onnx_file):
            subprocess.run([
                "python", "-m", "transformers.onnx",
                f"--model={model_path}", onnx_file
            ], check=True)

        if InferenceSession and os.path.exists(onnx_file):
            session = InferenceSession(onnx_file)
            def summarize_chunk(txt):
                inputs = tokenizer(txt, return_tensors='pt', truncation=True)
                ort_inputs = {session.get_inputs()[0].name: inputs['input_ids'].cpu().numpy()}
                ort_outs = session.run(None, ort_inputs)
                return tokenizer.decode(ort_outs[0][0], skip_special_tokens=True)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            pipe = pipeline("summarization", model=model, tokenizer=tokenizer)
            summarize_chunk = lambda t: pipe(t, max_length=150, min_length=60)[0]['summary_text']

        parts = [bias_check(summarize_chunk(chunk)) for chunk in chunks]
        summary_text = "\n".join(parts)
        if len(parts) > MAX_PARTS:
            summary_text = summarize_chunk(summary_text)
        return {"summary": summary_text}

    # === BERTSum (Short) via extractive ranking ===
    else:
        # Load as a classifier
        cls_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        classifier = pipeline(
            "text-classification",
            model=cls_model,
            tokenizer=tokenizer,
            return_all_scores=True
        )
        # Split into sentences
        sentences = [s.strip() for s in req.text.replace('\n', ' ').split('. ') if s.strip()]
        scored = []
        for sent in sentences:
            scores = classifier(sent)[0]
            key_score = next(item['score'] for item in scores if item['label'] in ('LABEL_1', '1'))
            scored.append((key_score, sent))
        top_n = sorted(scored, key=lambda x: x[0], reverse=True)[:5]
        extractive_summary = '. '.join([s for _, s in top_n])

        # Rephrase with Pegasus
        peg_path = snapshot_download(MODEL_REPOS['pegasus'], use_auth_token=hf_token)
        peg_tok = AutoTokenizer.from_pretrained(peg_path)
        peg_mod = AutoModelForSeq2SeqLM.from_pretrained(
            peg_path, load_in_8bit=True, device_map='auto'
        )
        rephraser = pipeline("summarization", model=peg_mod, tokenizer=peg_tok)
        final = rephraser(extractive_summary, max_length=100, min_length=40)[0]['summary_text']
        return {"summary": bias_check(final)}

@router.post("/file")
async def summarize_file(
    file: UploadFile = File(...),
    summary_length: str = Form('medium')
):
    tmp_dir = "/tmp/bookbot"
    os.makedirs(tmp_dir, exist_ok=True)
    file_path = os.path.join(tmp_dir, file.filename)
    with open(file_path, 'wb') as f:
        f.write(await file.read())

    # Determine extension and convert to TXT if needed
    ext = file.filename.rsplit('.', 1)[-1].lower()
    txt_path = file_path.rsplit('.', 1)[0] + '.txt'
    # Use Calibre's ebook-convert for supported formats
    if ext in ['mobi', 'epub', 'pdf', 'docx']:
        try:
            subprocess.run(['ebook-convert', file_path, txt_path], check=True)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Conversion to TXT failed: {e}")
        # Read the converted TXT
        try:
            with open(txt_path, 'r', encoding='utf-8') as tf:
                text = tf.read()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed reading converted TXT: {e}")
    elif ext == 'txt':
        try:
            with open(file_path, 'r', encoding='utf-8') as tf:
                text = tf.read()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed reading TXT file: {e}")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Supported: .txt, .mobi, .epub, .pdf, .docx")

    # Delegate to summarize() with converted text
    req = SummarizeRequest(text=text, summary_length=summary_length)
    return summarize(req)