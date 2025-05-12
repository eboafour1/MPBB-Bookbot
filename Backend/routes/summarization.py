import os
import subprocess
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    pipeline
)
from huggingface_hub import snapshot_download
from utils.chunking import chunk_text
from utils.bias_checker import bias_check

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
    length_key = req.summary_length.lower()
    if length_key not in LENGTH_MAP:
        raise HTTPException(400, "Invalid summary_length. Use 'detailed', 'medium', or 'short'.")
    model_key = LENGTH_MAP[length_key]

    hf_token = os.getenv("HF_TOKEN")
    repo_id = MODEL_REPOS[model_key]
    model_path = snapshot_download(repo_id, use_auth_token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    chunks = chunk_text(req.text, max_words=800)
    MAX_PARTS = 10

    if model_key == "pegasus":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        abstractive = pipeline("summarization", model=model, tokenizer=tokenizer)

        max_input_len = tokenizer.model_max_length
        drafts = []
        for chunk in chunks:
            tokens = tokenizer(chunk, truncation=True, max_length=max_input_len, return_tensors='pt')
            truncated_input = tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)
            summary = abstractive(truncated_input, max_length=200, min_length=50, truncation=True)[0]['summary_text']
            drafts.append(summary)

        draft_text = "\n".join(drafts)

        # Extractive pass with BERTSum
        bert_path = snapshot_download(MODEL_REPOS['bertsum'], use_auth_token=hf_token)
        bert_tok = AutoTokenizer.from_pretrained(bert_path)
        bert_mod = AutoModelForSequenceClassification.from_pretrained(bert_path)
        extractive = pipeline("text-classification", model=bert_mod, tokenizer=bert_tok, return_all_scores=True)
        sentences = req.text.replace('\n', ' ').split('. ')
        scored = []
        for sent in sentences:
            if not sent.strip():
                continue
            scores = extractive(sent)[0]
            key_score = next(item['score'] for item in scores if item['label'] in ('LABEL_1', '1'))
            scored.append((key_score, sent))
        top_n = sorted(scored, key=lambda x: x[0], reverse=True)[:5]
        key_points = '. '.join([s for _, s in top_n])

        combined = draft_text + "\n" + key_points
        if len(drafts) > MAX_PARTS:
            combined = abstractive(combined, max_length=150, min_length=50)[0]['summary_text']

        return {"summary": bias_check(combined)}

    elif model_key == "bart":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        pipe = pipeline("summarization", model=model, tokenizer=tokenizer)
        summarize_chunk = lambda t: pipe(t, max_length=150, min_length=60)[0]['summary_text']

        parts = [bias_check(summarize_chunk(chunk)) for chunk in chunks]
        summary_text = "\n".join(parts)
        if len(parts) > MAX_PARTS:
            summary_text = summarize_chunk(summary_text)

        return {"summary": summary_text}

    else:
        cls_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        classifier = pipeline("text-classification", model=cls_model, tokenizer=tokenizer, return_all_scores=True)
        sentences = [s.strip() for s in req.text.replace('\n', ' ').split('. ') if s.strip()]
        scored = []
        for sent in sentences:
            scores = classifier(sent)[0]
            key_score = next(item['score'] for item in scores if item['label'] in ('LABEL_1', '1'))
            scored.append((key_score, sent))
        top_n = sorted(scored, key=lambda x: x[0], reverse=True)[:5]
        extractive_summary = '. '.join([s for _, s in top_n])

        peg_path = snapshot_download(MODEL_REPOS['pegasus'], use_auth_token=hf_token)
        peg_tok = AutoTokenizer.from_pretrained(peg_path)
        peg_mod = AutoModelForSeq2SeqLM.from_pretrained(peg_path)
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

    ext = file.filename.rsplit('.', 1)[-1].lower()
    txt_path = file_path.rsplit('.', 1)[0] + '.txt'

    if ext in ['mobi', 'epub', 'pdf', 'docx']:
        try:
            subprocess.run(['ebook-convert', file_path, txt_path], check=True)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Conversion to TXT failed: {e}")
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

    req = SummarizeRequest(text=text, summary_length=summary_length)
    return summarize(req)
