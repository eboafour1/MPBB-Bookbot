# Project Requirements

Below are the **needed tools, libraries, and services** for both the **frontend** and **backend** of your Bookbot AI project, clearly separated.

---

## 1. Frontend Requirements

### 1.1 Runtime & Package Manager

* **Node.js** (v16+)
* **npm** (v8+) or **yarn**

### 1.2 Core Frameworks & Libraries

* **React.js** (via `create-react-app`)
* **Material UI** components:

  * `@mui/material`
  * `@emotion/react`
  * `@emotion/styled`
* **Axios** (for HTTP requests)

### 1.3 File-Parsing Dependencies

* **`pdfjs-dist`** (PDF text extraction)
* **`epubjs`** (EPUB parsing)
* **`mammoth`** (DOCX extraction)

### 1.4 Optional Tools & Plugins

* **React Router** (if you add multiple pages)
* **Redux** (or Context API) for complex state management
* **ESLint / Prettier** for linting and formatting
* **Jest + React Testing Library** for unit tests

### 1.5 Environment Setup

1. `npx create-react-app bookbot-frontend`
2. `npm install @mui/material @emotion/react @emotion/styled axios pdfjs-dist epubjs mammoth`
3. Run: `npm start`

---

## 2. Backend Requirements

### 2.1 Runtime & Environment

* **Python 3.9+**
* **Virtual Environment** (`venv` or `conda`)

### 2.2 Web Framework & Server

* **FastAPI** (v0.85+)
* **Uvicorn** (v0.18+) for ASGI server
* **python-multipart** (for file uploads)

### 2.3 AI & NLP Libraries

* **transformers** (v4.25+)
* **torch** (PyTorch v1.12+)
* **onnxruntime** (for ONNX inference, optional)
* **datasets** (Hugging Face datasets for fine-tuning scripts)
* **sentencepiece** (if using models that require it)

### 2.4 Utility Libraries

* **Pydantic** (v1.10+)
* **python-dotenv** (for environment variables, optional)
* **Calibre CLI** (`ebook-convert` command) for MOBI→TXT conversion

### 2.5 Database (Optional)

* **PostgreSQL**: `psycopg2-binary`
* **MongoDB**: `motor`

### 2.6 Fine-Tuning & Training

* **scikit-learn** (optional, for metrics)
* **Trainer** and **TrainingArguments** from `transformers`
* **GPU** with CUDA (for efficient training)

### 2.7 OS-Level Tools

* **Git** (version control)
* **Calibre** (for `ebook-convert`)

### 2.8 Environment Setup

1. Create and activate venv:

   ```bash
   python3 -m venv backend_env
   source backend_env/bin/activate
   ```
2. Install dependencies:

   ```bash
   pip install fastapi uvicorn[standard] python-multipart transformers torch onnxruntime datasets sentencepiece pydantic
   ```
3. (Optional) Database adapters:

   ```bash
   pip install psycopg2-binary    # PostgreSQL
   pip install motor               # MongoDB
   ```
4. Ensure `ebook-convert` is installed (Calibre CLI).

---

These requirements cover **all aspects** of development, from running the React frontend to serving and fine-tuning AI models in the FastAPI backend. Let me know if you need to include or adjust anything else!
