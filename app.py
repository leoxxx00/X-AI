import os
import re
import time
import traceback
import logging
from typing import List, Dict, TypedDict, Optional
from dataclasses import dataclass, field

import torch
import pandas as pd
import gradio as gr

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# ============================================================
# HUGGING FACE SPACES READY
# Medical CSV RAG Chatbot
# Optimized pipeline:
# RAG retrieval -> local ECG adapter reasoning -> grounded summary
# UI goal:
# polished mobile-friendly chatbot UX with minimal sources panel
# ============================================================

# -------------------------------
# LOGGING
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# -------------------------------
# CONFIG
# -------------------------------
@dataclass
class Config:
    # Paths
    base_model_path: str = os.getenv(
        "BASE_MODEL_PATH",
        "meta-llama/Llama-3.1-8B-Instruct"
    )
    adapter_dir: str = os.getenv(
        "ADAPTER_DIR",
        "adapter_refined_v10"
    )
    data_csv: str = os.getenv(
        "DATA_CSV",
        "RAGmaterials/ECG_RAG_only_clean.csv"
    )
    rag_dir: str = os.getenv(
        "RAG_DIR",
        "RAGmaterials"
    )
    vectorstore_dir: str = field(init=False)

    # Auth / APIs
    hf_token: str = os.getenv("HF_TOKEN", "")
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    # DeepSeek generation
    deepseek_temperature: float = float(os.getenv("DEEPSEEK_TEMPERATURE", "0.1"))
    deepseek_max_tokens: int = int(os.getenv("DEEPSEEK_MAX_TOKENS", "550"))

    # Embeddings
    embed_model_name: str = os.getenv(
        "EMBED_MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Retrieval
    similarity_k: int = int(os.getenv("SIMILARITY_K", "12"))
    top_k_final: int = int(os.getenv("TOP_K_FINAL", "4"))
    max_context_chars: int = int(os.getenv("MAX_CONTEXT_CHARS", "5200"))

    # Generation
    max_input_len: int = int(os.getenv("MAX_INPUT_LEN", "4096"))
    max_new_tokens_local: int = int(os.getenv("MAX_NEW_TOKENS_LOCAL", "180"))
    max_chat_history_turns: int = int(os.getenv("MAX_CHAT_HISTORY_TURNS", "6"))

    # Filtering
    min_lexical_overlap: float = float(os.getenv("MIN_LEXICAL_OVERLAP", "0.08"))
    min_faiss_similarity: float = float(os.getenv("MIN_FAISS_SIMILARITY", "0.20"))
    strong_retrieval_threshold: float = float(os.getenv("STRONG_RETRIEVAL_THRESHOLD", "0.30"))
    strong_retrieval_min_docs: int = int(os.getenv("STRONG_RETRIEVAL_MIN_DOCS", "3"))

    # Features
    use_query_cache: bool = os.getenv("USE_QUERY_CACHE", "true").lower() == "true"
    enable_query_expansion: bool = os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() == "true"
    enable_validator: bool = os.getenv("ENABLE_VALIDATOR", "true").lower() == "true"
    enable_typewriter_stream: bool = os.getenv("ENABLE_TYPEWRITER_STREAM", "true").lower() == "true"
    show_debug_panel: bool = os.getenv("SHOW_DEBUG_PANEL", "true").lower() == "true"
    allow_rebuild_vectorstore: bool = os.getenv("ALLOW_REBUILD_VECTORSTORE", "false").lower() == "true"

    # Model loading
    use_4bit: bool = os.getenv("USE_4BIT", "true").lower() == "true"

    # Launch
    launch_debug: bool = os.getenv("LAUNCH_DEBUG", "false").lower() == "true"
    server_name: str = os.getenv("SERVER_NAME", "0.0.0.0")
    server_port: int = int(os.getenv("SERVER_PORT", "7860"))

    # UI timings
    blink_stage_1: float = float(os.getenv("BLINK_STAGE_1", "0.40"))
    blink_stage_2: float = float(os.getenv("BLINK_STAGE_2", "0.55"))
    blink_stage_3: float = float(os.getenv("BLINK_STAGE_3", "0.50"))
    blink_before_answer: float = float(os.getenv("BLINK_BEFORE_ANSWER", "0.25"))

    def __post_init__(self):
        self.vectorstore_dir = os.path.join(self.rag_dir, "faiss_store")
        os.makedirs(self.rag_dir, exist_ok=True)

        if not self.deepseek_api_key:
            raise ValueError("Missing DEEPSEEK_API_KEY. Add it in Hugging Face Space Secrets.")

        for path, name in [
            (self.adapter_dir, "Adapter directory"),
            (self.data_csv, "CSV data"),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} not found at: {path}")


cfg = Config()
logger.info("Configuration loaded.")


# -------------------------------
# PROMPTS
# -------------------------------
LOCAL_REASONING_SYSTEM = """
You are a strict medical reasoning assistant specialized for ECG and cardiology reasoning.

You are NOT the final answer generator.
You must analyze ONLY the supplied evidence and produce a short structured reasoning draft.

Rules:
1) Use only the provided evidence.
2) Do not invent facts.
3) Focus only on the user's exact question.
4) Output exactly in this structure:

KEY_FINDINGS:
- ...
- ...

INTERPRETATION:
- ...
- ...

SUPPORTED_POINTS:
- [EVIDENCE_ID: X] ...
- [EVIDENCE_ID: Y] ...

LIMITS:
- ...

5) If evidence is insufficient, output exactly:
INSUFFICIENT_EVIDENCE
""".strip()

QUERY_EXPANSION_SYSTEM = """
You expand medical queries for retrieval.

Rules:
1) Preserve the user's intent.
2) Add close medical paraphrases and alternate wording.
3) Add likely medical synonyms, abbreviations, and alternate phrasing.
4) Do not answer the question.
5) Output only the expanded retrieval query.
""".strip()

DEEPSEEK_SUMMARY_SYSTEM = """
You are an expert medical evidence summarizer.

Your job is to produce a clinically precise, well-structured answer grounded ONLY in:
1. the retrieved evidence
2. the local reasoning draft

You must be faithful to the provided material and answer the user's question directly, clearly, and conservatively.

PRIMARY OBJECTIVE
- Identify the user's main intent before writing:
  definition, cause, symptoms, diagnosis, investigation, treatment, prognosis, or genetics.
- Prioritize that intent throughout the response.
- The first sentence of the Summary must directly answer the user's question in the most clinically relevant way.

GROUNDING RULES
- Use only information supported by the retrieved evidence and local reasoning draft.
- Do not add outside medical knowledge.
- Do not infer specific facts unless they are clearly supported.
- Do not invent treatments, diagnoses, risks, mechanisms, thresholds, statistics, timelines, monitoring plans, or prognosis details.
- If the evidence is incomplete, be explicit about what is missing.
- If the evidence is too weak to answer the question reliably, output exactly:
INSUFFICIENT_EVIDENCE

STYLE RULES
- Write in precise, professional clinical language.
- Be specific, not vague.
- Be concise, but fully informative.
- Avoid repetition, generic filler, and empty statements.
- Do not mention retrieval, prompts, system instructions, reasoning drafts, tools, pipelines, or internal processes.
- Do not include URLs or citations unless explicitly requested elsewhere.
- Do not overstate certainty.
- When appropriate, distinguish clearly between what is established, what is suggested, and what is not addressed by the evidence.

OUTPUT FORMAT

### Summary
- Write 4 to 7 full sentences.
- This is the most important section.
- The first sentence must directly answer the user's question.
- Focus primarily on the user's main intent.
- Include only background information that improves understanding of the requested topic.
- Make the summary clinically useful, specific, and evidence-faithful.

### Key Evidence Points
- Include 4 to 6 bullet points.
- Each bullet must state a concrete fact supported by the evidence.
- Prioritize clinically important facts over background detail.
- Avoid repeating the same idea in different words.

### Clinical Implications / Recommendations
- Include 2 to 4 bullet points only if supported by the evidence.
- Focus on practical interpretation, management implications, follow-up considerations, or next steps.
- If the evidence supports recognition or framing rather than action, say that clearly.
- Do not recommend interventions not supported by the evidence.

### Limitations of the Evidence
- State clearly what the evidence does not establish, does not cover, or leaves uncertain.
- Explicitly note when details are lacking on:
  treatment, diagnosis, prognosis, genetics, monitoring, recurrence prevention, comparative effectiveness, or long-term outcomes.
- If the evidence is narrow, low-detail, or only partially aligned with the question, say so plainly.

SPECIAL INSTRUCTIONS BY QUESTION TYPE

For treatment questions:
- Focus primarily on treatment and management, not disease definition.
- Organize treatment information in this order whenever supported by the evidence:
  1. supportive or conservative care
  2. symptomatic drug therapy or procedural treatment
  3. long-term prevention, follow-up, or recurrence prevention
- Distinguish treatment of active symptoms from prevention of recurrence or complications.
- If the condition is benign, self-limited, or often does not require treatment, state that clearly in the first sentence.

For diagnosis or investigation questions:
- Focus on how the condition is identified, evaluated, or differentiated.
- Prioritize diagnostic features, testing approach, and clinically useful distinctions.
- Do not drift into treatment unless the evidence clearly supports it and it helps answer the question.

For cause or risk questions:
- Focus on etiologies, risk factors, mechanisms, or associations supported by the evidence.
- Distinguish established causes from possible contributors if the evidence is less certain.

For prognosis questions:
- Focus on expected course, complications, recurrence, or outcome-related information supported by the evidence.
- Do not add prognostic claims not explicitly supported.

QUALITY CHECK BEFORE OUTPUT
Before finalizing, ensure that:
- the first sentence directly answers the question
- the response matches the user's primary intent
- every important claim is grounded in the provided material
- no unsupported medical detail has been added
- the Limitations section honestly reflects evidence gaps

If these conditions cannot be met, output exactly:
INSUFFICIENT_EVIDENCE
""".strip()

VALIDATOR_SYSTEM = """
You are a strict medical evidence validator.

Your job is to compare the ANSWER against the EVIDENCE.

Rules:
1) Mark SUPPORTED if the answer is well grounded in the evidence.
2) Mark PARTLY_UNSUPPORTED if some claims are supported but others go beyond the evidence.
3) Mark INSUFFICIENT_EVIDENCE if the answer is mostly unsupported or the evidence is too weak.
4) Output only one short verdict line beginning with exactly one of:
SUPPORTED:
PARTLY_UNSUPPORTED:
INSUFFICIENT_EVIDENCE:
""".strip()


# -------------------------------
# HELPERS
# -------------------------------
def clean_text(x: str) -> str:
    x = str(x).replace("\x00", " ").strip()
    x = re.sub(r"\s+", " ", x)
    return x


def strip_bad_sections(txt: str) -> str:
    t = str(txt).strip()
    cut_markers = [
        "References:",
        "Sources:",
        "Source:",
        "URLs:",
        "This response is based",
        "Please let me know",
        "Is there anything else",
    ]
    for marker in cut_markers:
        pos = t.lower().find(marker.lower())
        if pos != -1:
            t = t[:pos].strip()

    t = re.sub(r"https?://\S+|www\.\S+", "", t).strip()
    return t


def infer_tags(question: str, answer: str) -> List[str]:
    text = f"{question} {answer}".lower()
    tags: List[str] = []

    keyword_map = {
        "treatment": ["treat", "therapy", "management", "drug", "surgery"],
        "diagnosis": ["diagnosis", "diagnose", "criteria"],
        "symptoms": ["symptom", "presentation", "sign", "feature"],
        "ecg": ["ecg", "ekg", "st elevation", "qrs", "p wave", "arrhythmia", "tachycardia", "bradycardia"],
        "investigation": ["test", "investigation", "mri", "ct", "lab", "imaging"],
        "prognosis": ["prognosis", "outcome", "survival", "risk"],
        "genetics": ["gene", "genetic", "mutation", "variant", "chromosome", "inherited", "inheritance"],
        "etiology": ["cause", "causes", "caused by", "associated with", "risk factor"],
    }

    for tag, words in keyword_map.items():
        if any(w in text for w in words):
            tags.append(tag)

    return tags


def make_row_text(q: str, a: str) -> str:
    return f"QUESTION:\n{q}\n\nANSWER:\n{a}".strip()


def score_to_similarity(raw_score: float) -> float:
    try:
        raw_score = float(raw_score)
    except Exception:
        return -1.0
    return 1.0 / (1.0 + max(raw_score, 0.0))


def lexical_overlap(query: str, text: str) -> float:
    q_words = set(re.findall(r"\w+", query.lower()))
    t_words = set(re.findall(r"\w+", text.lower()))
    if not q_words:
        return 0.0
    return len(q_words & t_words) / max(1, len(q_words))


def rerank_docs(query: str, docs: List[Document], top_n: Optional[int] = None) -> List[Document]:
    if top_n is None:
        top_n = cfg.top_k_final

    q_words = set(re.findall(r"\w+", query.lower()))
    scored = []

    for d in docs:
        question = d.metadata.get("question", "")
        answer = d.metadata.get("answer", "")
        tags = " ".join(d.metadata.get("tags", []))
        text = f"{question} {answer} {tags}".lower()

        t_words = set(re.findall(r"\w+", text))
        overlap = len(q_words & t_words) / max(1, len(q_words))
        question_boost = 0.20 if any(w in question.lower() for w in q_words) else 0.0
        tag_boost = 0.10 if any(w in tags.lower() for w in q_words) else 0.0
        sim_score = float(d.metadata.get("sim_score", 0.0))

        final_score = overlap + question_boost + tag_boost + (0.35 * sim_score)
        scored.append((d, final_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in scored[:top_n]]


def history_to_text(chat_history: List[Dict[str, str]], max_turns: Optional[int] = None) -> str:
    if max_turns is None:
        max_turns = cfg.max_chat_history_turns

    items = chat_history[-max_turns:]
    if not items:
        return "[EMPTY]"

    return "\n".join([f"{m['role'].upper()}: {m['content']}" for m in items]).strip()


def build_context_string(docs: List[Document], max_chars: Optional[int] = None) -> str:
    if max_chars is None:
        max_chars = cfg.max_context_chars

    blocks = []
    total = 0

    for i, d in enumerate(docs, 1):
        q = d.metadata.get("question", "")
        a = d.metadata.get("answer", "")
        tags = ", ".join(d.metadata.get("tags", [])) or "N/A"
        sim = d.metadata.get("sim_score", None)

        block = f"""
==============================
EVIDENCE_ID: {i}
SOURCE_ID: {d.metadata.get('id')}
SOURCE_QUESTION: {q}
SOURCE_TAGS: {tags}
SIMILARITY: {sim if sim is not None else 'N/A'}
EVIDENCE_TEXT:
{a}
==============================
""".strip()

        if total + len(block) > max_chars:
            break

        blocks.append(block)
        total += len(block) + 2

    return "\n\n".join(blocks).strip()


def compute_confidence(result: Dict) -> float:
    best_score = result.get("best_score", -1.0)
    validation = result.get("validation_status", "")

    if validation.startswith("SUPPORTED"):
        conf = best_score
    elif validation.startswith("PARTLY_UNSUPPORTED"):
        conf = best_score * 0.70
    else:
        conf = best_score * 0.40

    return max(0.0, min(1.0, conf))


def strong_retrieval(best_score: float, docs: List[Document]) -> bool:
    return (
        best_score >= cfg.strong_retrieval_threshold
        and len(docs) >= cfg.strong_retrieval_min_docs
    )


def stream_text(text: str, step: int = 110):
    acc = ""
    for i in range(0, len(text), step):
        acc += text[i:i + step]
        yield acc


# -------------------------------
# EMBEDDINGS + VECTORSTORE
# -------------------------------
logger.info("Loading embeddings...")
embeddings = HuggingFaceEmbeddings(model_name=cfg.embed_model_name)


def build_vectorstore():
    logger.info(f"Reading CSV: {cfg.data_csv}")
    df = pd.read_csv(cfg.data_csv)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"instruction", "response"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {required}. Found: {df.columns.tolist()}")

    df = df[["instruction", "response"]].dropna().reset_index(drop=True)
    df["instruction"] = df["instruction"].map(clean_text)
    df["response"] = df["response"].map(clean_text)

    docs = []
    for i, row in df.iterrows():
        q = row["instruction"]
        a = row["response"]
        docs.append(
            Document(
                page_content=make_row_text(q, a),
                metadata={
                    "id": int(i),
                    "question": q,
                    "answer": a,
                    "tags": infer_tags(q, a),
                }
            )
        )

    vectorstore_local = FAISS.from_documents(docs, embeddings)
    vectorstore_local.save_local(cfg.vectorstore_dir)
    logger.info(f"Saved vectorstore with {len(docs)} docs to {cfg.vectorstore_dir}")


def load_vectorstore():
    return FAISS.load_local(
        cfg.vectorstore_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )


if not os.path.exists(cfg.vectorstore_dir):
    logger.info("Vectorstore not found. Building from CSV...")
    build_vectorstore()

vectorstore = load_vectorstore()
logger.info("Vectorstore ready.")


# -------------------------------
# LOCAL MODEL + ECG ADAPTER
# -------------------------------
logger.info("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    cfg.base_model_path,
    use_fast=True,
    token=cfg.hf_token if cfg.hf_token else None
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

logger.info("Loading base model...")
has_cuda = torch.cuda.is_available()
base_model = None

if cfg.use_4bit and has_cuda:
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_path,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            token=cfg.hf_token if cfg.hf_token else None,
        )
        logger.info("Loaded base model in 4-bit mode.")
    except Exception as e:
        logger.warning(f"4-bit load failed: {e}")

if base_model is None:
    dtype = torch.float16 if has_cuda else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_path,
        device_map="auto" if has_cuda else None,
        torch_dtype=dtype,
        token=cfg.hf_token if cfg.hf_token else None,
    )
    if not has_cuda:
        base_model = base_model.to("cpu")
    logger.info("Loaded base model without 4-bit.")

base_model.eval()

logger.info("Loading ECG reasoning adapter...")
reason_model = PeftModel.from_pretrained(base_model, cfg.adapter_dir)
reason_model.eval()


def get_primary_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def run_local_reasoner(user_query: str, context: str) -> str:
    try:
        messages = [
            {"role": "system", "content": LOCAL_REASONING_SYSTEM},
            {
                "role": "user",
                "content": f"QUESTION:\n{user_query}\n\nEVIDENCE:\n{context if context.strip() else '[EMPTY]'}"
            },
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.max_input_len,
        )

        model_device = get_primary_model_device(reason_model)
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        out = reason_model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens_local,
            do_sample=False,
            use_cache=True,
            repetition_penalty=1.08,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        gen_ids = out[0, inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        text = strip_bad_sections(text)

        return text if text else "INSUFFICIENT_EVIDENCE"

    except Exception as e:
        logger.error(f"Local reasoner error: {e}")
        traceback.print_exc()
        return "INSUFFICIENT_EVIDENCE"


# -------------------------------
# REMOTE LLM (DEEPSEEK)
# -------------------------------
deepseek_llm = ChatOpenAI(
    model=cfg.deepseek_model,
    api_key=cfg.deepseek_api_key,
    base_url=cfg.deepseek_base_url,
    temperature=cfg.deepseek_temperature,
    max_tokens=cfg.deepseek_max_tokens,
)

_query_expansion_cache: Dict[str, str] = {}


def llm_text(system_prompt: str, user_prompt: str, fallback: str = "INSUFFICIENT_EVIDENCE") -> str:
    try:
        resp = deepseek_llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
        text = resp.content if hasattr(resp, "content") else str(resp)
        text = strip_bad_sections(text)
        return text if text.strip() else fallback
    except Exception as e:
        logger.error(f"DeepSeek error: {e}")
        traceback.print_exc()
        return fallback


def run_query_expansion(user_query: str) -> str:
    if not cfg.enable_query_expansion:
        return user_query

    if cfg.use_query_cache and user_query in _query_expansion_cache:
        logger.info(f"Using cached expansion for: {user_query[:80]}")
        return _query_expansion_cache[user_query]

    prompt = f"""
USER_QUERY:
{user_query}

Expand this for retrieval with close medical phrasing, synonyms, and alternate wording.
Do not answer the question.
""".strip()

    expanded = llm_text(QUERY_EXPANSION_SYSTEM, prompt, fallback=user_query)
    expanded = expanded.strip() if expanded else user_query

    if cfg.use_query_cache:
        _query_expansion_cache[user_query] = expanded

    return expanded


def run_deepseek_summary(
    user_query: str,
    context: str,
    reasoning_draft: str,
    chat_history: List[Dict[str, str]],
) -> str:
    prompt = f"""
CHAT_HISTORY:
{history_to_text(chat_history)}

USER_QUESTION:
{user_query}

RETRIEVED_EVIDENCE:
{context if context.strip() else '[EMPTY]'}

LOCAL_REASONING_DRAFT:
{reasoning_draft if reasoning_draft.strip() else '[EMPTY]'}

Write a grounded final summary answer using only the evidence and reasoning draft.
""".strip()

    return llm_text(
        DEEPSEEK_SUMMARY_SYSTEM,
        prompt,
        fallback="I could not generate a grounded summary from the retrieved evidence."
    )


def run_validator(context: str, answer: str) -> str:
    if not cfg.enable_validator:
        return "SUPPORTED (validator disabled)"

    prompt = f"""
EVIDENCE:
{context if context.strip() else '[EMPTY]'}

ANSWER:
{answer if answer.strip() else '[EMPTY]'}
""".strip()

    return llm_text(VALIDATOR_SYSTEM, prompt, fallback="PARTLY_UNSUPPORTED: validator unavailable")


# -------------------------------
# WARMUP
# -------------------------------
def warmup_models():
    logger.info("Warming up local reasoner...")
    try:
        _ = run_local_reasoner(
            "What are ECG findings in hyperkalemia?",
            """
==============================
EVIDENCE_ID: 1
SOURCE_QUESTION: What are ECG findings in hyperkalemia?
SOURCE_TAGS: ecg
EVIDENCE_TEXT:
Hyperkalemia may cause peaked T waves, PR prolongation, QRS widening, and severe conduction abnormalities.
==============================
""".strip(),
        )
        logger.info("Warmup completed.")
    except Exception as e:
        logger.warning(f"Warmup failed: {e}")


warmup_models()


# -------------------------------
# STATE
# -------------------------------
class ChatState(TypedDict, total=False):
    user_query: str
    expanded_query: str
    chat_history: List[Dict[str, str]]

    retrieved_docs: List[Document]
    best_score: float
    used_context: bool
    context: str
    retrieval_attempts: int
    retrieval_mode: str

    reasoning_draft: str
    final_answer: str
    validation_status: str


# -------------------------------
# RETRIEVAL
# -------------------------------
def retrieve_docs_once(query_for_search: str, original_query: str):
    try:
        scored = vectorstore.similarity_search_with_score(
            query_for_search,
            k=cfg.similarity_k,
        )
    except Exception as e:
        logger.error(f"Retriever error: {e}")
        traceback.print_exc()
        return [], -1.0

    if not scored:
        return [], -1.0

    filtered_docs = []
    best_score = -1.0

    for doc, raw_score in scored:
        sim = score_to_similarity(raw_score)
        best_score = max(best_score, sim)

        q = doc.metadata.get("question", "")
        a = doc.metadata.get("answer", "")
        ov = lexical_overlap(original_query, f"{q} {a}")

        if ov >= cfg.min_lexical_overlap and sim >= cfg.min_faiss_similarity:
            new_doc = Document(page_content=doc.page_content, metadata=dict(doc.metadata))
            new_doc.metadata["sim_score"] = sim
            new_doc.metadata["lexical_overlap"] = ov
            filtered_docs.append(new_doc)

    reranked = rerank_docs(original_query, filtered_docs, top_n=cfg.top_k_final)
    return reranked, best_score


# -------------------------------
# LANGGRAPH NODES
# -------------------------------
def retrieve_node(state: ChatState) -> ChatState:
    query = state.get("expanded_query") or state["user_query"]
    retrieval_attempts = int(state.get("retrieval_attempts", 0)) + 1
    retrieval_mode = "expanded" if state.get("expanded_query") else "original"

    docs, best_score = retrieve_docs_once(
        query_for_search=query,
        original_query=state["user_query"],
    )

    if not docs:
        return {
            "retrieved_docs": [],
            "best_score": best_score,
            "used_context": False,
            "context": "",
            "retrieval_attempts": retrieval_attempts,
            "retrieval_mode": retrieval_mode,
        }

    return {
        "retrieved_docs": docs,
        "best_score": best_score,
        "used_context": True,
        "context": build_context_string(docs, max_chars=cfg.max_context_chars),
        "retrieval_attempts": retrieval_attempts,
        "retrieval_mode": retrieval_mode,
    }


def should_retry_retrieval(state: ChatState) -> str:
    used_context = state.get("used_context", False)
    best_score = state.get("best_score", -1.0)
    attempts = int(state.get("retrieval_attempts", 0))

    if used_context and best_score >= cfg.min_faiss_similarity:
        return "local_reasoning"

    if not cfg.enable_query_expansion:
        return "local_reasoning"

    if attempts >= 2:
        return "local_reasoning"

    return "expand_query"


def expand_query_node(state: ChatState) -> ChatState:
    expanded = run_query_expansion(state["user_query"])
    if not expanded.strip():
        expanded = state["user_query"]
    return {"expanded_query": expanded}


def local_reasoning_node(state: ChatState) -> ChatState:
    context = state.get("context", "").strip()
    if not context:
        return {"reasoning_draft": "INSUFFICIENT_EVIDENCE"}

    reasoning = run_local_reasoner(state["user_query"], context)
    return {"reasoning_draft": reasoning}


def generate_node(state: ChatState) -> ChatState:
    context = state.get("context", "").strip()
    reasoning = state.get("reasoning_draft", "INSUFFICIENT_EVIDENCE")
    history = state.get("chat_history", [])

    if not context:
        return {"final_answer": "I could not find sufficiently relevant evidence in the RAG database for this question."}

    answer = run_deepseek_summary(
        user_query=state["user_query"],
        context=context,
        reasoning_draft=reasoning,
        chat_history=history,
    )
    return {"final_answer": answer}


def validate_node(state: ChatState) -> ChatState:
    context = state.get("context", "").strip()
    answer = state.get("final_answer", "").strip()
    best_score = state.get("best_score", -1.0)
    docs = state.get("retrieved_docs", [])

    if not context or not answer:
        return {"validation_status": "INSUFFICIENT_EVIDENCE: missing context or answer"}

    if strong_retrieval(best_score, docs):
        return {"validation_status": "SUPPORTED (validator skipped due to strong retrieval)"}

    verdict = run_validator(context, answer)

    if verdict.startswith("SUPPORTED"):
        return {"validation_status": verdict}

    if verdict.startswith("PARTLY_UNSUPPORTED"):
        return {
            "validation_status": verdict,
            "final_answer": answer + "\n\nEvidence limits: some parts may not be fully supported by the retrieved evidence."
        }

    if verdict.startswith("INSUFFICIENT_EVIDENCE"):
        return {
            "validation_status": verdict,
            "final_answer": answer + "\n\nEvidence limits: the retrieved evidence was weak or only partially relevant."
        }

    return {"validation_status": verdict}


def finalize_node(state: ChatState) -> ChatState:
    answer = strip_bad_sections(state.get("final_answer", ""))
    if not answer:
        answer = "I could not generate an answer."
    return {"final_answer": answer}


# -------------------------------
# GRAPH
# -------------------------------
builder = StateGraph(ChatState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("expand_query", expand_query_node)
builder.add_node("local_reasoning", local_reasoning_node)
builder.add_node("generate", generate_node)
builder.add_node("validate", validate_node)
builder.add_node("finalize", finalize_node)

builder.add_edge(START, "retrieve")
builder.add_conditional_edges(
    "retrieve",
    should_retry_retrieval,
    {
        "expand_query": "expand_query",
        "local_reasoning": "local_reasoning",
    }
)
builder.add_edge("expand_query", "retrieve")
builder.add_edge("local_reasoning", "generate")
builder.add_edge("generate", "validate")
builder.add_edge("validate", "finalize")
builder.add_edge("finalize", END)

graph = builder.compile()
logger.info("LangGraph compiled.")


# -------------------------------
# FORMATTING HELPERS
# -------------------------------
def format_sources_minimal(result: Optional[Dict]) -> str:
    if not result:
        return "## Retrieved Sources\n\nNo sources yet."

    docs = result.get("retrieved_docs", [])
    best_score = result.get("best_score", -1.0)

    if not docs:
        return (
            "## Retrieved Sources\n\n"
            "No sufficiently relevant evidence retrieved.\n\n"
            f"**Best score:** `{best_score:.3f}`"
        )

    lines = [
        "## Retrieved Sources",
        f"**Best score:** `{best_score:.3f}`",
        "",
    ]

    for i, d in enumerate(docs, 1):
        question = d.metadata.get("question", "")
        answer = d.metadata.get("answer", "")
        similarity = d.metadata.get("sim_score", "N/A")
        preview = answer[:210].strip()
        if len(answer) > 210:
            preview += "..."

        lines.extend([
            f"### Evidence {i}",
            f"- **Question:** {question}",
            f"- **Similarity:** `{similarity}`",
            f"- **Preview:** {preview}",
            "",
        ])

    return "\n".join(lines)


def format_debug_text(result: Optional[Dict]) -> str:
    if not result:
        return "No debug result yet."

    return f"""
BEST SCORE: {result.get('best_score', -1.0)}
USED CONTEXT: {result.get('used_context', False)}
RETRIEVAL ATTEMPTS: {result.get('retrieval_attempts', 0)}
RETRIEVAL MODE: {result.get('retrieval_mode', 'N/A')}
VALIDATION STATUS: {result.get('validation_status', 'N/A')}

----- CONTEXT -----
{result.get('context', '')}

----- LOCAL REASONING DRAFT -----
{result.get('reasoning_draft', '')}
""".strip()


# -------------------------------
# UI HELPERS
# -------------------------------
CUSTOM_CSS = """
:root {
    --bg-main: #07111f;
    --bg-soft: #0b1728;
    --card: rgba(10, 19, 35, 0.86);
    --card-2: rgba(14, 25, 43, 0.94);
    --border: rgba(148, 163, 184, 0.16);
    --text: #e5eefb;
    --muted: #94a3b8;
    --primary: #7c3aed;
    --primary-2: #2563eb;
    --success: #10b981;
}

html, body, .gradio-container {
    margin: 0 !important;
    padding: 0 !important;
    min-height: 100%;
    background:
        radial-gradient(circle at top left, rgba(124,58,237,0.22), transparent 28%),
        radial-gradient(circle at top right, rgba(37,99,235,0.18), transparent 24%),
        linear-gradient(180deg, #050b16 0%, #091321 100%);
    color: var(--text);
}

.gradio-container {
    max-width: 100% !important;
    padding: 12px !important;
}

footer {
    visibility: hidden;
}

.top-card {
    border: 1px solid var(--border);
    background: linear-gradient(135deg, rgba(11,23,40,0.95), rgba(18,31,56,0.92));
    border-radius: 22px;
    padding: 16px;
    margin-bottom: 12px;
    box-shadow: 0 14px 40px rgba(0,0,0,0.20);
}

.hero-title {
    font-size: 1.6rem;
    font-weight: 800;
    color: #f8fbff;
    margin-bottom: 6px;
    line-height: 1.15;
}

.hero-subtitle {
    color: #cbd5e1;
    font-size: 0.95rem;
    line-height: 1.5;
}

.badges {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 12px;
}

.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 10px;
    border-radius: 999px;
    font-size: 11px;
    color: #e6eefc;
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.06);
}

.panel-wrap {
    border: 1px solid var(--border);
    background: linear-gradient(180deg, rgba(10,19,35,0.96), rgba(7,14,26,0.94));
    border-radius: 20px;
    padding: 12px;
    box-shadow: 0 16px 45px rgba(0,0,0,0.22);
}

#chatbot {
    height: min(62vh, 640px) !important;
    min-height: 360px !important;
    border-radius: 18px !important;
    border: 1px solid var(--border) !important;
    overflow: hidden !important;
    box-shadow: 0 14px 40px rgba(0,0,0,0.26) !important;
}

.status-card {
    padding: 12px 14px;
    border-radius: 16px;
    background: linear-gradient(135deg, #0f172a 0%, #172554 100%);
    color: #f9fafb;
    font-size: 14px;
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}

.muted {
    color: #a5b4fc;
    font-size: 12px;
}

.blink-dots {
    font-size: 22px;
    font-weight: 800;
    letter-spacing: 4px;
    animation: blinkDots 1s steps(1, end) infinite;
    display: inline-block;
    padding: 2px 0;
}

@keyframes blinkDots {
    0% { opacity: 1; }
    50% { opacity: 0.15; }
    100% { opacity: 1; }
}

textarea, .gr-textbox textarea {
    border-radius: 16px !important;
    font-size: 15px !important;
}

.gr-textbox label, .gr-markdown, .gr-button {
    font-size: 14px !important;
}

button {
    border-radius: 14px !important;
    min-height: 44px !important;
    font-weight: 600 !important;
}

.mobile-stack {
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.mobile-scroll {
    max-height: 34vh;
    overflow-y: auto;
}

.command-note {
    color: #cbd5e1;
    font-size: 0.88rem;
    line-height: 1.45;
}

@media (max-width: 1024px) {
    .gradio-container {
        padding: 10px !important;
    }

    .hero-title {
        font-size: 1.45rem;
    }

    .hero-subtitle {
        font-size: 0.92rem;
    }

    #chatbot {
        height: 56vh !important;
    }
}

@media (max-width: 768px) {
    .gradio-container {
        padding: 8px !important;
    }

    .top-card {
        padding: 14px;
        border-radius: 18px;
    }

    .hero-title {
        font-size: 1.28rem;
    }

    .hero-subtitle {
        font-size: 0.88rem;
        line-height: 1.45;
    }

    .badge {
        font-size: 10px;
        padding: 5px 8px;
    }

    .panel-wrap {
        padding: 10px;
        border-radius: 16px;
    }

    #chatbot {
        height: 52vh !important;
        min-height: 320px !important;
        border-radius: 16px !important;
    }

    button {
        width: 100% !important;
    }

    .mobile-scroll {
        max-height: 240px;
    }
}

@media (max-width: 480px) {
    .hero-title {
        font-size: 1.15rem;
    }

    .hero-subtitle {
        font-size: 0.83rem;
    }

    #chatbot {
        height: 50vh !important;
        min-height: 300px !important;
    }

    textarea, .gr-textbox textarea {
        font-size: 14px !important;
    }
}
"""


def hero_html() -> str:
    return """
    <div class="top-card">
        <div class="hero-title">🫀 Mr Cardio</div>
        <div class="hero-subtitle">
            ECG-focused clinical chatbot with RAG retrieval, local ECG reasoning,
            and grounded evidence summaries. Mobile-friendly layout included.
        </div>
        <div class="badges">
            <div class="badge">ECG Reasoning</div>
            <div class="badge">FAISS Retrieval</div>
            <div class="badge">LoRA Adapter</div>
            <div class="badge">Validated Output</div>
        </div>
    </div>
    """


def thinking_html(stage: str) -> str:
    return f"""
    <div class="status-card">
        <div style="display:flex;align-items:center;gap:12px;">
            <div style="font-size:19px;">⏳</div>
            <div>
                <div style="font-weight:700;">{stage}</div>
                <div class="muted">Retrieval → reasoning → grounded answer</div>
                <div class="blink-dots">...</div>
            </div>
        </div>
    </div>
    """


def initialize_session():
    return {"chat_history": [], "last_result": None}


def add_assistant_placeholder(history, text="..."):
    history = history or []
    history.append({
        "role": "assistant",
        "content": text,
        "metadata": {"title": "Thinking"}
    })
    return history


def update_last_assistant_message(history, text, title=None):
    history = history or []
    if not history or history[-1]["role"] != "assistant":
        msg = {"role": "assistant", "content": text}
        if title:
            msg["metadata"] = {"title": title}
        history.append(msg)
        return history

    history[-1] = {"role": "assistant", "content": text}
    if title:
        history[-1]["metadata"] = {"title": title}
    return history


def user_submit(user_message, chat_ui_history):
    chat_ui_history = chat_ui_history or []
    user_message = (user_message or "").strip()

    if not user_message:
        return "", chat_ui_history

    chat_ui_history.append({"role": "user", "content": user_message})
    return "", chat_ui_history


# -------------------------------
# CORE CHAT
# -------------------------------
def run_chat_turn(user_message: str, memory_state: Dict) -> Dict:
    if memory_state is None:
        memory_state = {"chat_history": [], "last_result": None}

    state_in = {
        "user_query": user_message,
        "chat_history": memory_state["chat_history"],
        "retrieval_attempts": 0,
    }

    try:
        result = graph.invoke(state_in)
    except Exception as e:
        logger.error(f"Graph invocation error: {e}")
        traceback.print_exc()
        result = {
            "final_answer": f"I hit a runtime error while processing the request: {e}",
            "best_score": -1.0,
            "used_context": False,
            "validation_status": "ERROR",
            "retrieved_docs": [],
            "context": "",
            "reasoning_draft": "",
            "retrieval_attempts": 0,
            "retrieval_mode": "error",
        }

    answer = result.get("final_answer", "").strip() or "I could not generate an answer."
    best_score = result.get("best_score", -1.0)
    validation_status = result.get("validation_status", "N/A")
    confidence = compute_confidence(result)

    answer_with_footer = (
        f"{answer}\n\n---\n"
        f"📊 confidence={confidence:.2f} | best_score={best_score:.3f} | validation={validation_status}"
    )

    memory_state["chat_history"].append({"role": "user", "content": user_message})
    memory_state["chat_history"].append({"role": "assistant", "content": answer})
    memory_state["chat_history"] = memory_state["chat_history"][-12:]
    memory_state["last_result"] = result

    return {
        "answer": answer_with_footer,
        "memory_state": memory_state,
        "sources_markdown": format_sources_minimal(result),
        "debug_text": format_debug_text(result),
    }


def bot_respond_stream(chat_ui_history, session_state):
    global vectorstore

    if session_state is None:
        session_state = initialize_session()

    if not chat_ui_history:
        yield (
            chat_ui_history,
            session_state,
            "## Retrieved Sources\n\nNo sources yet.",
            "No debug result yet.",
            ""
        )
        return

    user_message = str(chat_ui_history[-1]["content"]).strip()

    if user_message == "/sources":
        result = session_state.get("last_result")
        chat_ui_history.append({
            "role": "assistant",
            "content": format_sources_minimal(result),
            "metadata": {"title": "Sources"}
        })
        yield (
            chat_ui_history,
            session_state,
            format_sources_minimal(result),
            format_debug_text(result),
            ""
        )
        return

    if user_message == "/debug":
        result = session_state.get("last_result")
        chat_ui_history.append({
            "role": "assistant",
            "content": format_debug_text(result),
            "metadata": {"title": "Debug"}
        })
        yield (
            chat_ui_history,
            session_state,
            format_sources_minimal(result),
            format_debug_text(result),
            ""
        )
        return

    if user_message == "/rebuild":
        if not cfg.allow_rebuild_vectorstore:
            chat_ui_history.append({
                "role": "assistant",
                "content": "Vector store rebuild is disabled on this Space.",
                "metadata": {"title": "Restricted"}
            })
            yield (
                chat_ui_history,
                session_state,
                format_sources_minimal(session_state.get("last_result")),
                format_debug_text(session_state.get("last_result")),
                ""
            )
            return

        chat_ui_history = add_assistant_placeholder(chat_ui_history)
        yield (
            chat_ui_history,
            session_state,
            "",
            "",
            thinking_html("Rebuilding vector store")
        )

        time.sleep(cfg.blink_stage_1)

        chat_ui_history = update_last_assistant_message(
            chat_ui_history,
            "Rebuilding vector store and reloading embeddings...",
            title="Maintenance"
        )
        yield (
            chat_ui_history,
            session_state,
            "",
            "",
            thinking_html("Rebuilding vector store")
        )

        build_vectorstore()
        vectorstore = load_vectorstore()

        chat_ui_history = update_last_assistant_message(
            chat_ui_history,
            "✅ Vector store rebuilt and reloaded.",
            title="Done"
        )
        yield (
            chat_ui_history,
            session_state,
            format_sources_minimal(session_state.get("last_result")),
            format_debug_text(session_state.get("last_result")),
            ""
        )
        return

    chat_ui_history = add_assistant_placeholder(chat_ui_history, text="...")
    yield (
        chat_ui_history,
        session_state,
        "",
        "",
        thinking_html("Starting")
    )
    time.sleep(cfg.blink_stage_1)

    yield (
        chat_ui_history,
        session_state,
        "",
        "",
        thinking_html("Retrieving evidence")
    )
    time.sleep(cfg.blink_stage_2)

    yield (
        chat_ui_history,
        session_state,
        "",
        "",
        thinking_html("Running ECG adapter reasoning")
    )
    time.sleep(cfg.blink_stage_3)

    out = run_chat_turn(user_message, session_state)

    yield (
        chat_ui_history,
        session_state,
        out["sources_markdown"],
        out["debug_text"],
        thinking_html("Generating grounded summary")
    )
    time.sleep(cfg.blink_before_answer)

    if cfg.enable_typewriter_stream:
        for partial in stream_text(out["answer"], step=120):
            chat_ui_history = update_last_assistant_message(
                chat_ui_history,
                partial,
                title="Answer"
            )
            yield (
                chat_ui_history,
                session_state,
                out["sources_markdown"],
                out["debug_text"],
                ""
            )

    chat_ui_history = update_last_assistant_message(
        chat_ui_history,
        out["answer"],
        title="Answer"
    )

    yield (
        chat_ui_history,
        out["memory_state"],
        out["sources_markdown"],
        out["debug_text"],
        ""
    )


def clear_chat():
    return [], initialize_session(), "## Retrieved Sources\n\nNo sources yet.", "No debug result yet.", ""


def rebuild_from_button(session_state, chatbot_history):
    global vectorstore

    if not cfg.allow_rebuild_vectorstore:
        chatbot_history = chatbot_history or []
        chatbot_history.append({
            "role": "assistant",
            "content": "Vector store rebuild is disabled on this Space.",
            "metadata": {"title": "Restricted"}
        })
        return (
            chatbot_history,
            session_state,
            format_sources_minimal(session_state.get("last_result")),
            format_debug_text(session_state.get("last_result")),
            ""
        )

    build_vectorstore()
    vectorstore = load_vectorstore()

    chatbot_history = chatbot_history or []
    chatbot_history.append({
        "role": "assistant",
        "content": "✅ Vector store rebuilt and reloaded.",
        "metadata": {"title": "Done"}
    })

    return (
        chatbot_history,
        session_state,
        format_sources_minimal(session_state.get("last_result")),
        format_debug_text(session_state.get("last_result")),
        ""
    )


# -------------------------------
# APP
# -------------------------------
with gr.Blocks(
    title="Medical CSV RAG Chatbot",
    css=CUSTOM_CSS,
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
        neutral_hue="slate",
        radius_size="lg",
        text_size="md",
    ),
) as demo:

    gr.HTML(hero_html())

    session_state = gr.State(initialize_session())

    with gr.Column(elem_classes=["mobile-stack"]):
        with gr.Group(elem_classes=["panel-wrap"]):
            chatbot = gr.Chatbot(
                label="Clinical Chat",
                height=640,
                elem_id="chatbot",
                type="messages",
                show_copy_button=True,
                bubble_full_width=False,
                avatar_images=(None, None),
            )

            user_box = gr.Textbox(
                label="Ask a medical question",
                placeholder="e.g. What are the ECG findings in hyperkalemia?",
                lines=2,
                autofocus=True,
            )

            status_html = gr.HTML("")

            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear")
                rebuild_btn = gr.Button("Rebuild Store")

            gr.HTML(
                """
                <div class="command-note">
                    Commands: <code>/sources</code>, <code>/debug</code>, <code>/rebuild</code>
                </div>
                """
            )

        with gr.Accordion("Retrieved Sources", open=False):
            with gr.Group(elem_classes=["panel-wrap", "mobile-scroll"]):
                sources_panel = gr.Markdown("## Retrieved Sources\n\nNo sources yet.")

        if cfg.show_debug_panel:
            with gr.Accordion("Debug Panel", open=False):
                with gr.Group(elem_classes=["panel-wrap", "mobile-scroll"]):
                    debug_panel = gr.Textbox(
                        label="Debug",
                        value="No debug result yet.",
                        lines=18,
                        max_lines=28,
                        interactive=False,
                    )
        else:
            debug_panel = gr.Textbox(visible=False, value="")

    submit_event = user_box.submit(
        fn=user_submit,
        inputs=[user_box, chatbot],
        outputs=[user_box, chatbot],
        queue=True,
    )

    submit_event.then(
        fn=bot_respond_stream,
        inputs=[chatbot, session_state],
        outputs=[chatbot, session_state, sources_panel, debug_panel, status_html],
        queue=True,
    )

    send_click = send_btn.click(
        fn=user_submit,
        inputs=[user_box, chatbot],
        outputs=[user_box, chatbot],
        queue=True,
    )

    send_click.then(
        fn=bot_respond_stream,
        inputs=[chatbot, session_state],
        outputs=[chatbot, session_state, sources_panel, debug_panel, status_html],
        queue=True,
    )

    clear_btn.click(
        fn=clear_chat,
        inputs=[],
        outputs=[chatbot, session_state, sources_panel, debug_panel, status_html],
        queue=False,
    )

    rebuild_btn.click(
        fn=rebuild_from_button,
        inputs=[session_state, chatbot],
        outputs=[chatbot, session_state, sources_panel, debug_panel, status_html],
        queue=True,
    )

demo.queue(default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch(
        debug=cfg.launch_debug,
        server_name=cfg.server_name,
        server_port=cfg.server_port,
    )