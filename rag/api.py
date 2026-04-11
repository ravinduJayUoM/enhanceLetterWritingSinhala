"""
api.py — FastAPI application for the Sinhala Letter RAG system.

This file owns:
  - App and CORS setup
  - Pydantic request/response models
  - Route definitions (thin handlers that delegate to Pipeline and LetterDatabase)
  - Application startup (component wiring)

Business logic lives entirely in the pipeline steps and knowledge_base modules.
"""

import os
import sys
import pandas as pd
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

import auth as _auth

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class UserQuery(BaseModel):
    prompt: str
    missing_info: Optional[Dict[str, str]] = None


class LetterRequest(BaseModel):
    enhanced_prompt: str
    sender_info: Optional[Dict[str, str]] = None


class RegisterRequest(BaseModel):
    username: str
    password: str
    full_name: str
    title: Optional[str] = ""
    address_line1: Optional[str] = ""
    address_line2: Optional[str] = ""
    phone: Optional[str] = ""


class LoginRequest(BaseModel):
    username: str
    password: str


class ProfileUpdateRequest(BaseModel):
    full_name: str
    title: Optional[str] = ""
    address_line1: Optional[str] = ""
    address_line2: Optional[str] = ""
    phone: Optional[str] = ""


class KnowledgeBaseEntry(BaseModel):
    content: str
    title: str
    letter_category: str = "general"
    doc_type: str = "example"
    register: str = "formal"
    tags: Optional[str] = ""
    original_prompt: Optional[str] = None
    rating: Optional[float] = None
    source: str = "user_generated"


class LetterRatingRequest(BaseModel):
    letter_content: str
    original_prompt: str
    letter_category: str = "general"
    quality_overall: int    # 1–5
    quality_match: int      # 1–5
    quality_language: int   # 1–5
    quality_structure: int  # 1–5
    comments: Optional[str] = ""


class SystemFeedbackRequest(BaseModel):
    ease_of_use: int           # 1–5
    ease_of_describing: int    # 1–5
    gap_questions_helpful: int # 1–5
    confidence_in_output: int  # 1–5
    would_use_again: int       # 1–5
    liked_most: Optional[str] = ""
    needs_improvement: Optional[str] = ""
    issues_faced: Optional[str] = ""


# Admin credentials (hardcoded)
ADMIN_USERNAME = "admin@sinhalalipi.lk"
ADMIN_PASSWORD = "Admin@123"


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Sinhala Letter RAG System")

_auth.init_db()
_auth.init_feedback_db()

_bearer = HTTPBearer()


def _current_user(credentials: HTTPAuthorizationCredentials = Depends(_bearer)) -> dict:
    """FastAPI dependency — validates JWT and returns the user dict."""
    username = _auth.decode_token(credentials.credentials)
    if not username:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    user = _auth.get_user_by_username(username)
    if not user:
        raise HTTPException(status_code=401, detail="User not found.")
    return user


class AdminLoginRequest(BaseModel):
    username: str
    password: str


def _require_admin(credentials: HTTPAuthorizationCredentials = Depends(_bearer)):
    """FastAPI dependency — validates admin token (simple base64-encoded credentials check)."""
    import base64
    try:
        decoded = base64.b64decode(credentials.credentials).decode()
        username, password = decoded.split(":", 1)
        if username != ADMIN_USERNAME or password != ADMIN_PASSWORD:
            raise HTTPException(status_code=403, detail="Forbidden.")
    except Exception:
        raise HTTPException(status_code=403, detail="Forbidden.")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Component instances (populated in startup)
# ---------------------------------------------------------------------------

from config import get_config
from knowledge_base.letter_database import LetterDatabase
from knowledge_base.llm_factory import get_llm
from steps.extractor import InfoExtractor
from steps.gap_filler import GapFiller
from steps.retriever import Retriever
from steps.prompt_builder import PromptBuilder
from steps.generator import LetterGenerator
from pipeline import Pipeline

_config = get_config()
letter_db = LetterDatabase()
_pipeline: Optional[Pipeline] = None


@app.on_event("startup")
async def startup_event():
    global _pipeline
    try:
        letter_db.build_knowledge_base()
        from config import LLMProvider
        extraction_provider = _config.llm.extraction_provider or _config.llm.provider
        generation_provider = _config.llm.generation_provider or _config.llm.provider
        extraction_llm = get_llm(temperature=0.1, provider=extraction_provider)
        generation_llm = get_llm(temperature=0.3, provider=generation_provider)
        print(f"[API] Extraction LLM: {extraction_provider.value} | Generation LLM: {generation_provider.value}")
        _pipeline = Pipeline(
            extractor=InfoExtractor(llm=extraction_llm),
            gap_filler=GapFiller(),
            retriever=Retriever(letter_database=letter_db),
            prompt_builder=PromptBuilder(),
            generator=LetterGenerator(llm=generation_llm),
        )
        print("[API] Pipeline initialised successfully.")
    except Exception as exc:
        print(f"[API] Pipeline initialisation failed: {exc}")
        _pipeline = None


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _require_pipeline() -> Pipeline:
    if _pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialised.")
    return _pipeline


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------

@app.post("/auth/register", status_code=201)
def register(req: RegisterRequest):
    if _auth.get_user_by_username(req.username):
        raise HTTPException(status_code=400, detail="Username already taken.")
    user = _auth.create_user(
        username=req.username,
        password=req.password,
        full_name=req.full_name,
        title=req.title or "",
        address_line1=req.address_line1 or "",
        address_line2=req.address_line2 or "",
        phone=req.phone or "",
    )
    return {"message": "Account created successfully.", "username": user["username"]}


@app.post("/auth/login")
def login(req: LoginRequest):
    user = _auth.authenticate_user(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    token = _auth.create_access_token(user["username"])
    return {"access_token": token, "token_type": "bearer"}


@app.get("/auth/me")
def get_me(user: dict = Depends(_current_user)):
    return {k: v for k, v in user.items() if k != "hashed_password"}


@app.put("/auth/me")
def update_me(req: ProfileUpdateRequest, user: dict = Depends(_current_user)):
    updated = _auth.update_user_profile(
        username=user["username"],
        full_name=req.full_name,
        title=req.title or "",
        address_line1=req.address_line1 or "",
        address_line2=req.address_line2 or "",
        phone=req.phone or "",
    )
    return {k: v for k, v in updated.items() if k != "hashed_password"}


@app.get("/")
async def root():
    return {
        "status": "Sinhala Letter RAG System is running",
        "pipeline_available": _pipeline is not None,
        "knowledge_base_available": letter_db.db is not None,
    }


@app.post("/extract/")
async def extract_info(query: UserQuery):
    """Step 1 only — extract structured metadata from a prompt."""
    pipeline = _require_pipeline()
    try:
        extracted = pipeline.extractor.extract(query.prompt)
        return {"prompt": query.prompt, "extracted_info": extracted}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/process_query/")
async def process_query(query: UserQuery, user: dict = Depends(_current_user)):
    """Run Steps 1–4: extract → gap-fill → retrieve → build prompt."""
    pipeline = _require_pipeline()
    try:
        return pipeline.process(query.prompt, query.missing_info)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/generate_letter/")
def generate_letter(request: LetterRequest, user: dict = Depends(_current_user)):
    """Step 5 — generate the Sinhala letter from an enhanced prompt."""
    pipeline = _require_pipeline()
    try:
        letter = pipeline.generate_letter(request.enhanced_prompt, request.sender_info)
        return {"generated_letter": letter}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Letter generation failed: {exc}")


@app.post("/feedback/letter/")
async def submit_letter_rating(req: LetterRatingRequest, user: dict = Depends(_current_user)):
    """Save structured per-letter rating.

    If quality_overall >= 4 the letter is also added to the FAISS index
    so it improves future retrievals immediately.
    """
    for field, val in [
        ("quality_overall", req.quality_overall),
        ("quality_match", req.quality_match),
        ("quality_language", req.quality_language),
        ("quality_structure", req.quality_structure),
    ]:
        if not (1 <= val <= 5):
            raise HTTPException(status_code=400, detail=f"{field} must be between 1 and 5.")

    rating_id = _auth.save_letter_rating(
        user_id=user["id"],
        username=user["username"],
        original_prompt=req.original_prompt,
        letter_content=req.letter_content,
        letter_category=req.letter_category,
        quality_overall=req.quality_overall,
        quality_match=req.quality_match,
        quality_language=req.quality_language,
        quality_structure=req.quality_structure,
        comments=req.comments or "",
    )

    result: dict = {"rating_saved": True, "rating_id": rating_id, "added_to_index": False}

    if req.quality_overall >= 4:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        prefix_map = {
            "request": "REQ", "apology": "APO", "invitation": "INV",
            "complaint": "CMP", "application": "APP", "general": "GEN",
            "notification": "NOT", "appreciation": "APR",
        }
        category = req.letter_category.lower()
        prefix = prefix_map.get(category, "GEN")
        new_id = f"UG_{prefix}_{timestamp}"
        title = f"{req.letter_category} ලිපිය"
        metadata = {
            "id": new_id,
            "title": title,
            "letter_category": category,
            "doc_type": "example",
            "register": "formal",
            "source": "user_generated",
            "tags": req.original_prompt[:150],
        }
        try:
            index_size = letter_db.add_document(f"{title}\n\n{req.letter_content}", metadata)
            result["added_to_index"] = True
            result["id"] = new_id
            result["index_size"] = index_size
            print(f"[API] High-rated letter (overall={req.quality_overall}) added to FAISS: {new_id}")
        except Exception as exc:
            print(f"[API] FAISS add failed after rating: {exc}")
            result["add_error"] = str(exc)

    return result


@app.post("/feedback/system/")
async def submit_system_feedback(req: SystemFeedbackRequest, user: dict = Depends(_current_user)):
    """Save system usability survey response."""
    for field, val in [
        ("ease_of_use", req.ease_of_use),
        ("ease_of_describing", req.ease_of_describing),
        ("gap_questions_helpful", req.gap_questions_helpful),
        ("confidence_in_output", req.confidence_in_output),
        ("would_use_again", req.would_use_again),
    ]:
        if not (1 <= val <= 5):
            raise HTTPException(status_code=400, detail=f"{field} must be between 1 and 5.")

    feedback_id = _auth.save_system_feedback(
        user_id=user["id"],
        username=user["username"],
        ease_of_use=req.ease_of_use,
        ease_of_describing=req.ease_of_describing,
        gap_questions_helpful=req.gap_questions_helpful,
        confidence_in_output=req.confidence_in_output,
        would_use_again=req.would_use_again,
        liked_most=req.liked_most or "",
        needs_improvement=req.needs_improvement or "",
        issues_faced=req.issues_faced or "",
    )
    return {"feedback_saved": True, "feedback_id": feedback_id}


# ---------------------------------------------------------------------------
# Admin routes
# ---------------------------------------------------------------------------

@app.post("/admin/login")
def admin_login(req: AdminLoginRequest):
    """Returns a base64 admin token if credentials match."""
    import base64
    if req.username != ADMIN_USERNAME or req.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid admin credentials.")
    token = base64.b64encode(f"{req.username}:{req.password}".encode()).decode()
    return {"access_token": token, "token_type": "bearer"}


@app.get("/admin/stats")
def admin_stats(_=Depends(_require_admin)):
    return _auth.get_feedback_stats()


@app.get("/admin/letter-ratings")
def admin_letter_ratings(_=Depends(_require_admin)):
    return {"data": _auth.get_all_letter_ratings()}


@app.get("/admin/system-feedback")
def admin_system_feedback(_=Depends(_require_admin)):
    return {"data": _auth.get_all_system_feedback()}


@app.get("/admin/export/letter-ratings")
def admin_export_letter_ratings(_=Depends(_require_admin)):
    """Return letter ratings as CSV."""
    import io, csv
    from fastapi.responses import StreamingResponse
    rows = _auth.get_all_letter_ratings()
    if not rows:
        return {"data": []}
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=letter_ratings.csv"},
    )


@app.get("/admin/export/system-feedback")
def admin_export_system_feedback(_=Depends(_require_admin)):
    """Return system feedback as CSV."""
    import io, csv
    from fastapi.responses import StreamingResponse
    rows = _auth.get_all_system_feedback()
    if not rows:
        return {"data": []}
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=system_feedback.csv"},
    )


@app.get("/search/")
async def search_kb(query: str = Query(...), top_k: int = Query(3)):
    """Direct vector-store search (bypasses the full pipeline)."""
    if letter_db.db is None:
        raise HTTPException(status_code=500, detail="Knowledge base not initialised.")
    try:
        results = letter_db.search(query, top_k=top_k)
        return {
            "query": query,
            "result_count": len(results),
            "results": [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in results
            ],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/add_to_knowledge_base/")
async def add_to_knowledge_base(entry: KnowledgeBaseEntry):
    """Append a new letter to the CSV dataset.

    The FAISS index is NOT updated automatically — call /rebuild_knowledge_base/ afterwards.
    """
    import filelock
    from datetime import datetime

    try:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        prefix_map = {
            "request": "REQ", "apology": "APO", "invitation": "INV",
            "complaint": "CMP", "application": "APP", "general": "GEN",
            "notification": "NOT", "appreciation": "APR",
        }
        prefix = prefix_map.get(entry.letter_category.lower(), "GEN")
        new_id = f"{prefix}_{timestamp}"

        new_row = {
            "id": new_id,
            "letter_category": entry.letter_category,
            "doc_type": entry.doc_type,
            "register": entry.register,
            "language": "si",
            "source": entry.source,
            "title": entry.title,
            "content": entry.content,
            "tags": entry.tags or "",
            "rating": entry.rating,
        }

        csv_path = _config.data.csv_path
        lock = filelock.FileLock(csv_path + ".lock", timeout=10)
        with lock:
            new_df = pd.DataFrame([new_row])
            if os.path.exists(csv_path):
                new_df.to_csv(csv_path, mode="a", header=False, index=False)
            else:
                new_df.to_csv(csv_path, mode="w", header=True, index=False)

        return {
            "status": "success",
            "id": new_id,
            "note": "Call /rebuild_knowledge_base/ to include this entry in vector search.",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/rebuild_knowledge_base/")
async def rebuild_knowledge_base():
    """Force-rebuild the FAISS index from the current CSV."""
    global _pipeline
    try:
        stats = letter_db.rebuild_knowledge_base()
        # Re-wire pipeline with the fresh DB
        llm = get_llm(temperature=0.1)
        _pipeline = Pipeline(
            extractor=InfoExtractor(llm=llm),
            gap_filler=GapFiller(),
            retriever=Retriever(letter_database=letter_db),
            prompt_builder=PromptBuilder(),
            generator=LetterGenerator(llm=llm),
        )
        return {"status": "success", **stats}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/config/")
async def get_current_config():
    """Return the active configuration settings."""
    provider = _config.llm.provider
    model_map = {
        "gemini":       _config.llm.gemini_model,
        "ollama":       _config.llm.ollama_model,
        "huggingface":  _config.llm.huggingface_model,
        "azure_openai": _config.llm.azure_deployment_name or "(from env)",
        "openai":       _config.llm.openai_model,
    }
    return {
        "experiment_name": _config.experiment_name,
        "retrieval": {
            "use_sinhala_query_builder": _config.retrieval.use_sinhala_query_builder,
            "use_reranker": _config.retrieval.use_reranker,
            "initial_retrieval_k": _config.retrieval.initial_retrieval_k,
            "final_top_k": _config.retrieval.final_top_k,
        },
        "embedding_model": _config.embedding.model_name,
        "llm_provider": provider.value,
        "llm_model": model_map.get(provider.value, "unknown"),
        "data_path": _config.data.csv_path,
    }


@app.get("/diagnostics/")
async def run_diagnostics():
    """Summarise the state of the knowledge base and run a quick smoke-test search."""
    if letter_db.db is None:
        raise HTTPException(status_code=500, detail="Knowledge base not initialised.")
    try:
        test_results = letter_db.search("application", top_k=1)
        return {
            "status": "ok",
            "document_count": letter_db.document_count(),
            "embedding_model": _config.embedding.model_name,
            "sample_documents": letter_db.sample_documents(3),
            "smoke_test": {
                "query": "application",
                "results_found": bool(test_results),
                "first_result_preview": test_results[0].page_content[:200] if test_results else None,
            },
            "data_source": {
                "csv_path": _config.data.csv_path,
                "csv_exists": os.path.exists(_config.data.csv_path),
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
