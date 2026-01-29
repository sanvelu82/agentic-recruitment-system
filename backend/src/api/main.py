"""
FastAPI main application.

This is the entry point for the REST API.
"""

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
import os

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..agents.orchestrator import OrchestratorAgent
from ..schemas.job import JobDescription
from ..schemas.candidates import CandidateProfile
from ..schemas.messages import PipelineStage


# ---------------------
# Pydantic Models (API Layer)
# ---------------------

class JobCreateRequest(BaseModel):
    """Request to create a new job posting."""
    title: str = Field(..., min_length=3, max_length=200)
    company: str = Field(default="Anonymous Company")
    department: str = Field(default="")
    raw_description: str = Field(..., min_length=50)
    location: str = Field(default="Remote")
    employment_type: str = Field(default="full_time")
    experience_years_min: int = Field(default=0, ge=0)
    experience_years_max: int = Field(default=20, le=50)
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "Senior Python Developer",
                "company": "Tech Corp",
                "raw_description": "We are looking for a Senior Python Developer with 5+ years of experience...",
                "location": "Remote",
                "experience_years_min": 5,
                "experience_years_max": 10,
            }
        }


class CandidateCreateRequest(BaseModel):
    """Request to add a candidate."""
    resume_text: str = Field(..., min_length=100)
    resume_format: str = Field(default="txt")
    source: str = Field(default="direct_application")


class PipelineCreateRequest(BaseModel):
    """Request to create and run a pipeline."""
    job_id: str
    candidate_ids: List[str] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)


class TestSubmissionRequest(BaseModel):
    """Candidate's test submission."""
    candidate_id: str
    pipeline_id: str
    responses: List[Dict[str, Any]]  # [{question_id, selected_option, time_seconds}]


class HumanReviewRequest(BaseModel):
    """Human review decision."""
    pipeline_id: str
    approved: bool
    notes: str = Field(default="")
    reviewer: str = Field(default="anonymous")


# ---------------------
# In-Memory Storage (Replace with DB in production)
# ---------------------

jobs_db: Dict[str, JobDescription] = {}
candidates_db: Dict[str, Dict[str, Any]] = {}
pipelines_db: Dict[str, Dict[str, Any]] = {}
orchestrators: Dict[str, OrchestratorAgent] = {}


# ---------------------
# Application Lifecycle
# ---------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    print("🚀 Agentic Recruitment System API starting...")
    yield
    print("👋 Agentic Recruitment System API shutting down...")


# ---------------------
# FastAPI Application
# ---------------------

app = FastAPI(
    title="Agentic Recruitment System",
    description="""
    An AI-powered recruitment pipeline with multiple specialized agents.
    
    ## Features
    - **Multi-Agent System**: 9 specialized agents working together
    - **Explainable AI**: Every decision is explained and auditable
    - **Bias Mitigation**: Built-in fairness checks and bias detection
    - **Human-in-the-Loop**: Decision gates for human review
    
    ## Agents
    1. **JD Analyzer**: Parses job descriptions, detects bias
    2. **Resume Parser**: Extracts structured data from resumes
    3. **Matcher**: Calculates resume-JD similarity
    4. **Shortlister**: Applies threshold-based filtering
    5. **Test Generator**: Creates skill assessments
    6. **Test Evaluator**: Scores candidate responses
    7. **Ranker**: Produces final candidate rankings
    8. **Bias Auditor**: Ensures fairness throughout
    9. **Orchestrator**: Coordinates the entire pipeline
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite & React
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------
# Health Check
# ---------------------

@app.get("/health", tags=["System"])
async def health_check():
    """Check API health."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "groq_configured": bool(os.getenv("GROQ_API_KEY")),
    }


# ---------------------
# Job Endpoints
# ---------------------

@app.post("/api/jobs", tags=["Jobs"])
async def create_job(job: JobCreateRequest):
    """Create a new job posting."""
    from uuid import uuid4
    
    job_id = uuid4().hex[:12]
    job_desc = JobDescription(
        job_id=job_id,
        title=job.title,
        company=job.company,
        department=job.department,
        raw_description=job.raw_description,
        location=job.location,
        employment_type=job.employment_type,
        experience_years_min=job.experience_years_min,
        experience_years_max=job.experience_years_max,
    )
    
    jobs_db[job_id] = job_desc
    
    return {"job_id": job_id, "message": "Job created successfully"}


@app.get("/api/jobs", tags=["Jobs"])
async def list_jobs():
    """List all job postings."""
    return {
        "jobs": [
            {"job_id": jid, "title": j.title, "company": j.company}
            for jid, j in jobs_db.items()
        ]
    }


@app.get("/api/jobs/{job_id}", tags=["Jobs"])
async def get_job(job_id: str):
    """Get a specific job posting."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs_db[job_id].to_dict()


# ---------------------
# Candidate Endpoints
# ---------------------

@app.post("/api/jobs/{job_id}/candidates", tags=["Candidates"])
async def add_candidate(job_id: str, candidate: CandidateCreateRequest):
    """Add a candidate to a job posting."""
    from uuid import uuid4
    
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    candidate_id = uuid4().hex[:12]
    candidates_db[candidate_id] = {
        "candidate_id": candidate_id,
        "job_id": job_id,
        "resume_text": candidate.resume_text,
        "resume_format": candidate.resume_format,
        "source": candidate.source,
        "status": "pending",
    }
    
    return {"candidate_id": candidate_id, "message": "Candidate added successfully"}


@app.get("/api/jobs/{job_id}/candidates", tags=["Candidates"])
async def list_candidates(job_id: str):
    """List all candidates for a job."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_candidates = [
        c for c in candidates_db.values() if c["job_id"] == job_id
    ]
    return {"candidates": job_candidates}


# ---------------------
# Pipeline Endpoints
# ---------------------

@app.post("/api/pipelines", tags=["Pipeline"])
async def create_pipeline(request: PipelineCreateRequest, background_tasks: BackgroundTasks):
    """Create and start a recruitment pipeline."""
    from uuid import uuid4
    
    if request.job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_db[request.job_id]
    
    # Get candidates (either specified or all for this job)
    if request.candidate_ids:
        candidates_data = [
            candidates_db[cid] for cid in request.candidate_ids 
            if cid in candidates_db
        ]
    else:
        candidates_data = [
            c for c in candidates_db.values() if c["job_id"] == request.job_id
        ]
    
    if not candidates_data:
        raise HTTPException(status_code=400, detail="No candidates found for this job")
    
    # Create orchestrator with config
    config = {
        "shortlist_threshold": request.config.get("shortlist_threshold", 0.7),
        "test_questions": request.config.get("test_questions", 10),
        "test_passing_score": request.config.get("test_passing_score", 0.6),
        "top_k_candidates": request.config.get("top_k_candidates", 10),
        "ranking_weights": request.config.get("ranking_weights", {"resume": 0.5, "test": 0.5}),
    }
    
    orchestrator = OrchestratorAgent(config=config)
    
    # Convert to CandidateProfile objects
    candidate_profiles = [
        CandidateProfile(
            candidate_id=c["candidate_id"],
            # Other fields populated during parsing
        )
        for c in candidates_data
    ]
    
    # Create pipeline
    state = orchestrator.create_pipeline(job, candidate_profiles)
    
    # Update candidates in state with resume text
    state.candidates = candidates_data
    
    pipeline_id = state.pipeline_id
    orchestrators[pipeline_id] = orchestrator
    pipelines_db[pipeline_id] = {
        "pipeline_id": pipeline_id,
        "job_id": request.job_id,
        "status": "created",
        "config": config,
    }
    
    return {
        "pipeline_id": pipeline_id,
        "job_id": request.job_id,
        "candidate_count": len(candidates_data),
        "status": "created",
        "message": "Pipeline created. Call /api/pipelines/{id}/run to start.",
    }


@app.post("/api/pipelines/{pipeline_id}/run", tags=["Pipeline"])
async def run_pipeline(pipeline_id: str):
    """Run the recruitment pipeline."""
    if pipeline_id not in orchestrators:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    orchestrator = orchestrators[pipeline_id]
    
    try:
        final_state = orchestrator.run_pipeline()
        
        pipelines_db[pipeline_id]["status"] = final_state.current_stage.value
        pipelines_db[pipeline_id]["state"] = final_state.to_dict()
        
        return {
            "pipeline_id": pipeline_id,
            "status": final_state.current_stage.value,
            "summary": orchestrator.get_pipeline_summary(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")


@app.get("/api/pipelines/{pipeline_id}", tags=["Pipeline"])
async def get_pipeline(pipeline_id: str):
    """Get pipeline status and details."""
    if pipeline_id not in pipelines_db:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    pipeline = pipelines_db[pipeline_id]
    
    if pipeline_id in orchestrators:
        summary = orchestrators[pipeline_id].get_pipeline_summary()
        return {**pipeline, "summary": summary}
    
    return pipeline


@app.get("/api/pipelines/{pipeline_id}/audit", tags=["Pipeline"])
async def get_audit_log(pipeline_id: str):
    """Get the complete audit log for a pipeline."""
    if pipeline_id not in orchestrators:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    return {"audit_log": orchestrators[pipeline_id].get_audit_log()}


# ---------------------
# Test Endpoints
# ---------------------

@app.get("/api/pipelines/{pipeline_id}/test", tags=["Tests"])
async def get_test_questions(pipeline_id: str, candidate_id: str):
    """Get test questions for a candidate (without answers)."""
    if pipeline_id not in orchestrators:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    orchestrator = orchestrators[pipeline_id]
    state = orchestrator.state
    
    if not state or candidate_id not in state.shortlisted_candidates:
        raise HTTPException(status_code=403, detail="Candidate not shortlisted for testing")
    
    # Return questions without correct answers
    questions = []
    for q in state.test_questions:
        questions.append({
            "question_id": q["question_id"],
            "question_text": q["question_text"],
            "options": q["options"],
            "skill_tested": q["skill_tested"],
            "difficulty": q["difficulty"],
            # Exclude: correct_option, explanation
        })
    
    return {"test_id": f"test_{state.job_id}", "questions": questions}


@app.post("/api/pipelines/{pipeline_id}/test/submit", tags=["Tests"])
async def submit_test(pipeline_id: str, submission: TestSubmissionRequest):
    """Submit test responses for a candidate."""
    if pipeline_id not in orchestrators:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    orchestrator = orchestrators[pipeline_id]
    state = orchestrator.state
    
    if submission.candidate_id not in state.shortlisted_candidates:
        raise HTTPException(status_code=403, detail="Candidate not authorized to submit")
    
    # Store responses
    state.candidate_test_responses[submission.candidate_id] = submission.responses
    
    return {
        "message": "Test submitted successfully",
        "candidate_id": submission.candidate_id,
        "questions_answered": len(submission.responses),
    }


# ---------------------
# Human Review Endpoints
# ---------------------

@app.post("/api/pipelines/{pipeline_id}/review", tags=["Human Review"])
async def submit_human_review(review: HumanReviewRequest):
    """Submit human review decision for a paused pipeline."""
    if review.pipeline_id not in orchestrators:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    orchestrator = orchestrators[review.pipeline_id]
    state = orchestrator.state
    
    if state.current_stage != PipelineStage.AWAITING_HUMAN_REVIEW:
        raise HTTPException(
            status_code=400, 
            detail=f"Pipeline not awaiting review. Current stage: {state.current_stage.value}"
        )
    
    state.human_review_notes.append(
        f"[{review.reviewer}]: {'Approved' if review.approved else 'Rejected'} - {review.notes}"
    )
    
    if review.approved:
        # Resume pipeline from last stage
        # In a full implementation, would track which stage to resume from
        return {"message": "Review approved. Call /api/pipelines/{id}/run to continue."}
    else:
        state.current_stage = PipelineStage.FAILED
        state.errors.append(f"Rejected by human review: {review.notes}")
        return {"message": "Pipeline rejected by human review."}


# ---------------------
# Results Endpoints
# ---------------------

@app.get("/api/pipelines/{pipeline_id}/results", tags=["Results"])
async def get_pipeline_results(pipeline_id: str):
    """Get final pipeline results including rankings."""
    if pipeline_id not in orchestrators:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    orchestrator = orchestrators[pipeline_id]
    state = orchestrator.state
    
    if state.current_stage not in [PipelineStage.COMPLETED, PipelineStage.BIAS_AUDIT]:
        raise HTTPException(
            status_code=400,
            detail=f"Pipeline not complete. Current stage: {state.current_stage.value}"
        )
    
    return {
        "pipeline_id": pipeline_id,
        "job_id": state.job_id,
        "final_rankings": state.final_rankings,
        "bias_audit": state.bias_audit_results,
        "total_candidates": len(state.candidates),
        "shortlisted": len(state.shortlisted_candidates),
        "tested": len(state.test_results),
        "ranked": len(state.final_rankings),
    }


@app.get("/api/pipelines/{pipeline_id}/candidate/{candidate_id}", tags=["Results"])
async def get_candidate_result(pipeline_id: str, candidate_id: str):
    """Get detailed results for a specific candidate."""
    if pipeline_id not in orchestrators:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    orchestrator = orchestrators[pipeline_id]
    state = orchestrator.state
    
    # Find candidate data
    candidate = next(
        (c for c in state.candidates if c.get("candidate_id") == candidate_id),
        None
    )
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    
    # Find match result
    match_result = next(
        (m for m in state.match_results if m.get("candidate_id") == candidate_id),
        None
    )
    
    # Find test result
    test_result = next(
        (t for t in state.test_results if t.get("candidate_id") == candidate_id),
        None
    )
    
    # Find ranking
    ranking = next(
        (r for r in state.final_rankings if r.get("candidate_id") == candidate_id),
        None
    )
    
    return {
        "candidate_id": candidate_id,
        "parsed_resume": candidate.get("parsed_resume", {}),
        "match_result": match_result,
        "test_result": test_result,
        "ranking": ranking,
        "shortlisted": candidate_id in state.shortlisted_candidates,
    }


# ---------------------
# Run directly for development
# ---------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
