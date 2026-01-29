"""
Job Description schemas for the recruitment pipeline.

These schemas define the structure for job postings and
their parsed representations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4


@dataclass
class SkillRequirement:
    """
    A skill requirement from a job description.
    """
    skill_name: str = ""
    category: str = ""  # technical, soft, domain
    importance: str = "required"  # required, preferred, nice_to_have
    minimum_proficiency: str = ""  # beginner, intermediate, advanced, expert
    years_experience: Optional[int] = None
    context: str = ""  # How the skill will be used

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "category": self.category,
            "importance": self.importance,
            "minimum_proficiency": self.minimum_proficiency,
            "years_experience": self.years_experience,
            "context": self.context,
        }


@dataclass
class ExperienceRequirement:
    """Experience requirements from a job description."""
    minimum_years: int = 0
    preferred_years: Optional[int] = None
    relevant_domains: List[str] = field(default_factory=list)
    specific_roles: List[str] = field(default_factory=list)  # e.g., "team lead", "architect"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "minimum_years": self.minimum_years,
            "preferred_years": self.preferred_years,
            "relevant_domains": self.relevant_domains,
            "specific_roles": self.specific_roles,
        }


@dataclass
class EducationRequirement:
    """Education requirements from a job description."""
    minimum_degree: str = ""  # high_school, bachelors, masters, phd
    preferred_degree: Optional[str] = None
    accepted_fields: List[str] = field(default_factory=list)
    certifications_required: List[str] = field(default_factory=list)
    certifications_preferred: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "minimum_degree": self.minimum_degree,
            "preferred_degree": self.preferred_degree,
            "accepted_fields": self.accepted_fields,
            "certifications_required": self.certifications_required,
            "certifications_preferred": self.certifications_preferred,
        }


@dataclass
class JobDescription:
    """
    Raw job description input.
    """
    job_id: str = field(default_factory=lambda: uuid4().hex)
    title: str = ""
    company: str = ""  # Company name (may be anonymized)
    department: str = ""
    location: str = ""
    employment_type: str = ""  # full_time, part_time, contract
    raw_description: str = ""
    experience_years_min: int = 0
    experience_years_max: int = 20
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""  # Hiring manager ID
    
    # Hiring parameters
    positions_available: int = 1
    target_start_date: Optional[datetime] = None
    
    # Evaluation parameters
    shortlist_threshold: float = 0.7  # Minimum match score to shortlist
    top_k_candidates: int = 10  # How many finalists to select

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "title": self.title,
            "company": self.company,
            "department": self.department,
            "location": self.location,
            "employment_type": self.employment_type,
            "raw_description": self.raw_description,
            "experience_years_min": self.experience_years_min,
            "experience_years_max": self.experience_years_max,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "positions_available": self.positions_available,
            "target_start_date": self.target_start_date.isoformat() if self.target_start_date else None,
            "shortlist_threshold": self.shortlist_threshold,
            "top_k_candidates": self.top_k_candidates,
        }


@dataclass
class ParsedJD:
    """
    Structured representation of a parsed job description.
    
    This is what the JD Analyzer agent produces from raw job descriptions.
    """
    job_id: str = ""
    parsing_timestamp: datetime = field(default_factory=datetime.utcnow)
    parsing_confidence: float = 0.0
    
    # Core information
    job_title_normalized: str = ""  # Standardized job title
    seniority_level: str = ""  # entry, mid, senior, lead, principal, executive
    job_function: str = ""  # engineering, data_science, product, etc.
    
    # Requirements
    skills: List[SkillRequirement] = field(default_factory=list)
    experience_requirements: ExperienceRequirement = field(default_factory=ExperienceRequirement)
    education_requirements: EducationRequirement = field(default_factory=EducationRequirement)
    
    # Responsibilities
    key_responsibilities: List[str] = field(default_factory=list)
    
    # For test generation
    technical_topics: List[str] = field(default_factory=list)  # Topics to test
    difficulty_level: str = ""  # beginner, intermediate, advanced
    
    # Scoring weights (for transparency)
    scoring_weights: Dict[str, float] = field(default_factory=lambda: {
        "skills": 0.4,
        "experience": 0.35,
        "education": 0.25,
    })
    
    # Quality and bias checks
    jd_quality_score: float = 0.0
    potential_bias_flags: List[str] = field(default_factory=list)  # e.g., gendered language
    parsing_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "parsing_timestamp": self.parsing_timestamp.isoformat(),
            "parsing_confidence": self.parsing_confidence,
            "job_title_normalized": self.job_title_normalized,
            "seniority_level": self.seniority_level,
            "job_function": self.job_function,
            "skills": [s.to_dict() for s in self.skills],
            "experience_requirements": self.experience_requirements.to_dict(),
            "education_requirements": self.education_requirements.to_dict(),
            "key_responsibilities": self.key_responsibilities,
            "technical_topics": self.technical_topics,
            "difficulty_level": self.difficulty_level,
            "scoring_weights": self.scoring_weights,
            "jd_quality_score": self.jd_quality_score,
            "potential_bias_flags": self.potential_bias_flags,
            "parsing_warnings": self.parsing_warnings,
        }

    def get_required_skills(self) -> List[SkillRequirement]:
        """Get only required skills."""
        return [s for s in self.skills if s.importance == "required"]

    def get_preferred_skills(self) -> List[SkillRequirement]:
        """Get only preferred skills."""
        return [s for s in self.skills if s.importance == "preferred"]


@dataclass 
class TestQuestion:
    """
    A single MCQ test question generated for a job.
    """
    question_id: str = field(default_factory=lambda: uuid4().hex)
    job_id: str = ""
    
    # Question content
    question_text: str = ""
    options: Dict[str, str] = field(default_factory=dict)  # {"A": "...", "B": "...", ...}
    correct_option: str = ""
    explanation: str = ""  # Why this answer is correct
    
    # Metadata
    skill_tested: str = ""
    topic: str = ""
    difficulty: str = ""  # easy, medium, hard
    time_limit_seconds: int = 60
    
    # Scoring
    points: float = 1.0
    partial_credit_allowed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "job_id": self.job_id,
            "question_text": self.question_text,
            "options": self.options,
            "correct_option": self.correct_option,
            "explanation": self.explanation,
            "skill_tested": self.skill_tested,
            "topic": self.topic,
            "difficulty": self.difficulty,
            "time_limit_seconds": self.time_limit_seconds,
            "points": self.points,
            "partial_credit_allowed": self.partial_credit_allowed,
        }
