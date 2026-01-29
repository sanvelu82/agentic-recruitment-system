"""
Resume Parser Agent

Responsibility: Extract structured information from raw resumes.
Single purpose: Parse resumes into a standardized format.

This agent does NOT score or evaluate - only extracts and structures.
"""

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base import BaseAgent
from ..schemas.candidates import (
    ParsedResume,
    SkillExtraction,
    ExperienceEntry,
    EducationEntry,
)

# LangChain imports for LLM integration (using Groq)
try:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


# Common skill categories for classification
TECHNICAL_SKILLS = {
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "ruby",
    "sql", "nosql", "mongodb", "postgresql", "mysql", "redis", "elasticsearch",
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible",
    "react", "angular", "vue", "node.js", "django", "flask", "fastapi", "spring",
    "machine learning", "deep learning", "nlp", "computer vision", "tensorflow", "pytorch",
    "git", "ci/cd", "jenkins", "github actions", "linux", "bash", "rest api", "graphql",
}

SOFT_SKILLS = {
    "leadership", "communication", "teamwork", "problem solving", "analytical",
    "project management", "agile", "scrum", "mentoring", "presentation",
    "time management", "critical thinking", "collaboration", "adaptability",
}

# Proficiency indicators in resume text
PROFICIENCY_INDICATORS = {
    "expert": ["expert", "mastery", "extensive experience", "10+ years", "lead", "architect"],
    "advanced": ["advanced", "proficient", "5+ years", "senior", "strong"],
    "intermediate": ["intermediate", "working knowledge", "2+ years", "familiar"],
    "beginner": ["beginner", "basic", "learning", "exposure", "foundational"],
}


class ResumeParserAgent(BaseAgent[Dict[str, Any], ParsedResume]):
    """
    Extracts structured information from candidate resumes.
    
    Input: Raw resume content (text or file path)
    Output: ParsedResume with skills, experience, education
    
    Key responsibilities:
    - Extract skills with proficiency levels
    - Parse work experience entries
    - Parse education history
    - Anonymize personal information
    - Flag parsing quality issues
    
    Does NOT:
    - Score or rank candidates
    - Make hiring decisions
    - Compare to job descriptions
    """
    
    @property
    def description(self) -> str:
        return (
            "Parses raw resumes into structured data. Extracts skills, "
            "experience, and education while anonymizing personal information."
        )
    
    @property
    def required_confidence_threshold(self) -> float:
        return 0.6  # Lower threshold as parsing can handle some ambiguity
    
    def __init__(self, agent_id: Optional[str] = None, model: str = "llama-3.3-70b-versatile"):
        super().__init__(agent_id)
        self.model_name = model
        self._llm = None
    
    def _get_llm(self):
        """Lazy initialization of Groq LLM client."""
        if self._llm is None and LANGCHAIN_AVAILABLE:
            api_key = os.getenv("GROQ_API_KEY")
            if api_key:
                self._llm = ChatGroq(
                    model=self.model_name,
                    temperature=0.1,
                    api_key=api_key,
                )
        return self._llm
    
    def run(
        self,
        input_data: Dict[str, Any],
        state: Optional["PipelineState"] = None
    ) -> "AgentResult[ParsedResume]":
        """
        Execute resume parsing and return result.
        
        Args:
            input_data: Dict with candidate_id, resume_text, resume_format
            state: Optional pipeline state (created if not provided)
        
        Returns:
            AgentResult with ParsedResume and updated state
        """
        from ..schemas.messages import PipelineState
        from .base import AgentResult, AgentResponse, AgentStatus
        
        # Create default state if not provided
        if state is None:
            state = PipelineState()
        
        try:
            parsed, confidence, explanation = self._process(input_data)
            
            response = AgentResponse(
                agent_name=self.name,
                status=AgentStatus.SUCCESS,
                output=parsed,
                confidence_score=confidence,
                explanation=explanation,
            )
            
            return AgentResult(response=response, state=state)
            
        except Exception as e:
            response = AgentResponse(
                agent_name=self.name,
                status=AgentStatus.FAILURE,
                output=None,
                confidence_score=0.0,
                explanation=f"Resume parsing failed: {str(e)}",
            )
            return AgentResult(response=response, state=state)

    def _process(
        self, 
        input_data: Dict[str, Any]
    ) -> tuple[ParsedResume, float, str]:
        """
        Parse a resume into structured format using LLM.
        
        Args:
            input_data: {
                "candidate_id": str,
                "resume_text": str,  # Raw text content
                "resume_format": str  # pdf, docx, txt
            }
        
        Returns:
            ParsedResume, confidence_score, explanation
        """
        self.log_reasoning("Starting resume parsing")
        
        candidate_id = input_data.get("candidate_id", "")
        resume_text = input_data.get("resume_text", "")
        
        if not resume_text:
            raise ValueError("No resume text provided")
        
        self.log_reasoning(f"Processing resume for candidate {candidate_id[:8]}...")
        self.log_reasoning(f"Resume length: {len(resume_text)} characters")
        
        # Try LLM extraction first, fallback to rule-based
        llm = self._get_llm()
        parsing_warnings = []
        
        if llm and LANGCHAIN_AVAILABLE:
            try:
                extracted_data = self._extract_with_llm(resume_text, llm)
                self.log_reasoning("LLM extraction successful")
            except Exception as e:
                self.log_reasoning(f"LLM extraction failed: {str(e)}, falling back to rule-based")
                extracted_data = self._extract_with_rules(resume_text)
                parsing_warnings.append(f"LLM extraction failed, used rule-based fallback: {str(e)}")
        else:
            self.log_reasoning("LLM not available, using rule-based extraction")
            extracted_data = self._extract_with_rules(resume_text)
            parsing_warnings.append("LLM not configured, used rule-based extraction")
        
        # Build skill extractions
        skills = []
        for skill_data in extracted_data.get("skills", []):
            skill = SkillExtraction(
                skill_name=skill_data.get("name", ""),
                category=skill_data.get("category", "technical"),
                proficiency_level=skill_data.get("proficiency", "intermediate"),
                evidence=skill_data.get("evidence", ""),
                confidence=skill_data.get("confidence", 0.7),
            )
            skills.append(skill)
        
        # Build experience entries
        experience = []
        total_months = 0
        for exp_data in extracted_data.get("experience", []):
            months = exp_data.get("duration_months", 0)
            total_months += months
            exp = ExperienceEntry(
                company_anonymized=f"Company_{len(experience) + 1}",  # Anonymize
                role=exp_data.get("role", ""),
                duration_months=months,
                responsibilities=exp_data.get("responsibilities", []),
                achievements=exp_data.get("achievements", []),
                skills_used=exp_data.get("skills_used", []),
            )
            experience.append(exp)
        
        # Build education entries
        education = []
        for edu_data in extracted_data.get("education", []):
            edu = EducationEntry(
                degree=edu_data.get("degree", ""),
                field_of_study=edu_data.get("field", ""),
                institution_anonymized=f"Institution_{len(education) + 1}",  # Anonymize
                graduation_year=edu_data.get("graduation_year"),
                gpa=edu_data.get("gpa"),
                relevant_coursework=edu_data.get("coursework", []),
            )
            education.append(edu)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(extracted_data, resume_text)
        
        # Build the parsed resume
        parsed = ParsedResume(
            candidate_id=candidate_id,
            parsing_confidence=extracted_data.get("confidence", 0.7),
            professional_summary=extracted_data.get("summary", ""),
            skills=skills,
            experience=experience,
            education=education,
            certifications=extracted_data.get("certifications", []),
            projects=extracted_data.get("projects", []),
            total_experience_months=total_months,
            unique_skills_count=len(skills),
            resume_quality_score=quality_score,
            parsing_warnings=parsing_warnings,
        )
        
        confidence = extracted_data.get("confidence", 0.7)
        explanation = (
            f"Parsed resume for candidate {candidate_id[:8]}. "
            f"Extracted {len(skills)} skills, {len(experience)} experience entries, "
            f"and {len(education)} education entries. "
            f"Total experience: {total_months // 12} years {total_months % 12} months. "
            f"Quality score: {quality_score:.2f}."
        )
        
        self.log_reasoning(f"Resume parsing completed: {explanation}")
        
        return parsed, confidence, explanation
    
    def _extract_with_llm(self, resume_text: str, llm) -> Dict[str, Any]:
        """Extract structured data from resume using LLM."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert resume parser. Extract structured information from the resume.
            
IMPORTANT: 
- Be thorough but accurate
- Estimate experience duration in months
- Classify skills as 'technical', 'soft', or 'domain'
- Infer proficiency from context (expert/advanced/intermediate/beginner)
- DO NOT include any personal identifying information (names, emails, phone numbers, addresses)

Return a JSON object with this exact structure:
{{
    "summary": "Professional summary (2-3 sentences)",
    "skills": [
        {{"name": "skill name", "category": "technical|soft|domain", "proficiency": "expert|advanced|intermediate|beginner", "evidence": "quote from resume", "confidence": 0.0-1.0}}
    ],
    "experience": [
        {{"role": "job title", "duration_months": 24, "responsibilities": ["..."], "achievements": ["..."], "skills_used": ["..."]}}
    ],
    "education": [
        {{"degree": "Bachelor's/Master's/PhD", "field": "Computer Science", "graduation_year": 2020, "gpa": 3.5, "coursework": ["..."]}}
    ],
    "certifications": ["AWS Certified", "..."],
    "projects": [{{"name": "...", "description": "...", "technologies": ["..."]}}],
    "confidence": 0.0-1.0
}}"""),
            ("human", "Parse this resume:\n\n{resume_text}")
        ])
        
        parser = JsonOutputParser()
        chain = prompt | llm | parser
        
        result = chain.invoke({"resume_text": resume_text[:8000]})  # Limit to 8k chars
        return result
    
    def _extract_with_rules(self, resume_text: str) -> Dict[str, Any]:
        """Fallback rule-based extraction when LLM is unavailable."""
        text_lower = resume_text.lower()
        
        # Extract skills by keyword matching
        skills = []
        for skill in TECHNICAL_SKILLS:
            if skill in text_lower:
                # Try to determine proficiency
                proficiency = "intermediate"
                for level, indicators in PROFICIENCY_INDICATORS.items():
                    for indicator in indicators:
                        # Check if indicator appears near the skill
                        pattern = rf"{re.escape(indicator)}.*{re.escape(skill)}|{re.escape(skill)}.*{re.escape(indicator)}"
                        if re.search(pattern, text_lower):
                            proficiency = level
                            break
                
                skills.append({
                    "name": skill.title(),
                    "category": "technical",
                    "proficiency": proficiency,
                    "evidence": f"Found '{skill}' in resume",
                    "confidence": 0.6,
                })
        
        for skill in SOFT_SKILLS:
            if skill in text_lower:
                skills.append({
                    "name": skill.title(),
                    "category": "soft",
                    "proficiency": "intermediate",
                    "evidence": f"Found '{skill}' in resume",
                    "confidence": 0.5,
                })
        
        # Extract experience (basic pattern matching)
        experience = []
        exp_patterns = [
            r"(\d+)\+?\s*years?\s*(?:of\s+)?experience",
            r"(\d{4})\s*[-–]\s*(\d{4}|present|current)",
        ]
        
        total_years = 0
        for pattern in exp_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if isinstance(match, tuple):
                    # Year range pattern
                    start = int(match[0])
                    end = datetime.now().year if match[1] in ["present", "current"] else int(match[1])
                    years = end - start
                else:
                    years = int(match)
                total_years = max(total_years, years)
        
        if total_years > 0:
            experience.append({
                "role": "Professional Role",
                "duration_months": total_years * 12,
                "responsibilities": [],
                "achievements": [],
                "skills_used": [s["name"] for s in skills[:5]],
            })
        
        # Extract education (basic pattern matching)
        education = []
        degree_patterns = {
            "phd": "PhD",
            "ph.d": "PhD",
            "doctorate": "PhD",
            "master": "Master's",
            "msc": "Master's",
            "mba": "MBA",
            "bachelor": "Bachelor's",
            "bsc": "Bachelor's",
            "b.tech": "Bachelor's",
            "b.e": "Bachelor's",
        }
        
        for pattern, degree in degree_patterns.items():
            if pattern in text_lower:
                education.append({
                    "degree": degree,
                    "field": "Not specified",
                    "graduation_year": None,
                    "gpa": None,
                    "coursework": [],
                })
                break
        
        # Extract certifications
        certifications = []
        cert_keywords = ["certified", "certification", "certificate", "aws", "azure", "gcp", "pmp", "scrum"]
        for keyword in cert_keywords:
            if keyword in text_lower:
                # Try to extract the full certification name
                pattern = rf"[\w\s]*{keyword}[\w\s]*"
                matches = re.findall(pattern, text_lower)
                for match in matches[:3]:  # Limit to 3
                    cert_name = match.strip().title()
                    if len(cert_name) > 5 and cert_name not in certifications:
                        certifications.append(cert_name)
        
        return {
            "summary": "Professional with diverse skills and experience.",
            "skills": skills[:20],  # Limit to 20 skills
            "experience": experience,
            "education": education,
            "certifications": certifications[:5],
            "projects": [],
            "confidence": 0.5,  # Lower confidence for rule-based
        }
    
    def _calculate_quality_score(self, extracted_data: Dict[str, Any], resume_text: str) -> float:
        """Calculate a quality score for the resume based on completeness."""
        score = 0.0
        max_score = 0.0
        
        # Has professional summary
        max_score += 0.15
        if extracted_data.get("summary") and len(extracted_data["summary"]) > 20:
            score += 0.15
        
        # Has skills
        max_score += 0.25
        skills_count = len(extracted_data.get("skills", []))
        if skills_count >= 5:
            score += 0.25
        elif skills_count >= 3:
            score += 0.15
        elif skills_count >= 1:
            score += 0.05
        
        # Has experience
        max_score += 0.30
        exp_count = len(extracted_data.get("experience", []))
        if exp_count >= 2:
            score += 0.30
        elif exp_count >= 1:
            score += 0.20
        
        # Has education
        max_score += 0.15
        if len(extracted_data.get("education", [])) >= 1:
            score += 0.15
        
        # Resume length (not too short, not too long)
        max_score += 0.15
        word_count = len(resume_text.split())
        if 200 <= word_count <= 1500:
            score += 0.15
        elif 100 <= word_count <= 2000:
            score += 0.10
        elif word_count > 50:
            score += 0.05
        
        return score / max_score if max_score > 0 else 0.0
    
    def validate_input(self, input_data: Dict[str, Any]) -> list[str]:
        """Validate the input before processing."""
        errors = []
        
        if not input_data.get("candidate_id"):
            errors.append("candidate_id is required")
        
        if not input_data.get("resume_text"):
            errors.append("resume_text is required")
        
        return errors
