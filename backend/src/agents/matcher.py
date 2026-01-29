"""
Matcher Agent

Responsibility: Compare parsed resumes against parsed job descriptions.
Single purpose: Calculate similarity scores with explainable metrics.

This agent does NOT make shortlisting decisions - only calculates scores.
"""

import os
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseAgent
from ..schemas.candidates import ParsedResume, MatchResult, SkillMatch
from ..schemas.job import ParsedJD, SkillRequirement

# Optional: Sentence transformers for semantic similarity
try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Note: Groq does not provide embeddings API, so we rely on sentence-transformers
# If sentence-transformers is unavailable, we fall back to fuzzy string matching
GROQ_EMBEDDINGS_AVAILABLE = False  # Groq doesn't have embeddings endpoint


class MatcherInput:
    """Input structure for the Matcher agent."""
    def __init__(
        self, 
        candidate_id: str = "",
        parsed_resume: Any = None,  # Can be ParsedResume or dict
        parsed_jd: Any = None,  # Can be ParsedJD or dict
    ):
        self.candidate_id = candidate_id
        self.parsed_resume = parsed_resume
        self.parsed_jd = parsed_jd


class MatcherAgent(BaseAgent[MatcherInput, MatchResult]):
    """
    Matches candidate resumes against job descriptions using semantic similarity.
    
    Input: ParsedResume + ParsedJD
    Output: MatchResult with detailed scoring breakdown
    
    Key responsibilities:
    - Calculate skill match scores using semantic similarity
    - Calculate experience match scores
    - Calculate education match scores
    - Provide detailed match explanations
    - Identify strengths and gaps
    
    Does NOT:
    - Make shortlisting decisions
    - Rank candidates against each other
    - Generate tests
    """
    
    def __init__(self, agent_id: Optional[str] = None, embedding_model: str = "all-MiniLM-L6-v2"):
        super().__init__(agent_id)
        self.embedding_model_name = embedding_model
        self._embedding_model = None
        self._skill_cache: Dict[str, List[float]] = {}  # Cache embeddings
    
    def _get_embedding_model(self):
        """Lazy initialization of embedding model (sentence-transformers)."""
        if self._embedding_model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self._embedding_model = SentenceTransformer(self.embedding_model_name)
            except Exception as e:
                self.log_reasoning(f"Failed to load SentenceTransformer: {e}")
        return self._embedding_model
    
    @property
    def description(self) -> str:
        return (
            "Calculates similarity scores between resumes and job descriptions "
            "with detailed, explainable metrics for each component."
        )
    
    def run(
        self,
        input_data: MatcherInput,
        state: Optional["PipelineState"] = None
    ) -> "AgentResult[MatchResult]":
        """
        Execute matching and return result.
        
        Args:
            input_data: MatcherInput with parsed resume and JD
            state: Optional pipeline state
        
        Returns:
            AgentResult with MatchResult and updated state
        """
        from ..schemas.messages import PipelineState
        from .base import AgentResult, AgentResponse, AgentStatus
        
        if state is None:
            state = PipelineState()
        
        try:
            result, confidence, explanation = self._process(input_data)
            
            response = AgentResponse(
                agent_name=self.name,
                status=AgentStatus.SUCCESS,
                output=result,
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
                explanation=f"Matching failed: {str(e)}",
            )
            return AgentResult(response=response, state=state)
    
    def _process(
        self, 
        input_data: MatcherInput
    ) -> tuple[MatchResult, float, str]:
        """
        Match a resume against a job description using semantic similarity.
        
        Args:
            input_data: MatcherInput containing ParsedResume and ParsedJD
        
        Returns:
            MatchResult, confidence_score, explanation
        """
        resume = input_data.parsed_resume
        jd = input_data.parsed_jd
        
        self.log_reasoning(f"Matching candidate {resume.candidate_id[:8]} to job {jd.job_id[:8]}")
        
        # Calculate component scores with detailed breakdowns
        skills_score, skill_matches, skills_analysis = self._calculate_skills_match(resume, jd)
        self.log_reasoning(f"Skills match score: {skills_score:.2f}")
        
        experience_score, exp_gap = self._calculate_experience_match(resume, jd)
        self.log_reasoning(f"Experience match score: {experience_score:.2f}")
        
        education_score, edu_analysis = self._calculate_education_match(resume, jd)
        self.log_reasoning(f"Education match score: {education_score:.2f}")
        
        # Calculate weighted overall score
        weights = jd.scoring_weights
        overall_score = (
            skills_score * weights.get("skills", 0.4) +
            experience_score * weights.get("experience", 0.35) +
            education_score * weights.get("education", 0.25)
        )
        self.log_reasoning(f"Overall weighted score: {overall_score:.2f}")
        
        # Count skills coverage
        required_skills = jd.get_required_skills()
        preferred_skills = jd.get_preferred_skills()
        
        required_met = sum(1 for m in skill_matches if m.match_score >= 0.7 
                         and any(r.skill_name.lower() in m.required_skill.lower() 
                                for r in required_skills))
        preferred_met = sum(1 for m in skill_matches if m.match_score >= 0.6
                          and any(p.skill_name.lower() in m.required_skill.lower()
                                 for p in preferred_skills))
        
        # Calculate confidence based on data quality
        confidence = self._calculate_confidence(resume, jd, skill_matches)
        
        # Identify strengths and gaps
        strengths = self._identify_strengths(resume, jd, skill_matches, experience_score, education_score)
        gaps = self._identify_gaps(resume, jd, skill_matches, experience_score, education_score)
        
        # Build detailed match explanation
        match_explanation = self._build_match_explanation(
            overall_score, skills_score, experience_score, education_score,
            skill_matches, exp_gap, strengths, gaps
        )
        
        # Build match result
        result = MatchResult(
            candidate_id=resume.candidate_id,
            job_id=jd.job_id,
            overall_match_score=overall_score,
            confidence=confidence,
            skills_match_score=skills_score,
            experience_match_score=experience_score,
            education_match_score=education_score,
            skill_matches=skill_matches,
            required_skills_met=required_met,
            required_skills_total=len(required_skills),
            preferred_skills_met=preferred_met,
            preferred_skills_total=len(preferred_skills),
            meets_experience_requirement=experience_score >= 0.7,
            experience_gap_months=exp_gap,
            match_explanation=match_explanation,
            strengths=strengths,
            gaps=gaps,
        )
        
        explanation = (
            f"Matched candidate to job with {overall_score:.0%} overall score. "
            f"Skills: {skills_score:.0%} ({required_met}/{len(required_skills)} required met), "
            f"Experience: {experience_score:.0%}, Education: {education_score:.0%}. "
            f"Found {len(strengths)} strengths and {len(gaps)} gaps."
        )
        
        return result, confidence, explanation
    
    def _calculate_skills_match(
        self, 
        resume: ParsedResume, 
        jd: ParsedJD
    ) -> Tuple[float, List[SkillMatch], Dict[str, Any]]:
        """
        Calculate skill match using semantic similarity.
        
        Returns:
            Tuple of (overall_score, list of SkillMatch, analysis dict)
        """
        skill_matches = []
        analysis = {"method": "unknown", "details": []}
        
        # Get candidate skills
        candidate_skills = [s.skill_name for s in resume.skills]
        candidate_skill_set = set(s.lower() for s in candidate_skills)
        
        # Get all required and preferred skills from JD
        all_jd_skills = jd.skills
        if not all_jd_skills:
            self.log_reasoning("No skills in JD, returning neutral score")
            return 0.5, [], {"method": "no_jd_skills"}
        
        if not candidate_skills:
            self.log_reasoning("No candidate skills found, returning low score")
            return 0.1, [], {"method": "no_candidate_skills"}
        
        # Try semantic matching first
        embedding_model = self._get_embedding_model()
        
        if embedding_model and SENTENCE_TRANSFORMERS_AVAILABLE:
            analysis["method"] = "semantic_similarity"
            self.log_reasoning("Using semantic similarity for skill matching")
            
            try:
                # Get embeddings for JD skills
                jd_skill_names = [s.skill_name for s in all_jd_skills]
                jd_embeddings = embedding_model.encode(jd_skill_names, convert_to_tensor=True)
                
                # Get embeddings for candidate skills  
                candidate_embeddings = embedding_model.encode(candidate_skills, convert_to_tensor=True)
                
                # Calculate similarity matrix
                similarity_matrix = util.cos_sim(jd_embeddings, candidate_embeddings)
                
                total_weighted_score = 0.0
                total_weight = 0.0
                
                for i, jd_skill in enumerate(all_jd_skills):
                    # Find best matching candidate skill
                    best_score = 0.0
                    best_match_idx = -1
                    
                    for j in range(len(candidate_skills)):
                        sim_score = float(similarity_matrix[i][j])
                        if sim_score > best_score:
                            best_score = sim_score
                            best_match_idx = j
                    
                    # Determine match type
                    candidate_match = candidate_skills[best_match_idx] if best_match_idx >= 0 else ""
                    
                    if jd_skill.skill_name.lower() == candidate_match.lower():
                        match_type = "exact"
                        best_score = 1.0  # Boost exact matches
                    elif best_score >= 0.8:
                        match_type = "semantic"
                    elif best_score >= 0.5:
                        match_type = "partial"
                    else:
                        match_type = "missing"
                        candidate_match = ""
                    
                    # Create skill match record
                    skill_match = SkillMatch(
                        required_skill=jd_skill.skill_name,
                        candidate_skill=candidate_match,
                        match_type=match_type,
                        match_score=best_score,
                        explanation=self._explain_skill_match(
                            jd_skill.skill_name, candidate_match, match_type, best_score
                        ),
                    )
                    skill_matches.append(skill_match)
                    
                    # Weight by importance
                    weight = 1.5 if jd_skill.importance == "required" else 1.0
                    if jd_skill.importance == "nice_to_have":
                        weight = 0.5
                    
                    total_weighted_score += best_score * weight
                    total_weight += weight
                
                overall_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
                
            except Exception as e:
                self.log_reasoning(f"Semantic matching failed: {e}, falling back to fuzzy matching")
                return self._fuzzy_skill_match(resume, jd)
        else:
            # Fallback to fuzzy string matching
            return self._fuzzy_skill_match(resume, jd)
        
        return overall_score, skill_matches, analysis
    
    def _fuzzy_skill_match(
        self, 
        resume: ParsedResume, 
        jd: ParsedJD
    ) -> Tuple[float, List[SkillMatch], Dict[str, Any]]:
        """
        Fallback fuzzy string matching for skills.
        """
        self.log_reasoning("Using fuzzy string matching for skills")
        
        skill_matches = []
        candidate_skills = [s.skill_name.lower() for s in resume.skills]
        all_jd_skills = jd.skills
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for jd_skill in all_jd_skills:
            jd_skill_lower = jd_skill.skill_name.lower()
            best_score = 0.0
            best_match = ""
            
            for cand_skill in candidate_skills:
                # Exact match
                if jd_skill_lower == cand_skill:
                    score = 1.0
                # Substring match
                elif jd_skill_lower in cand_skill or cand_skill in jd_skill_lower:
                    score = 0.85
                # Fuzzy match using SequenceMatcher
                else:
                    score = SequenceMatcher(None, jd_skill_lower, cand_skill).ratio()
                
                if score > best_score:
                    best_score = score
                    best_match = cand_skill
            
            # Determine match type
            if best_score >= 0.95:
                match_type = "exact"
            elif best_score >= 0.7:
                match_type = "partial"
            elif best_score >= 0.5:
                match_type = "fuzzy"
            else:
                match_type = "missing"
                best_match = ""
            
            skill_match = SkillMatch(
                required_skill=jd_skill.skill_name,
                candidate_skill=best_match.title() if best_match else "",
                match_type=match_type,
                match_score=best_score,
                explanation=self._explain_skill_match(
                    jd_skill.skill_name, best_match, match_type, best_score
                ),
            )
            skill_matches.append(skill_match)
            
            # Weight by importance
            weight = 1.5 if jd_skill.importance == "required" else 1.0
            if jd_skill.importance == "nice_to_have":
                weight = 0.5
            
            total_weighted_score += best_score * weight
            total_weight += weight
        
        overall_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        
        return overall_score, skill_matches, {"method": "fuzzy_matching"}
    
    def _explain_skill_match(
        self, 
        required: str, 
        candidate: str, 
        match_type: str, 
        score: float
    ) -> str:
        """Generate explanation for a skill match."""
        if match_type == "exact":
            return f"Exact match: Candidate has '{candidate}' which directly matches the requirement."
        elif match_type == "semantic":
            return f"Strong semantic match: '{candidate}' is closely related to required '{required}' (similarity: {score:.0%})."
        elif match_type == "partial":
            return f"Partial match: '{candidate}' partially matches '{required}' (similarity: {score:.0%})."
        elif match_type == "fuzzy":
            return f"Weak match: '{candidate}' may be related to '{required}' (similarity: {score:.0%})."
        else:
            return f"Missing: No matching skill found for required '{required}'."
    
    def _calculate_experience_match(
        self, 
        resume: ParsedResume, 
        jd: ParsedJD
    ) -> Tuple[float, int]:
        """
        Calculate experience match based on years and relevance.
        
        Returns:
            Tuple of (score, experience_gap_months)
        """
        required_months = jd.experience_requirements.minimum_years * 12
        candidate_months = resume.total_experience_months
        
        # Calculate gap (positive = exceeds requirement)
        gap_months = candidate_months - required_months
        
        if required_months == 0:
            # No experience requirement
            return 0.8, gap_months
        
        # Base score on meeting minimum requirement
        if candidate_months >= required_months:
            # Meets or exceeds - score increases with more experience up to a point
            base_score = 0.8
            
            # Bonus for preferred experience
            preferred_months = (jd.experience_requirements.preferred_years or 
                              jd.experience_requirements.minimum_years) * 12
            if candidate_months >= preferred_months:
                base_score = 1.0
            else:
                # Interpolate between min and preferred
                extra_ratio = (candidate_months - required_months) / max(1, preferred_months - required_months)
                base_score = 0.8 + (0.2 * min(1.0, extra_ratio))
            
            score = base_score
        else:
            # Below minimum - calculate partial credit
            ratio = candidate_months / required_months
            score = ratio * 0.7  # Max 70% if below requirement
        
        # Consider relevance of experience domains
        if jd.experience_requirements.relevant_domains and resume.experience:
            domain_match = self._check_domain_relevance(resume, jd)
            score = score * 0.7 + domain_match * 0.3  # 30% weight to domain relevance
        
        return min(1.0, score), gap_months
    
    def _check_domain_relevance(self, resume: ParsedResume, jd: ParsedJD) -> float:
        """Check if candidate's experience is in relevant domains."""
        required_domains = [d.lower() for d in jd.experience_requirements.relevant_domains]
        if not required_domains:
            return 0.8  # Neutral if no domains specified
        
        # Check each experience entry for domain relevance
        matches = 0
        for exp in resume.experience:
            role_lower = exp.role.lower()
            responsibilities = " ".join(exp.responsibilities).lower()
            
            for domain in required_domains:
                if domain in role_lower or domain in responsibilities:
                    matches += 1
                    break
        
        if resume.experience:
            return min(1.0, matches / len(resume.experience))
        return 0.5
    
    def _calculate_education_match(
        self, 
        resume: ParsedResume, 
        jd: ParsedJD
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate education match.
        
        Returns:
            Tuple of (score, analysis dict)
        """
        analysis = {"degree_match": False, "field_match": False}
        
        edu_req = jd.education_requirements
        candidate_education = resume.education
        
        if not edu_req.minimum_degree or edu_req.minimum_degree == "none":
            return 0.8, {"no_requirement": True}
        
        if not candidate_education:
            return 0.3, {"no_education_data": True}
        
        # Degree hierarchy
        degree_levels = {
            "high_school": 1,
            "associate": 2,
            "bachelors": 3,
            "masters": 4,
            "phd": 5,
        }
        
        required_level = degree_levels.get(edu_req.minimum_degree.lower(), 3)
        
        # Get highest candidate degree
        candidate_level = 0
        candidate_fields = []
        for edu in candidate_education:
            degree_lower = edu.degree.lower()
            for deg_name, level in degree_levels.items():
                if deg_name in degree_lower:
                    candidate_level = max(candidate_level, level)
                    break
            if edu.field_of_study:
                candidate_fields.append(edu.field_of_study.lower())
        
        # Score based on degree level
        if candidate_level >= required_level:
            degree_score = 1.0
            analysis["degree_match"] = True
        elif candidate_level == required_level - 1:
            degree_score = 0.7  # One level below
        else:
            degree_score = 0.4  # Significantly below
        
        # Check field of study match
        field_score = 0.8  # Default if no specific fields required
        if edu_req.accepted_fields:
            required_fields = [f.lower() for f in edu_req.accepted_fields]
            field_matches = any(
                any(req in cand for req in required_fields)
                for cand in candidate_fields
            )
            field_score = 1.0 if field_matches else 0.5
            analysis["field_match"] = field_matches
        
        # Check certifications
        cert_score = 0.8  # Default
        if edu_req.certifications_required:
            cert_matches = sum(
                1 for cert in edu_req.certifications_required
                if any(cert.lower() in c.lower() for c in resume.certifications)
            )
            cert_score = cert_matches / len(edu_req.certifications_required)
            analysis["required_certs_met"] = cert_matches
            analysis["required_certs_total"] = len(edu_req.certifications_required)
        
        # Weighted combination
        overall_score = degree_score * 0.5 + field_score * 0.3 + cert_score * 0.2
        
        return overall_score, analysis
    
    def _calculate_confidence(
        self, 
        resume: ParsedResume, 
        jd: ParsedJD, 
        skill_matches: List[SkillMatch]
    ) -> float:
        """Calculate confidence score based on data quality."""
        confidence = 0.9  # Start high
        
        # Lower confidence if resume parsing quality is low
        if resume.resume_quality_score < 0.7:
            confidence -= 0.1
        
        # Lower confidence if few skills could be matched
        if skill_matches:
            missing_count = sum(1 for m in skill_matches if m.match_type == "missing")
            missing_ratio = missing_count / len(skill_matches)
            if missing_ratio > 0.5:
                confidence -= 0.15
        
        # Lower confidence if JD parsing had warnings
        if jd.parsing_warnings:
            confidence -= 0.05
        
        # Lower confidence if resume has parsing warnings
        if resume.parsing_warnings:
            confidence -= 0.05
        
        return max(0.5, min(1.0, confidence))
    
    def _identify_strengths(
        self, 
        resume: ParsedResume, 
        jd: ParsedJD,
        skill_matches: List[SkillMatch],
        experience_score: float,
        education_score: float
    ) -> List[str]:
        """Identify candidate's strengths relative to the job."""
        strengths = []
        
        # Skill-based strengths
        strong_skill_matches = [m for m in skill_matches if m.match_score >= 0.8]
        if strong_skill_matches:
            top_skills = strong_skill_matches[:3]
            skill_names = ", ".join(m.candidate_skill for m in top_skills if m.candidate_skill)
            if skill_names:
                strengths.append(f"Strong skills match: {skill_names}")
        
        # Experience strengths
        if experience_score >= 0.9:
            strengths.append(f"Exceeds experience requirements with {resume.total_experience_months // 12}+ years")
        elif experience_score >= 0.8:
            strengths.append("Meets experience requirements")
        
        # Education strengths
        if education_score >= 0.9:
            strengths.append("Education exceeds requirements")
        
        # Certification strengths
        if resume.certifications:
            strengths.append(f"Has {len(resume.certifications)} relevant certifications")
        
        # Project strengths
        if resume.projects and len(resume.projects) >= 3:
            strengths.append("Demonstrates hands-on experience through projects")
        
        # Additional skills beyond requirements
        candidate_skill_count = len(resume.skills)
        required_skill_count = len(jd.get_required_skills())
        if candidate_skill_count > required_skill_count * 1.5:
            strengths.append("Brings additional valuable skills beyond requirements")
        
        return strengths[:5]  # Return top 5 strengths
    
    def _identify_gaps(
        self, 
        resume: ParsedResume, 
        jd: ParsedJD,
        skill_matches: List[SkillMatch],
        experience_score: float,
        education_score: float
    ) -> List[str]:
        """Identify gaps between candidate profile and job requirements."""
        gaps = []
        
        # Missing required skills
        missing_required = [
            m.required_skill for m in skill_matches 
            if m.match_type == "missing" 
            and any(s.skill_name == m.required_skill and s.importance == "required" 
                   for s in jd.skills)
        ]
        if missing_required:
            gaps.append(f"Missing required skills: {', '.join(missing_required[:3])}")
        
        # Experience gap
        if experience_score < 0.7:
            required_years = jd.experience_requirements.minimum_years
            candidate_years = resume.total_experience_months // 12
            gap = required_years - candidate_years
            if gap > 0:
                gaps.append(f"Experience gap: {gap} years below minimum requirement")
        
        # Education gap
        if education_score < 0.7:
            gaps.append("Education may not meet requirements")
        
        # Missing certifications
        if jd.education_requirements.certifications_required:
            missing_certs = [
                cert for cert in jd.education_requirements.certifications_required
                if not any(cert.lower() in c.lower() for c in resume.certifications)
            ]
            if missing_certs:
                gaps.append(f"Missing required certifications: {', '.join(missing_certs[:2])}")
        
        return gaps[:5]  # Return top 5 gaps
    
    def _build_match_explanation(
        self,
        overall_score: float,
        skills_score: float,
        experience_score: float,
        education_score: float,
        skill_matches: List[SkillMatch],
        exp_gap: int,
        strengths: List[str],
        gaps: List[str]
    ) -> str:
        """Build a comprehensive match explanation."""
        
        # Overall assessment
        if overall_score >= 0.85:
            assessment = "Excellent match"
        elif overall_score >= 0.7:
            assessment = "Good match"
        elif overall_score >= 0.5:
            assessment = "Moderate match"
        else:
            assessment = "Below requirements"
        
        # Build explanation
        explanation_parts = [
            f"{assessment} with {overall_score:.0%} overall score.",
            f"Skills: {skills_score:.0%} match",
        ]
        
        # Add skill details
        matched_count = sum(1 for m in skill_matches if m.match_score >= 0.7)
        total_count = len(skill_matches)
        explanation_parts.append(f"({matched_count}/{total_count} skills matched).")
        
        # Experience
        if exp_gap >= 0:
            explanation_parts.append(f"Experience: Exceeds requirement by {exp_gap // 12} years.")
        else:
            explanation_parts.append(f"Experience: {abs(exp_gap) // 12} years below requirement.")
        
        # Top strength
        if strengths:
            explanation_parts.append(f"Key strength: {strengths[0]}")
        
        # Top concern
        if gaps:
            explanation_parts.append(f"Key concern: {gaps[0]}")
        
        return " ".join(explanation_parts)
