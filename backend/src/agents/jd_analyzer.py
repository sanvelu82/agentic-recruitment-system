"""
JD Analyzer Agent

Responsibility: Parse and structure job descriptions.
Single purpose: Extract requirements, skills, and evaluation criteria from JDs.

This agent does NOT match candidates - only analyzes job descriptions.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

from .base import BaseAgent
from ..core.llm import get_agent_decision
from ..schemas.job import (
    JobDescription,
    ParsedJD,
    SkillRequirement,
    ExperienceRequirement,
    EducationRequirement,
)

# LangChain imports for LLM integration (using Groq)
try:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

JD_SYSTEM_PROMPT = """You are an expert Technical Recruiter and HR Analyst with 15+ years of experience in talent acquisition.
Your task is to meticulously analyze job descriptions and extract structured information.

You MUST return a JSON object with this EXACT schema:
{
    "normalized_title": "standardized job title (e.g., 'Senior Software Engineer')",
    "seniority_level": "entry|junior|mid|senior|lead|principal|executive",
    "job_function": "engineering|data_science|product|design|marketing|sales|hr|finance|operations|other",
    "skills": [
        {
            "name": "skill name",
            "category": "technical|soft|domain",
            "importance": "required|preferred|nice_to_have",
            "proficiency": "beginner|intermediate|advanced|expert",
            "years": null or number,
            "context": "how the skill will be used"
        }
    ],
    "experience": {
        "minimum_years": number,
        "preferred_years": number or null,
        "domains": ["relevant industry/domain"],
        "roles": ["relevant previous roles"]
    },
    "education": {
        "minimum_degree": "high_school|associate|bachelors|masters|phd|none",
        "preferred_degree": "degree or null",
        "fields": ["accepted fields of study"],
        "certifications_required": ["required certifications"],
        "certifications_preferred": ["preferred certifications"]
    },
    "responsibilities": ["key responsibility 1", "key responsibility 2"],
    "technical_topics": ["topic for technical assessment"],
    "confidence": 0.0 to 1.0
}

Guidelines:
1. Extract ALL technical skills (languages, frameworks, tools, platforms)
2. Identify soft skills (communication, leadership, teamwork)
3. Be specific with technical topics for assessment (e.g., "Python data structures", "REST API design")
4. Infer seniority from years of experience and responsibilities if not explicit
5. Set confidence based on how clearly the JD specifies requirements
"""
# Comprehensive bias term dictionaries
GENDERED_TERMS = {
    # Masculine-coded terms
    "rockstar": "high performer",
    "ninja": "expert",
    "guru": "specialist", 
    "hacker": "developer",
    "manpower": "workforce",
    "manning": "staffing",
    "chairman": "chairperson",
    "fireman": "firefighter",
    "policeman": "police officer",
    "salesman": "salesperson",
    "businessman": "business professional",
    "spokesman": "spokesperson",
    "craftsman": "craftsperson",
    "workmanship": "quality of work",
    "man-hours": "work hours",
    "aggressive": "ambitious",
    "dominant": "leading",
    "competitive": "results-driven",
    "assertive": "confident",
    # Feminine-coded terms that may discourage male applicants
    "nurturing": "supportive",
    "collaborative": None,  # Generally acceptable
}

AGE_BIASED_TERMS = {
    "young": "motivated",
    "energetic": "enthusiastic",
    "digital native": "tech-savvy",
    "recent graduate only": "entry-level candidates",
    "fresh graduate": "new graduate",
    "youthful": "dynamic",
    "mature": "experienced",
    "seasoned veteran": "experienced professional",
    "overqualified": None,  # Flag but no replacement
    "cultural fit": "team alignment",
    "fast-paced environment for young professionals": "dynamic work environment",
}

DISABILITY_BIASED_TERMS = {
    "must be able to stand": "position requires standing (reasonable accommodations available)",
    "must have valid driver's license": "transportation required (accommodations available)",
    "physically demanding": "role involves physical activity",
}

ETHNICITY_BIASED_TERMS = {
    "native english speaker": "fluent in English",
    "american accent": "clear communication skills",
    "local candidates only": "candidates in [location] area preferred",
}


class JDAnalyzerAgent(BaseAgent[JobDescription, ParsedJD]):
    """
    Analyzes job descriptions and extracts structured requirements using LLM.
    
    Input: Raw job description
    Output: ParsedJD with skills, requirements, and evaluation criteria
    
    Key responsibilities:
    - Extract required and preferred skills using LLM
    - Identify experience requirements
    - Determine education requirements
    - Identify topics for test generation
    - Flag potential bias in JD language
    
    Does NOT:
    - Match candidates to the JD
    - Score candidates
    - Generate tests
    """
    
    # Class attributes required by BaseAgent
    name: str = "jd_analyzer"
    
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
    
    @property
    def description(self) -> str:
        return (
            "Analyzes job descriptions to extract structured requirements "
            "including skills, experience, education, and topics for assessment."
        )
    
    def run(
        self,
        input_data: JobDescription,
        state: Optional["PipelineState"] = None
    ) -> "AgentResult[ParsedJD]":
        """
        Execute JD analysis and return result.
        
        Uses get_agent_decision utility to extract JD details via LLM,
        then maps the response to a ParsedJD schema.
        
        Args:
            input_data: JobDescription to analyze
            state: Optional pipeline state (created if not provided)
        
        Returns:
            AgentResult with ParsedJD and updated state
        """
        from ..schemas.messages import PipelineState
        from .base import AgentResult, AgentResponse, AgentStatus
        
        # Create default state if not provided
        if state is None:
            state = PipelineState(job_id=input_data.job_id)
        
        try:
            self.log_reasoning("Starting JD analysis with LLM")
            self.log_reasoning(f"Analyzing job: {input_data.title}")
            
            # Check for potential bias in JD
            bias_flags = self._check_for_bias(input_data.raw_description)
            if bias_flags:
                self.log_reasoning(f"Potential bias detected: {len(bias_flags)} issues found")
            
            # Build user payload for LLM
            user_payload = f"""Analyze this job description and extract structured information:

Job Title: {input_data.title}
Department: {input_data.department}
Location: {input_data.location}
Employment Type: {input_data.employment_type}
Experience Range: {input_data.experience_years_min}-{input_data.experience_years_max} years

Job Description:
{input_data.raw_description}"""
            
            # Call LLM using the utility function
            llm_response = get_agent_decision(JD_SYSTEM_PROMPT, user_payload)
            self.log_reasoning("LLM extraction successful")
            
            # Parse LLM response and map to schema
            extracted_data = json.loads(llm_response)
            parsed = self._map_to_schema(input_data, extracted_data, bias_flags)
            
            # Calculate quality score
            jd_quality_score = self._calculate_jd_quality(input_data, extracted_data, bias_flags)
            parsed.jd_quality_score = jd_quality_score
            
            confidence = parsed.parsing_confidence
            explanation = (
                f"Analyzed job description for '{input_data.title}'. "
                f"Extracted {len(parsed.skills)} skill requirements "
                f"({len(parsed.get_required_skills())} required, "
                f"{len(parsed.get_preferred_skills())} preferred), "
                f"experience requirements ({parsed.experience_requirements.minimum_years}+ years), "
                f"education requirements ({parsed.education_requirements.minimum_degree}), "
                f"and {len(parsed.technical_topics)} technical topics for assessment. "
                f"JD quality score: {jd_quality_score:.0%}. "
                f"Identified {len(bias_flags)} potential bias concerns."
            )
            
            self.log_reasoning("JD analysis completed")
            
            response = AgentResponse(
                agent_name=self.name,
                status=AgentStatus.SUCCESS,
                output=parsed,
                confidence_score=confidence,
                explanation=explanation,
            )
            
            # Update state with parsed JD
            new_state = PipelineState(
                pipeline_id=state.pipeline_id,
                job_id=state.job_id,
                current_stage=state.current_stage,
                job_description=state.job_description,
                parsed_jd=parsed.to_dict(),
                candidates=state.candidates,
            )
            
            return AgentResult(response=response, state=new_state)
            
        except Exception as e:
            self.log_reasoning(f"JD analysis failed: {str(e)}")
            response = AgentResponse(
                agent_name=self.name,
                status=AgentStatus.FAILURE,
                output=None,
                confidence_score=0.0,
                explanation=f"JD analysis failed: {str(e)}",
            )
            return AgentResult(response=response, state=state)
    
    def _map_to_schema(
        self,
        input_data: JobDescription,
        extracted_data: Dict[str, Any],
        bias_flags: List[str]
    ) -> ParsedJD:
        """
        Map LLM's JSON output to ParsedJD schema.
        
        Args:
            input_data: Original JobDescription
            extracted_data: Parsed JSON from LLM response
            bias_flags: List of detected bias issues
        
        Returns:
            ParsedJD object with all fields populated
        """
        # Build skill requirements
        skills = []
        for skill_data in extracted_data.get("skills", []):
            skill = SkillRequirement(
                skill_name=skill_data.get("name", ""),
                category=skill_data.get("category", "technical"),
                importance=skill_data.get("importance", "required"),
                minimum_proficiency=skill_data.get("proficiency", "intermediate"),
                years_experience=skill_data.get("years"),
                context=skill_data.get("context", ""),
            )
            skills.append(skill)
        
        # Build experience requirements
        exp_data = extracted_data.get("experience", {})
        experience_req = ExperienceRequirement(
            minimum_years=exp_data.get("minimum_years", 0),
            preferred_years=exp_data.get("preferred_years"),
            relevant_domains=exp_data.get("domains", []),
            specific_roles=exp_data.get("roles", []),
        )
        
        # Build education requirements
        edu_data = extracted_data.get("education", {})
        education_req = EducationRequirement(
            minimum_degree=edu_data.get("minimum_degree", ""),
            preferred_degree=edu_data.get("preferred_degree"),
            accepted_fields=edu_data.get("fields", []),
            certifications_required=edu_data.get("certifications_required", []),
            certifications_preferred=edu_data.get("certifications_preferred", []),
        )
        
        # Determine difficulty level from seniority
        seniority = extracted_data.get("seniority_level", "mid")
        difficulty_map = {
            "entry": "beginner",
            "junior": "beginner",
            "mid": "intermediate",
            "senior": "advanced",
            "lead": "advanced",
            "principal": "advanced",
            "executive": "advanced",
        }
        difficulty_level = difficulty_map.get(seniority, "intermediate")
        
        return ParsedJD(
            job_id=input_data.job_id,
            parsing_confidence=extracted_data.get("confidence", 0.85),
            job_title_normalized=extracted_data.get("normalized_title", input_data.title),
            seniority_level=seniority,
            job_function=extracted_data.get("job_function", "engineering"),
            skills=skills,
            experience_requirements=experience_req,
            education_requirements=education_req,
            key_responsibilities=extracted_data.get("responsibilities", []),
            technical_topics=extracted_data.get("technical_topics", []),
            difficulty_level=difficulty_level,
            potential_bias_flags=bias_flags,
            parsing_warnings=[],
        )
    
    def _process(
        self, 
        input_data: JobDescription
    ) -> tuple[ParsedJD, float, str]:
        """
        Parse a job description into structured format using LLM.
        
        Args:
            input_data: JobDescription object
        
        Returns:
            ParsedJD, confidence_score, explanation
        """
        self.log_reasoning("Starting JD analysis with LLM")
        self.log_reasoning(f"Analyzing job: {input_data.title}")
        
        # Check for potential bias in JD
        bias_flags = self._check_for_bias(input_data.raw_description)
        if bias_flags:
            self.log_reasoning(f"Potential bias detected: {len(bias_flags)} issues found")
        
        # Extract structured data using LLM
        llm = self._get_llm()
        parsing_warnings = []
        
        if llm and LANGCHAIN_AVAILABLE:
            try:
                extracted_data = self._extract_with_llm(input_data, llm)
                self.log_reasoning("LLM extraction successful")
            except Exception as e:
                self.log_reasoning(f"LLM extraction failed: {str(e)}, falling back to rule-based")
                extracted_data = self._extract_with_rules(input_data)
                parsing_warnings.append(f"LLM extraction failed, used rule-based fallback: {str(e)}")
        else:
            self.log_reasoning("LLM not available, using rule-based extraction")
            extracted_data = self._extract_with_rules(input_data)
            parsing_warnings.append("LLM not configured, used rule-based extraction")
        
        # Build skill requirements
        skills = []
        for skill_data in extracted_data.get("skills", []):
            skill = SkillRequirement(
                skill_name=skill_data.get("name", ""),
                category=skill_data.get("category", "technical"),
                importance=skill_data.get("importance", "required"),
                minimum_proficiency=skill_data.get("proficiency", "intermediate"),
                years_experience=skill_data.get("years"),
                context=skill_data.get("context", ""),
            )
            skills.append(skill)
        
        # Build experience requirements
        exp_data = extracted_data.get("experience", {})
        experience_req = ExperienceRequirement(
            minimum_years=exp_data.get("minimum_years", 0),
            preferred_years=exp_data.get("preferred_years"),
            relevant_domains=exp_data.get("domains", []),
            specific_roles=exp_data.get("roles", []),
        )
        
        # Build education requirements
        edu_data = extracted_data.get("education", {})
        education_req = EducationRequirement(
            minimum_degree=edu_data.get("minimum_degree", ""),
            preferred_degree=edu_data.get("preferred_degree"),
            accepted_fields=edu_data.get("fields", []),
            certifications_required=edu_data.get("certifications_required", []),
            certifications_preferred=edu_data.get("certifications_preferred", []),
        )
        
        # Calculate JD quality score
        jd_quality_score = self._calculate_jd_quality(input_data, extracted_data, bias_flags)
        
        # Determine difficulty level from seniority
        seniority = extracted_data.get("seniority_level", "mid")
        difficulty_map = {
            "entry": "beginner",
            "junior": "beginner", 
            "mid": "intermediate",
            "senior": "advanced",
            "lead": "advanced",
            "principal": "advanced",
            "executive": "advanced",
        }
        difficulty_level = difficulty_map.get(seniority, "intermediate")
        
        parsed = ParsedJD(
            job_id=input_data.job_id,
            parsing_confidence=extracted_data.get("confidence", 0.85),
            job_title_normalized=extracted_data.get("normalized_title", input_data.title),
            seniority_level=seniority,
            job_function=extracted_data.get("job_function", "engineering"),
            skills=skills,
            experience_requirements=experience_req,
            education_requirements=education_req,
            key_responsibilities=extracted_data.get("responsibilities", []),
            technical_topics=extracted_data.get("technical_topics", []),
            difficulty_level=difficulty_level,
            jd_quality_score=jd_quality_score,
            potential_bias_flags=bias_flags,
            parsing_warnings=parsing_warnings,
        )
        
        confidence = parsed.parsing_confidence
        explanation = (
            f"Analyzed job description for '{input_data.title}'. "
            f"Extracted {len(parsed.skills)} skill requirements "
            f"({len([s for s in skills if s.importance == 'required'])} required, "
            f"{len([s for s in skills if s.importance == 'preferred'])} preferred), "
            f"experience requirements ({parsed.experience_requirements.minimum_years}+ years), "
            f"education requirements ({parsed.education_requirements.minimum_degree}), "
            f"and {len(parsed.technical_topics)} technical topics for assessment. "
            f"JD quality score: {jd_quality_score:.0%}. "
            f"Identified {len(bias_flags)} potential bias concerns."
        )
        
        self.log_reasoning("JD analysis completed")
        
        return parsed, confidence, explanation
    
    def _extract_with_llm(self, jd: JobDescription, llm) -> Dict[str, Any]:
        """Extract structured data from JD using LLM."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert HR analyst and technical recruiter. 
Analyze the job description and extract structured information.
Return ONLY valid JSON with no additional text."""),
            ("user", """Analyze this job description and extract structured information:

Job Title: {title}
Department: {department}
Location: {location}
Employment Type: {employment_type}

Job Description:
{raw_description}

Extract and return a JSON object with this exact structure:
{{
    "normalized_title": "standardized job title",
    "seniority_level": "entry|junior|mid|senior|lead|principal|executive",
    "job_function": "engineering|data_science|product|design|marketing|sales|hr|finance|operations|other",
    "skills": [
        {{
            "name": "skill name",
            "category": "technical|soft|domain",
            "importance": "required|preferred|nice_to_have",
            "proficiency": "beginner|intermediate|advanced|expert",
            "years": null or number,
            "context": "how skill is used"
        }}
    ],
    "experience": {{
        "minimum_years": number,
        "preferred_years": number or null,
        "domains": ["relevant industry/domain"],
        "roles": ["relevant previous roles"]
    }},
    "education": {{
        "minimum_degree": "high_school|associate|bachelors|masters|phd|none",
        "preferred_degree": "degree or null",
        "fields": ["accepted fields of study"],
        "certifications_required": ["required certs"],
        "certifications_preferred": ["preferred certs"]
    }},
    "responsibilities": ["key responsibility 1", "key responsibility 2"],
    "technical_topics": ["topic for technical assessment 1", "topic 2", "topic 3"],
    "confidence": 0.0 to 1.0
}}

Focus on extracting:
1. All technical skills mentioned (programming languages, frameworks, tools)
2. Soft skills (communication, leadership, teamwork)
3. Domain knowledge requirements
4. Clear technical topics that could be assessed in an MCQ test
5. Be specific with technical topics - e.g., "Python data structures", "REST API design", "SQL query optimization"
""")
        ])
        
        parser = JsonOutputParser()
        chain = prompt | llm | parser
        
        result = chain.invoke({
            "title": jd.title,
            "department": jd.department,
            "location": jd.location,
            "employment_type": jd.employment_type,
            "raw_description": jd.raw_description,
        })
        
        return result
    
    def _extract_with_rules(self, jd: JobDescription) -> Dict[str, Any]:
        """Rule-based extraction fallback when LLM is not available."""
        text = jd.raw_description.lower()
        
        # Extract skills using pattern matching
        skills = []
        
        # Technical skills patterns
        tech_patterns = {
            "python": ("Python", "technical", "programming language"),
            "javascript": ("JavaScript", "technical", "programming language"),
            "typescript": ("TypeScript", "technical", "programming language"),
            "java": ("Java", "technical", "programming language"),
            "c++": ("C++", "technical", "programming language"),
            "c#": ("C#", "technical", "programming language"),
            "golang": ("Go", "technical", "programming language"),
            "rust": ("Rust", "technical", "programming language"),
            "react": ("React", "technical", "frontend framework"),
            "angular": ("Angular", "technical", "frontend framework"),
            "vue": ("Vue.js", "technical", "frontend framework"),
            "node.js": ("Node.js", "technical", "backend runtime"),
            "django": ("Django", "technical", "web framework"),
            "flask": ("Flask", "technical", "web framework"),
            "fastapi": ("FastAPI", "technical", "web framework"),
            "spring": ("Spring", "technical", "java framework"),
            "aws": ("AWS", "technical", "cloud platform"),
            "azure": ("Azure", "technical", "cloud platform"),
            "gcp": ("Google Cloud", "technical", "cloud platform"),
            "docker": ("Docker", "technical", "containerization"),
            "kubernetes": ("Kubernetes", "technical", "orchestration"),
            "sql": ("SQL", "technical", "database"),
            "postgresql": ("PostgreSQL", "technical", "database"),
            "mongodb": ("MongoDB", "technical", "database"),
            "redis": ("Redis", "technical", "database"),
            "git": ("Git", "technical", "version control"),
            "ci/cd": ("CI/CD", "technical", "devops"),
            "machine learning": ("Machine Learning", "technical", "ai/ml"),
            "deep learning": ("Deep Learning", "technical", "ai/ml"),
            "tensorflow": ("TensorFlow", "technical", "ml framework"),
            "pytorch": ("PyTorch", "technical", "ml framework"),
            "nlp": ("NLP", "technical", "ai/ml"),
            "data analysis": ("Data Analysis", "technical", "analytics"),
            "rest api": ("REST API", "technical", "api design"),
            "graphql": ("GraphQL", "technical", "api design"),
            "microservices": ("Microservices", "technical", "architecture"),
            "agile": ("Agile", "domain", "methodology"),
            "scrum": ("Scrum", "domain", "methodology"),
        }
        
        for pattern, (name, category, context) in tech_patterns.items():
            if pattern in text:
                # Determine importance from context
                importance = "required" if any(
                    req in text[max(0, text.find(pattern)-50):text.find(pattern)+50]
                    for req in ["required", "must have", "essential", "mandatory"]
                ) else "preferred"
                
                skills.append({
                    "name": name,
                    "category": category,
                    "importance": importance,
                    "proficiency": "intermediate",
                    "years": None,
                    "context": context,
                })
        
        # Soft skills
        soft_skills = ["communication", "leadership", "teamwork", "problem-solving", 
                       "analytical", "collaboration", "presentation"]
        for skill in soft_skills:
            if skill in text:
                skills.append({
                    "name": skill.title(),
                    "category": "soft",
                    "importance": "preferred",
                    "proficiency": "intermediate",
                    "years": None,
                    "context": "interpersonal skill",
                })
        
        # Extract years of experience
        years_pattern = r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)?'
        years_matches = re.findall(years_pattern, text)
        min_years = int(years_matches[0]) if years_matches else 3
        
        # Extract education
        education = {"minimum_degree": "bachelors", "fields": [], "certifications_required": [], "certifications_preferred": []}
        if "phd" in text or "doctorate" in text:
            education["minimum_degree"] = "phd"
        elif "master" in text:
            education["minimum_degree"] = "masters"
        elif "bachelor" in text or "degree" in text:
            education["minimum_degree"] = "bachelors"
        
        # Determine seniority
        seniority = "mid"
        if any(term in text for term in ["senior", "sr.", "lead", "principal", "staff"]):
            seniority = "senior"
        elif any(term in text for term in ["junior", "jr.", "entry", "associate"]):
            seniority = "entry"
        elif "lead" in text or "manager" in text:
            seniority = "lead"
        
        # Extract technical topics for assessment
        technical_topics = []
        for skill in skills[:10]:  # Top 10 skills as topics
            if skill["category"] == "technical":
                technical_topics.append(skill["name"])
        
        return {
            "normalized_title": jd.title,
            "seniority_level": seniority,
            "job_function": "engineering",
            "skills": skills,
            "experience": {
                "minimum_years": min_years,
                "preferred_years": min_years + 2,
                "domains": [],
                "roles": [],
            },
            "education": education,
            "responsibilities": [],
            "technical_topics": technical_topics[:8],  # Limit to 8 topics
            "confidence": 0.7,  # Lower confidence for rule-based
        }
    
    def _calculate_jd_quality(
        self, 
        jd: JobDescription, 
        extracted: Dict[str, Any],
        bias_flags: List[str]
    ) -> float:
        """Calculate a quality score for the job description."""
        score = 1.0
        
        # Penalize for bias
        score -= len(bias_flags) * 0.05
        
        # Penalize for too few skills extracted
        if len(extracted.get("skills", [])) < 3:
            score -= 0.1
        
        # Penalize for missing technical topics
        if len(extracted.get("technical_topics", [])) < 2:
            score -= 0.1
        
        # Penalize for very short JD
        if len(jd.raw_description) < 200:
            score -= 0.15
        
        # Penalize for missing experience info
        if extracted.get("experience", {}).get("minimum_years", 0) == 0:
            score -= 0.05
        
        return max(0.0, min(1.0, score))
    
    def _check_for_bias(self, text: str) -> list[str]:
        """
        Comprehensive check for potentially biased language in job descriptions.
        
        Uses multiple bias dictionaries to detect:
        - Gendered language
        - Age-related bias
        - Disability bias
        - Ethnicity/nationality bias
        """
        flags = []
        text_lower = text.lower()
        
        # Check for gendered language
        for term, replacement in GENDERED_TERMS.items():
            if term in text_lower:
                if replacement:
                    flags.append(
                        f"Gendered term '{term}' detected. Consider using '{replacement}' instead."
                    )
                else:
                    flags.append(f"Potentially gendered term '{term}' detected.")
        
        # Check for age-related bias  
        for term, replacement in AGE_BIASED_TERMS.items():
            if term in text_lower:
                if replacement:
                    flags.append(
                        f"Age-biased term '{term}' detected. Consider using '{replacement}' instead."
                    )
                else:
                    flags.append(f"Potentially age-biased term '{term}' detected.")
        
        # Check for disability bias
        for term, replacement in DISABILITY_BIASED_TERMS.items():
            if term in text_lower:
                flags.append(
                    f"Potential disability bias: '{term}'. Consider: '{replacement}'"
                )
        
        # Check for ethnicity/nationality bias
        for term, replacement in ETHNICITY_BIASED_TERMS.items():
            if term in text_lower:
                flags.append(
                    f"Potential ethnicity/nationality bias: '{term}'. Consider: '{replacement}'"
                )
        
        # Check for exclusionary patterns
        exclusionary_patterns = [
            (r"must be (\d+)-(\d+) years old", "Age requirement detected - may be illegal in many jurisdictions"),
            (r"no older than \d+", "Age restriction detected - likely discriminatory"),
            (r"preferably (male|female)", "Gender preference detected - discriminatory"),
        ]
        
        for pattern, message in exclusionary_patterns:
            if re.search(pattern, text_lower):
                flags.append(message)
        
        return flags
