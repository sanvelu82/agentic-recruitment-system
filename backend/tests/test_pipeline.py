#!/usr/bin/env python3
"""
Test script for the Agentic Recruitment System.

This script demonstrates the end-to-end flow of the recruitment pipeline.
Run with: python -m tests.test_pipeline
"""

import os
import sys

# Add the parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.orchestrator import OrchestratorAgent
from src.agents.jd_analyzer import JDAnalyzerAgent
from src.agents.resume_parser import ResumeParserAgent
from src.agents.matcher import MatcherAgent, MatcherInput
from src.agents.test_generator import TestGeneratorAgent, TestGeneratorInput
from src.agents.shortlister import ShortlisterAgent, ShortlistInput
from src.agents.test_evaluator import TestEvaluatorAgent, TestEvaluatorInput
from src.agents.ranker import RankerAgent, RankerInput
from src.agents.bias_auditor import BiasAuditorAgent, BiasAuditInput
from src.schemas.job import JobDescription
from src.schemas.candidates import CandidateProfile, MatchResult
from src.schemas.messages import PipelineState


# Sample Job Description
SAMPLE_JD = """
Senior Python Developer

About the Role:
We are looking for an experienced Python developer to join our growing team. 
You will be responsible for designing and implementing scalable backend services.

Requirements:
- 5+ years of experience with Python
- Strong experience with FastAPI or Django
- Experience with PostgreSQL and Redis
- Knowledge of containerization (Docker, Kubernetes)
- Understanding of microservices architecture
- Strong communication skills and ability to work in a team

Nice to Have:
- Experience with machine learning
- AWS or GCP certification
- Open source contributions

Responsibilities:
- Design and build backend services
- Write clean, maintainable code
- Participate in code reviews
- Mentor junior developers
"""

# Sample Resume
SAMPLE_RESUME = """
John Doe
Senior Software Engineer

SUMMARY
Experienced software engineer with 7 years of Python development experience.
Passionate about building scalable systems and mentoring teams.

SKILLS
- Python (7 years): FastAPI, Django, Flask
- Databases: PostgreSQL, Redis, MongoDB
- Cloud: AWS (Certified Solutions Architect)
- DevOps: Docker, Kubernetes, CI/CD
- Other: Git, Linux, Agile

EXPERIENCE

Tech Company A | Senior Python Developer | 2020-Present
- Led development of microservices handling 1M+ requests/day
- Mentored team of 4 junior developers
- Implemented CI/CD pipelines reducing deployment time by 60%

Startup B | Python Developer | 2017-2020
- Built REST APIs using FastAPI and Django
- Designed PostgreSQL database schemas
- Contributed to open source projects

EDUCATION
MS Computer Science, Stanford University, 2017

CERTIFICATIONS
- AWS Solutions Architect (Professional)
- Kubernetes Administrator (CKA)
"""


def test_jd_analyzer():
    """Test the JD Analyzer agent."""
    print("\n" + "="*60)
    print("Testing JD Analyzer Agent")
    print("="*60)
    
    job = JobDescription(
        job_id="test_job_001",
        title="Senior Python Developer",
        company="Test Corp",
        raw_description=SAMPLE_JD,
    )
    
    agent = JDAnalyzerAgent()
    result = agent.run(job)
    response = result.response
    
    print(f"Status: {response.status}")
    print(f"Confidence: {response.confidence_score:.2f}")
    print(f"Explanation: {response.explanation}")
    
    if response.output:
        print(f"\nExtracted Skills: {len(response.output.skills)} total")
        required = [s for s in response.output.skills if s.importance == "required"]
        preferred = [s for s in response.output.skills if s.importance == "preferred"]
        print(f"Required: {[s.skill_name for s in required[:5]]}")
        print(f"Preferred: {[s.skill_name for s in preferred[:5]]}")
        print(f"Bias Flags: {response.output.potential_bias_flags}")
    
    return response.status.value == "success"


def test_resume_parser():
    """Test the Resume Parser agent."""
    print("\n" + "="*60)
    print("Testing Resume Parser Agent")
    print("="*60)
    
    agent = ResumeParserAgent()
    result = agent.run({
        "candidate_id": "candidate_001",
        "resume_text": SAMPLE_RESUME,
        "resume_format": "txt",
    })
    response = result.response
    
    print(f"Status: {response.status}")
    print(f"Confidence: {response.confidence_score:.2f}")
    print(f"Explanation: {response.explanation}")
    
    if response.output:
        print(f"\nExtracted Skills: {response.output.unique_skills_count}")
        print(f"Total Experience: {response.output.total_experience_months // 12} years")
        print(f"Quality Score: {response.output.resume_quality_score:.2f}")
        if response.output.parsing_warnings:
            print(f"Warnings: {response.output.parsing_warnings}")
    
    return response.status.value == "success"


def test_matcher():
    """Test the Matcher agent."""
    print("\n" + "="*60)
    print("Testing Matcher Agent")
    print("="*60)
    
    # First parse JD and Resume
    jd_agent = JDAnalyzerAgent()
    jd_result = jd_agent.run(JobDescription(
        job_id="test_job_001",
        title="Senior Python Developer",
        raw_description=SAMPLE_JD,
    ))
    
    resume_agent = ResumeParserAgent()
    resume_result = resume_agent.run({
        "candidate_id": "candidate_001",
        "resume_text": SAMPLE_RESUME,
        "resume_format": "txt",
    })
    
    jd_response = jd_result.response
    resume_response = resume_result.response
    
    if not jd_response.output or not resume_response.output:
        print("Could not parse JD or resume")
        return False
    
    # Now test matcher
    matcher = MatcherAgent()
    match_input = MatcherInput(
        candidate_id="candidate_001",
        parsed_resume=resume_response.output,
        parsed_jd=jd_response.output,
    )
    
    result = matcher.run(match_input)
    response = result.response
    
    print(f"Status: {response.status}")
    print(f"Overall Match Score: {response.output.overall_match_score:.2f}" if response.output else "No data")
    
    if response.output:
        print(f"Skills Match: {response.output.skills_match_score:.2f}")
        print(f"Experience Match: {response.output.experience_match_score:.2f}")
        print(f"Strengths: {response.output.strengths[:3]}")
        print(f"Gaps: {response.output.gaps[:3]}")
    
    return response.status.value == "success"


def test_test_generator():
    """Test the Test Generator agent."""
    print("\n" + "="*60)
    print("Testing Test Generator Agent")
    print("="*60)
    
    # First parse JD
    jd_agent = JDAnalyzerAgent()
    jd_result = jd_agent.run(JobDescription(
        job_id="test_job_001",
        title="Senior Python Developer",
        raw_description=SAMPLE_JD,
    ))
    jd_response = jd_result.response
    
    if not jd_response.output:
        print("Could not parse JD")
        return False
    
    # Generate test
    generator = TestGeneratorAgent()
    test_input = TestGeneratorInput(
        job_id="test_job_001",
        parsed_jd=jd_response.output,
        num_questions=5,
        difficulty="mixed",
    )
    
    result = generator.run(test_input)
    response = result.response
    
    print(f"Status: {response.status}")
    print(f"Confidence: {response.confidence_score:.2f}")
    
    if response.output:
        print(f"\nGenerated {len(response.output.questions)} questions:")
        for i, q in enumerate(response.output.questions[:3], 1):
            print(f"\n  Q{i}: {q.question_text[:80]}...")
            print(f"      Skill: {q.skill_tested}, Difficulty: {q.difficulty}")
    
    return response.status.value == "success"


def test_shortlister():
    """Test the Shortlister agent."""
    print("\n" + "="*60)
    print("Testing Shortlister Agent")
    print("="*60)
    
    # Create mock match results using actual MatchResult schema
    match_results = [
        MatchResult(
            candidate_id="candidate_001",
            job_id="test_job_001",
            overall_match_score=0.85,
            skills_match_score=0.90,
            experience_match_score=0.80,
            education_match_score=0.75,
            required_skills_met=5,
            required_skills_total=6,
            strengths=["Strong Python skills", "Relevant experience"],
            gaps=["Limited cloud experience"],
            bias_flags=[],
        ),
        MatchResult(
            candidate_id="candidate_002",
            job_id="test_job_001",
            overall_match_score=0.65,
            skills_match_score=0.60,
            experience_match_score=0.70,
            education_match_score=0.70,
            required_skills_met=3,
            required_skills_total=6,
            strengths=["Meets experience requirements"],
            gaps=["Missing key framework skills"],
            bias_flags=[],
        ),
        MatchResult(
            candidate_id="candidate_003",
            job_id="test_job_001",
            overall_match_score=0.50,
            skills_match_score=0.45,
            experience_match_score=0.55,
            education_match_score=0.60,
            required_skills_met=1,
            required_skills_total=6,
            strengths=[],
            gaps=["Does not meet minimum skill requirements"],
            bias_flags=[],
        ),
    ]
    
    shortlister = ShortlisterAgent()
    shortlist_input = ShortlistInput(
        match_results=match_results,
        threshold=0.7,
        max_candidates=10,
    )
    
    result = shortlister.run(shortlist_input)
    response = result.response
    
    print(f"Status: {response.status}")
    print(f"Confidence: {response.confidence_score:.2f}")
    
    if response.output:
        print(f"\nShortlisted: {len(response.output.shortlisted)} candidates")
        print(f"Rejected: {len(response.output.rejected)} candidates")
        for decision in response.output.decisions:
            status = "✓" if decision.passed else "✗"
            print(f"  {status} {decision.gate_name}: {decision.actual_value:.2f} vs {decision.threshold:.2f}")
    
    return response.status.value == "success"


def test_test_evaluator():
    """Test the Test Evaluator agent."""
    print("\n" + "="*60)
    print("Testing Test Evaluator Agent")
    print("="*60)
    
    # Create sample test questions
    from src.schemas.job import TestQuestion
    
    questions = [
        TestQuestion(
            question_id="q001",
            question_text="What is the output of print(type([]))?",
            options={"A": "<class 'list'>", "B": "list", "C": "[]", "D": "None"},
            correct_option="A",
            explanation="type() returns the type object, repr shows <class 'list'>",
            difficulty="easy",
            skill_tested="Python",
            topic="Data Types",
        ),
        TestQuestion(
            question_id="q002",
            question_text="What is the time complexity of dict lookup in Python?",
            options={"A": "O(n)", "B": "O(log n)", "C": "O(1)", "D": "O(n^2)"},
            correct_option="C",
            explanation="Dict uses hash table with O(1) average case lookup",
            difficulty="medium",
            skill_tested="Python",
            topic="Data Structures",
        ),
        TestQuestion(
            question_id="q003",
            question_text="Which decorator makes a method not require self?",
            options={"A": "@classmethod", "B": "@staticmethod", "C": "@property", "D": "@abstract"},
            correct_option="B",
            explanation="@staticmethod removes the implicit first argument",
            difficulty="medium",
            skill_tested="Python",
            topic="OOP",
        ),
    ]
    
    # Simulate candidate responses
    responses = [
        {"question_id": "q001", "selected_option": "A", "time_seconds": 15.0},
        {"question_id": "q002", "selected_option": "C", "time_seconds": 25.0},
        {"question_id": "q003", "selected_option": "A", "time_seconds": 20.0},  # Wrong answer
    ]
    
    evaluator = TestEvaluatorAgent()
    eval_input = TestEvaluatorInput(
        candidate_id="candidate_001",
        job_id="test_job_001",
        test_id="test_001",
        questions=questions,
        responses=responses,
    )
    
    result = evaluator.run(eval_input)
    response = result.response
    
    print(f"Status: {response.status}")
    print(f"Confidence: {response.confidence_score:.2f}")
    
    if response.output:
        print(f"\nScore: {response.output.questions_correct}/{response.output.questions_total}")
        print(f"Percentage: {response.output.total_score:.0%}")
        print(f"Total time: {response.output.total_time_seconds:.1f}s")
        print(f"Integrity flags: {len(response.output.integrity_flags)}")
    
    return response.status.value == "success"


def test_ranker():
    """Test the Ranker agent."""
    print("\n" + "="*60)
    print("Testing Ranker Agent")
    print("="*60)
    
    from src.schemas.candidates import TestResult
    
    # Create match results using actual MatchResult schema
    match_results = [
        MatchResult(
            candidate_id="candidate_001",
            job_id="test_job_001",
            overall_match_score=0.85,
            skills_match_score=0.90,
            experience_match_score=0.80,
            education_match_score=0.75,
            required_skills_met=5,
            required_skills_total=5,
            strengths=["Strong Python skills"],
            gaps=[],
            bias_flags=[],
        ),
        MatchResult(
            candidate_id="candidate_002",
            job_id="test_job_001",
            overall_match_score=0.75,
            skills_match_score=0.70,
            experience_match_score=0.80,
            education_match_score=0.75,
            required_skills_met=4,
            required_skills_total=5,
            strengths=["Good experience"],
            gaps=["Missing Django"],
            bias_flags=[],
        ),
    ]
    
    # Create test results
    test_results = [
        TestResult(
            candidate_id="candidate_001",
            job_id="test_job_001",
            test_id="test_001",
            total_score=0.90,
            questions_attempted=10,
            questions_correct=9,
            questions_total=10,
            category_scores={"Python": 0.90},
            responses=[],
            total_time_seconds=300.0,
            average_time_per_question=30.0,
            integrity_flags=[],
        ),
        TestResult(
            candidate_id="candidate_002",
            job_id="test_job_001",
            test_id="test_001",
            total_score=0.70,
            questions_attempted=10,
            questions_correct=7,
            questions_total=10,
            category_scores={"Python": 0.70},
            responses=[],
            total_time_seconds=350.0,
            average_time_per_question=35.0,
            integrity_flags=[],
        ),
    ]
    
    ranker = RankerAgent()
    ranker_input = RankerInput(
        job_id="test_job_001",
        match_results=match_results,
        test_results=test_results,
        weights={"resume": 0.4, "test": 0.6},
        top_k=5,
    )
    
    result = ranker.run(ranker_input)
    response = result.response
    
    print(f"Status: {response.status}")
    print(f"Confidence: {response.confidence_score:.2f}")
    
    if response.output:
        print(f"\nRankings ({response.output.total_candidates} candidates):")
        for ranking in response.output.rankings:
            print(f"  #{ranking.rank}: {ranking.candidate_id} - Score: {ranking.final_composite_score:.2f}")
            print(f"        Resume: {ranking.resume_match_score:.2f}, Test: {ranking.test_score:.2f}")
            print(f"        Recommendation: {ranking.recommendation}")
    
    return response.status.value == "success"


def test_bias_auditor():
    """Test the Bias Auditor agent."""
    print("\n" + "="*60)
    print("Testing Bias Auditor Agent")
    print("="*60)
    
    from src.schemas.candidates import FinalRanking
    from src.schemas.job import ParsedJD
    
    # Create a pipeline state with parsed JD using actual schema
    state = PipelineState()
    state.parsed_jd = ParsedJD(
        job_id="test_job_001",
        job_title_normalized="Senior Developer",
        seniority_level="senior",
        job_function="engineering",
        technical_topics=["Python", "Web Development"],
        jd_quality_score=0.8,
        potential_bias_flags=[],  # No bias in JD
    )
    
    # Create rankings
    rankings = [
        FinalRanking(
            candidate_id="candidate_001",
            job_id="test_job_001",
            rank=1,
            resume_match_score=0.85,
            test_score=0.90,
            final_composite_score=0.88,
            weights_used={"resume": 0.4, "test": 0.6},
            recommendation="strongly_recommend",
            confidence=0.85,
            ranking_explanation="Top performer in both resume and test",
            key_strengths=["Strong Python skills"],
            key_concerns=[],
            bias_audit_passed=True,
            human_review_required=False,
            human_review_reason="",
        ),
        FinalRanking(
            candidate_id="candidate_002",
            job_id="test_job_001",
            rank=2,
            resume_match_score=0.75,
            test_score=0.70,
            final_composite_score=0.72,
            weights_used={"resume": 0.4, "test": 0.6},
            recommendation="recommend",
            confidence=0.80,
            ranking_explanation="Good overall performance",
            key_strengths=["Good experience"],
            key_concerns=["Missing some skills"],
            bias_audit_passed=True,
            human_review_required=False,
            human_review_reason="",
        ),
    ]
    
    auditor = BiasAuditorAgent()
    audit_input = BiasAuditInput(
        pipeline_state=state,
        rankings=rankings,
        match_results=None,
    )
    
    result = auditor.run(audit_input)
    response = result.response
    
    print(f"Status: {response.status}")
    print(f"Confidence: {response.confidence_score:.2f}")
    if response.explanation:
        print(f"Explanation: {response.explanation}")
    
    if response.output:
        print(f"\nAudit Passed: {response.output.audit_passed}")
        print(f"Fairness Score: {response.output.overall_fairness_score:.2f}")
        print(f"Findings: {len(response.output.findings)}")
        print(f"Requires Human Review: {response.output.requires_human_review}")
        if response.output.recommendations:
            print(f"Recommendations: {response.output.recommendations[:2]}")
    
    return response.status.value == "success"


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Agentic Recruitment System - Test Suite")
    print("="*60)
    
    results = {
        "JD Analyzer": test_jd_analyzer(),
        "Resume Parser": test_resume_parser(),
        "Matcher": test_matcher(),
        "Test Generator": test_test_generator(),
        "Shortlister": test_shortlister(),
        "Test Evaluator": test_test_evaluator(),
        "Ranker": test_ranker(),
        "Bias Auditor": test_bias_auditor(),
    }
    
    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)
    
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name}: {status}")
    
    total_passed = sum(results.values())
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")
    
    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
