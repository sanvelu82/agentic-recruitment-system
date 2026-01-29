"""
Ranker Agent

Responsibility: Rank candidates based on composite scores.
Single purpose: Produce final ranked list with explanations.

This agent combines all evaluation data to produce rankings.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .base import BaseAgent
from ..schemas.candidates import MatchResult, TestResult, FinalRanking


@dataclass
class RankerInput:
    """Input for the ranker agent."""
    job_id: str
    match_results: List[MatchResult]
    test_results: List[TestResult]
    weights: Dict[str, float] = None  # e.g., {"resume": 0.5, "test": 0.5}
    top_k: int = 10
    
    def __post_init__(self):
        if self.weights is None:
            self.weights = {"resume": 0.5, "test": 0.5}


@dataclass
class RankerOutput:
    """Output from the ranker agent."""
    job_id: str
    rankings: List[FinalRanking]
    total_candidates: int
    ranking_methodology: str


class RankerAgent(BaseAgent[RankerInput, RankerOutput]):
    """
    Produces final candidate rankings based on all evaluation data.
    
    Input: Match results + Test results + ranking parameters
    Output: Ranked list of candidates with explanations
    
    Key responsibilities:
    - Combine multiple score components
    - Apply configurable weights
    - Provide clear ranking explanations
    - Flag candidates needing human review
    
    IMPORTANT:
    - This produces RECOMMENDATIONS, not final decisions
    - Human review is required for hiring decisions
    - Rankings must be explainable and auditable
    """
    
    @property
    def description(self) -> str:
        return (
            "Ranks candidates by combining match and test scores with "
            "configurable weights, producing explainable recommendations."
        )
    
    @property
    def required_confidence_threshold(self) -> float:
        return 0.8  # Rankings need higher confidence
    
    def run(
        self,
        input_data: RankerInput,
        state: Optional["PipelineState"] = None
    ) -> "AgentResult[RankerOutput]":
        """
        Execute ranking and return result.
        
        Args:
            input_data: RankerInput with all evaluation data
            state: Optional pipeline state
        
        Returns:
            AgentResult with RankerOutput and updated state
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
                explanation=f"Ranking failed: {str(e)}",
            )
            return AgentResult(response=response, state=state)
    
    def _process(
        self, 
        input_data: RankerInput
    ) -> tuple[RankerOutput, float, str]:
        """
        Rank candidates based on combined scores.
        
        Args:
            input_data: RankerInput with all evaluation data
        
        Returns:
            RankerOutput, confidence_score, explanation
        """
        self.log_reasoning(f"Ranking candidates for job {input_data.job_id[:8]}")
        self.log_reasoning(f"Weights: {input_data.weights}")
        
        # Build lookup maps
        match_map = {m.candidate_id: m for m in input_data.match_results}
        test_map = {t.candidate_id: t for t in input_data.test_results}
        
        # Calculate composite scores
        scores = []
        for candidate_id in match_map.keys():
            match_result = match_map.get(candidate_id)
            test_result = test_map.get(candidate_id)
            
            if not match_result:
                continue
            
            resume_score = match_result.overall_match_score
            test_score = test_result.total_score if test_result else 0.0
            
            # Calculate weighted composite
            weights = input_data.weights
            composite = (
                resume_score * weights.get("resume", 0.5) +
                test_score * weights.get("test", 0.5)
            )
            
            scores.append({
                "candidate_id": candidate_id,
                "resume_score": resume_score,
                "test_score": test_score,
                "composite": composite,
                "match_result": match_result,
                "test_result": test_result,
            })
        
        # Sort by composite score
        scores.sort(key=lambda x: x["composite"], reverse=True)
        
        # Build rankings
        rankings = []
        for rank, data in enumerate(scores[:input_data.top_k], start=1):
            # Determine recommendation
            composite = data["composite"]
            if composite >= 0.85:
                recommendation = "strongly_recommend"
            elif composite >= 0.7:
                recommendation = "recommend"
            elif composite >= 0.5:
                recommendation = "consider"
            else:
                recommendation = "not_recommended"
            
            # Identify strengths and concerns
            strengths = data["match_result"].strengths[:3]  # Top 3
            concerns = data["match_result"].gaps[:3]
            
            if data["test_result"] and data["test_result"].integrity_flags:
                concerns.append("Test integrity flags present")
            
            ranking = FinalRanking(
                candidate_id=data["candidate_id"],
                job_id=input_data.job_id,
                rank=rank,
                resume_match_score=data["resume_score"],
                test_score=data["test_score"],
                final_composite_score=composite,
                weights_used=input_data.weights,
                recommendation=recommendation,
                confidence=0.85,  # TODO: Calculate based on data quality
                ranking_explanation=(
                    f"Rank {rank}: Composite score {composite:.2f} "
                    f"(Resume: {data['resume_score']:.2f}, Test: {data['test_score']:.2f})"
                ),
                key_strengths=strengths,
                key_concerns=concerns,
                bias_audit_passed=True,  # Will be set by BiasAuditor
                human_review_required=recommendation == "consider" or len(concerns) > 2,
                human_review_reason="Borderline candidate or multiple concerns" if len(concerns) > 2 else "",
            )
            rankings.append(ranking)
            
            self.log_reasoning(
                f"Rank {rank}: Candidate {data['candidate_id'][:8]} - "
                f"Score: {composite:.2f} - {recommendation}"
            )
        
        output = RankerOutput(
            job_id=input_data.job_id,
            rankings=rankings,
            total_candidates=len(scores),
            ranking_methodology=(
                f"Weighted composite: {input_data.weights}. "
                f"Sorted by composite score descending. "
                f"Top {input_data.top_k} selected."
            ),
        )
        
        confidence = 0.85
        explanation = (
            f"Ranked {len(rankings)} candidates from {len(scores)} total. "
            f"Top candidate score: {rankings[0].final_composite_score:.2f}. "
            f"{sum(1 for r in rankings if r.human_review_required)} require human review."
        )
        
        return output, confidence, explanation
