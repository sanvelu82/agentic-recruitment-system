"""
Shortlister Agent

Responsibility: Apply threshold-based filtering to matched candidates.
Single purpose: Decide which candidates proceed to testing.

This is a DECISION GATE agent - it makes pass/fail decisions with explanations.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .base import BaseAgent
from ..schemas.candidates import MatchResult
from ..schemas.messages import DecisionGate


@dataclass
class ShortlistInput:
    """Input for the shortlister agent."""
    match_results: List[MatchResult]
    threshold: float = 0.7
    max_candidates: int = 50  # Maximum to shortlist for testing


@dataclass
class ShortlistOutput:
    """Output from the shortlister agent."""
    shortlisted: List[str]  # Candidate IDs
    rejected: List[str]  # Candidate IDs
    decisions: List[DecisionGate]  # Decision explanation per candidate


class ShortlisterAgent(BaseAgent[ShortlistInput, ShortlistOutput]):
    """
    Applies threshold filtering to decide which candidates proceed.
    
    Input: List of MatchResults + threshold
    Output: Lists of shortlisted and rejected candidates with explanations
    
    Key responsibilities:
    - Apply configurable threshold
    - Provide clear decision rationale
    - Flag borderline cases for human review
    - Ensure fairness in filtering
    
    Does NOT:
    - Rank candidates (that's the Ranker's job)
    - Generate tests
    - Make final hiring decisions
    """
    
    @property
    def description(self) -> str:
        return (
            "Filters candidates based on match scores, applying threshold-based "
            "decisions with full transparency and bias monitoring."
        )
    
    @property
    def required_confidence_threshold(self) -> float:
        return 0.8  # Higher threshold for decision-making agents
    
    def run(
        self,
        input_data: ShortlistInput,
        state: Optional["PipelineState"] = None
    ) -> "AgentResult[ShortlistOutput]":
        """
        Execute shortlisting and return result.
        
        Args:
            input_data: ShortlistInput with match results and threshold
            state: Optional pipeline state
        
        Returns:
            AgentResult with ShortlistOutput and updated state
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
                explanation=f"Shortlisting failed: {str(e)}",
            )
            return AgentResult(response=response, state=state)
    
    def _process(
        self, 
        input_data: ShortlistInput
    ) -> tuple[ShortlistOutput, float, str]:
        """
        Filter candidates based on match threshold.
        
        Args:
            input_data: ShortlistInput with match results and threshold
        
        Returns:
            ShortlistOutput, confidence_score, explanation
        """
        threshold = input_data.threshold
        max_candidates = input_data.max_candidates
        
        self.log_reasoning(f"Applying shortlist threshold: {threshold:.2f}")
        self.log_reasoning(f"Processing {len(input_data.match_results)} candidates")
        
        shortlisted = []
        rejected = []
        decisions = []
        borderline_count = 0
        
        for match in input_data.match_results:
            score = match.overall_match_score
            
            # Create decision gate for this candidate
            gate = DecisionGate(
                gate_name="shortlist_threshold",
                condition=f"Match score >= {threshold}",
                threshold=threshold,
                actual_value=score,
                passed=score >= threshold,
            )
            
            # Check for borderline cases (within 10% of threshold)
            is_borderline = abs(score - threshold) < 0.1
            if is_borderline:
                borderline_count += 1
                gate.bias_flags.append("borderline_case_requires_review")
            
            # Add bias flags from match result
            gate.bias_flags.extend(match.bias_flags)
            
            if score >= threshold:
                shortlisted.append(match.candidate_id)
                gate.explanation = (
                    f"Candidate shortlisted with score {score:.2f} "
                    f"(threshold: {threshold:.2f})"
                )
            else:
                rejected.append(match.candidate_id)
                gate.explanation = (
                    f"Candidate not shortlisted. Score {score:.2f} "
                    f"below threshold {threshold:.2f}"
                )
            
            decisions.append(gate)
        
        # Apply max candidates limit if needed
        if len(shortlisted) > max_candidates:
            self.log_reasoning(
                f"Shortlist ({len(shortlisted)}) exceeds max ({max_candidates}). "
                "Recommend raising threshold or human review."
            )
        
        output = ShortlistOutput(
            shortlisted=shortlisted,
            rejected=rejected,
            decisions=decisions,
        )
        
        # Calculate confidence based on borderline cases
        if len(input_data.match_results) > 0:
            borderline_ratio = borderline_count / len(input_data.match_results)
            confidence = max(0.5, 1.0 - borderline_ratio)
        else:
            confidence = 0.5
        
        explanation = (
            f"Shortlisted {len(shortlisted)} of {len(input_data.match_results)} "
            f"candidates using threshold {threshold:.2f}. "
            f"{borderline_count} borderline cases flagged for review."
        )
        
        self.log_reasoning(explanation)
        
        return output, confidence, explanation
