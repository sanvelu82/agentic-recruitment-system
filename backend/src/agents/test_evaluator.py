"""
Test Evaluator Agent

Responsibility: Score candidate test responses.
Single purpose: Evaluate answers and calculate test scores.

This agent does NOT rank candidates - only evaluates individual tests.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .base import BaseAgent
from ..schemas.candidates import TestResult, TestResponse
from ..schemas.job import TestQuestion


@dataclass
class TestEvaluatorInput:
    """Input for test evaluation."""
    candidate_id: str
    job_id: str
    test_id: str
    questions: List[TestQuestion]
    responses: List[Dict[str, Any]]  # {question_id, selected_option, time_seconds}


class TestEvaluatorAgent(BaseAgent[TestEvaluatorInput, TestResult]):
    """
    Evaluates candidate test responses and calculates scores.
    
    Input: Test questions + candidate responses
    Output: TestResult with scores and analysis
    
    Key responsibilities:
    - Score each answer correctly
    - Calculate category-wise scores
    - Flag integrity concerns (unusual patterns)
    - Provide transparent scoring
    
    Does NOT:
    - Generate questions
    - Rank candidates
    - Make hiring decisions
    """
    
    @property
    def description(self) -> str:
        return (
            "Evaluates candidate test responses, calculating scores with "
            "full transparency and integrity monitoring."
        )
    
    def run(
        self,
        input_data: TestEvaluatorInput,
        state: Optional["PipelineState"] = None
    ) -> "AgentResult[TestResult]":
        """
        Execute test evaluation and return result.
        
        Args:
            input_data: TestEvaluatorInput with questions and responses
            state: Optional pipeline state
        
        Returns:
            AgentResult with TestResult and updated state
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
                explanation=f"Test evaluation failed: {str(e)}",
            )
            return AgentResult(response=response, state=state)
    
    def _process(
        self, 
        input_data: TestEvaluatorInput
    ) -> tuple[TestResult, float, str]:
        """
        Evaluate test responses.
        
        Args:
            input_data: TestEvaluatorInput with questions and responses
        
        Returns:
            TestResult, confidence_score, explanation
        """
        self.log_reasoning(
            f"Evaluating test {input_data.test_id[:8]} "
            f"for candidate {input_data.candidate_id[:8]}"
        )
        
        # Build question lookup
        question_map = {q.question_id: q for q in input_data.questions}
        
        # Evaluate each response
        evaluated_responses = []
        correct_count = 0
        total_time = 0.0
        category_scores: Dict[str, List[float]] = {}
        integrity_flags = []
        
        for resp in input_data.responses:
            question_id = resp.get("question_id", "")
            selected = resp.get("selected_option", "")
            time_seconds = resp.get("time_seconds", 0.0)
            
            question = question_map.get(question_id)
            if not question:
                self.log_reasoning(f"Unknown question ID: {question_id}")
                continue
            
            is_correct = selected == question.correct_option
            if is_correct:
                correct_count += 1
            
            # Track category scores
            category = question.skill_tested
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(1.0 if is_correct else 0.0)
            
            # Check for integrity concerns
            if time_seconds < 5:  # Too fast
                integrity_flags.append(
                    f"Question {question_id[:8]}: answered in {time_seconds}s (suspiciously fast)"
                )
            
            total_time += time_seconds
            
            evaluated_responses.append(TestResponse(
                question_id=question_id,
                selected_option=selected,
                response_time_seconds=time_seconds,
                is_correct=is_correct,
                partial_credit=1.0 if is_correct else 0.0,
            ))
        
        # Calculate final scores
        total_questions = len(input_data.questions)
        questions_attempted = len(evaluated_responses)
        total_score = correct_count / total_questions if total_questions > 0 else 0.0
        
        # Calculate per-category scores
        category_scores_final = {
            cat: sum(scores) / len(scores) if scores else 0.0
            for cat, scores in category_scores.items()
        }
        
        self.log_reasoning(f"Score: {correct_count}/{total_questions} ({total_score:.0%})")
        
        if integrity_flags:
            self.log_reasoning(f"Integrity flags: {len(integrity_flags)}")
        
        result = TestResult(
            candidate_id=input_data.candidate_id,
            job_id=input_data.job_id,
            test_id=input_data.test_id,
            total_score=total_score,
            questions_attempted=questions_attempted,
            questions_correct=correct_count,
            questions_total=total_questions,
            category_scores=category_scores_final,
            responses=evaluated_responses,
            total_time_seconds=total_time,
            average_time_per_question=total_time / questions_attempted if questions_attempted > 0 else 0.0,
            integrity_flags=integrity_flags,
        )
        
        confidence = 0.95  # Scoring is deterministic
        explanation = (
            f"Candidate scored {correct_count}/{total_questions} ({total_score:.0%}). "
            f"Time: {total_time:.0f}s. "
            f"{'No integrity concerns.' if not integrity_flags else f'{len(integrity_flags)} integrity flags raised.'}"
        )
        
        return result, confidence, explanation
