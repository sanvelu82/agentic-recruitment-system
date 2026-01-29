"""
Orchestrator Agent

Responsibility: Coordinate the entire recruitment pipeline.
Single purpose: Manage agent execution order, state, and decision flow.

This is the CENTRAL COORDINATOR that manages the agentic workflow.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

from .base import BaseAgent, AgentResponse, AgentStatus
from .jd_analyzer import JDAnalyzerAgent
from .resume_parser import ResumeParserAgent
from .matcher import MatcherAgent, MatcherInput
from .shortlister import ShortlisterAgent, ShortlistInput
from .test_generator import TestGeneratorAgent, TestGeneratorInput
from .test_evaluator import TestEvaluatorAgent, TestEvaluatorInput
from .ranker import RankerAgent, RankerInput
from .bias_auditor import BiasAuditorAgent, BiasAuditInput

from ..schemas.messages import (
    PipelineState,
    PipelineStage,
    AgentMessage,
    MessageType,
    DecisionGate,
)
from ..schemas.job import JobDescription
from ..schemas.candidates import CandidateProfile


class OrchestratorAction(str, Enum):
    """Actions the orchestrator can take."""
    CONTINUE = "continue"  # Proceed to next stage
    PAUSE = "pause"  # Wait for human input
    RETRY = "retry"  # Retry current stage
    ABORT = "abort"  # Stop the pipeline
    COMPLETE = "complete"  # Pipeline finished


@dataclass
class OrchestratorDecision:
    """Decision made by the orchestrator after each stage."""
    action: OrchestratorAction
    next_stage: Optional[PipelineStage] = None
    reason: str = ""
    requires_human_approval: bool = False
    blocked_by: List[str] = None  # Issues blocking progress


class OrchestratorAgent:
    """
    Central coordinator for the recruitment pipeline.
    
    The Orchestrator:
    - Manages pipeline state
    - Coordinates agent execution order
    - Handles decision gates
    - Ensures auditability
    - Manages human-in-the-loop checkpoints
    
    Design:
    - Framework-agnostic (can be wrapped by LangGraph, CrewAI, etc.)
    - Stateful (maintains pipeline state)
    - Event-driven (agents communicate via messages)
    - Fail-safe (handles errors gracefully)
    
    NOT responsible for:
    - Actual resume parsing (ResumeParser does that)
    - Scoring logic (Matcher, Ranker do that)
    - Bias detection (BiasAuditor does that)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize agents
        self.jd_analyzer = JDAnalyzerAgent()
        self.resume_parser = ResumeParserAgent()
        self.matcher = MatcherAgent()
        self.shortlister = ShortlisterAgent()
        self.test_generator = TestGeneratorAgent()
        self.test_evaluator = TestEvaluatorAgent()
        self.ranker = RankerAgent()
        self.bias_auditor = BiasAuditorAgent()
        
        # Pipeline state
        self.state: Optional[PipelineState] = None
        
        # Event handlers (for framework integration)
        self._event_handlers: Dict[str, List[Callable]] = {}
        
        # Audit log
        self._audit_log: List[Dict[str, Any]] = []
    
    def create_pipeline(self, job: JobDescription, candidates: List[CandidateProfile]) -> PipelineState:
        """
        Initialize a new recruitment pipeline.
        
        Args:
            job: The job description
            candidates: List of candidates to evaluate
        
        Returns:
            Initialized PipelineState
        """
        self.state = PipelineState(
            job_id=job.job_id,
            current_stage=PipelineStage.INITIALIZED,
            job_description=job.to_dict(),
            candidates=[c.to_dict() for c in candidates],
        )
        
        self._log_event("pipeline_created", {
            "job_id": job.job_id,
            "candidate_count": len(candidates),
        })
        
        return self.state
    
    def run_pipeline(self) -> PipelineState:
        """
        Execute the complete recruitment pipeline.
        
        Returns:
            Final pipeline state
        """
        if not self.state:
            raise ValueError("Pipeline not initialized. Call create_pipeline first.")
        
        self._log_event("pipeline_started", {"job_id": self.state.job_id})
        
        # Pipeline execution loop
        while self.state.current_stage not in [
            PipelineStage.COMPLETED,
            PipelineStage.FAILED,
            PipelineStage.AWAITING_HUMAN_REVIEW,
        ]:
            try:
                decision = self._execute_current_stage()
                
                if decision.action == OrchestratorAction.CONTINUE:
                    self.state.current_stage = decision.next_stage
                elif decision.action == OrchestratorAction.PAUSE:
                    self.state.current_stage = PipelineStage.AWAITING_HUMAN_REVIEW
                    break
                elif decision.action == OrchestratorAction.ABORT:
                    self.state.current_stage = PipelineStage.FAILED
                    self.state.errors.append(decision.reason)
                    break
                elif decision.action == OrchestratorAction.COMPLETE:
                    self.state.current_stage = PipelineStage.COMPLETED
                    break
                    
            except Exception as e:
                self._log_event("stage_error", {
                    "stage": self.state.current_stage.value,
                    "error": str(e),
                })
                self.state.errors.append(f"Error in {self.state.current_stage.value}: {str(e)}")
                self.state.current_stage = PipelineStage.FAILED
                break
        
        self._log_event("pipeline_finished", {
            "job_id": self.state.job_id,
            "final_stage": self.state.current_stage.value,
            "error_count": len(self.state.errors),
        })
        
        return self.state
    
    def _execute_current_stage(self) -> OrchestratorDecision:
        """Execute the current pipeline stage and decide next action."""
        stage = self.state.current_stage
        
        self._log_event("stage_started", {"stage": stage.value})
        
        if stage == PipelineStage.INITIALIZED:
            return self._run_jd_analysis()
        elif stage == PipelineStage.JD_ANALYSIS:
            return self._run_resume_parsing()
        elif stage == PipelineStage.RESUME_PARSING:
            return self._run_matching()
        elif stage == PipelineStage.MATCHING:
            return self._run_shortlisting()
        elif stage == PipelineStage.SHORTLISTING:
            return self._run_test_generation()
        elif stage == PipelineStage.TEST_GENERATION:
            # In real implementation, would wait for test completion
            return self._run_test_evaluation()
        elif stage == PipelineStage.TEST_EVALUATION:
            return self._run_ranking()
        elif stage == PipelineStage.RANKING:
            return self._run_bias_audit()
        elif stage == PipelineStage.BIAS_AUDIT:
            return OrchestratorDecision(
                action=OrchestratorAction.COMPLETE,
                reason="Pipeline completed successfully",
            )
        else:
            return OrchestratorDecision(
                action=OrchestratorAction.ABORT,
                reason=f"Unknown stage: {stage}",
            )
    
    def _run_jd_analysis(self) -> OrchestratorDecision:
        """Run JD analysis stage."""
        job_dict = self.state.job_description
        job = JobDescription(**{k: v for k, v in job_dict.items() 
                               if k in JobDescription.__dataclass_fields__})
        
        response = self.jd_analyzer.run(job)
        self._record_agent_response(response)
        
        if response.status == AgentStatus.SUCCESS:
            self.state.parsed_jd = response.data.to_dict() if response.data else {}
            return OrchestratorDecision(
                action=OrchestratorAction.CONTINUE,
                next_stage=PipelineStage.JD_ANALYSIS,
                reason="JD analysis completed",
            )
        else:
            return OrchestratorDecision(
                action=OrchestratorAction.ABORT,
                reason=f"JD analysis failed: {response.errors}",
            )
    
    def _run_resume_parsing(self) -> OrchestratorDecision:
        """Run resume parsing for all candidates."""
        parsed_resumes = []
        errors = []
        
        for candidate_dict in self.state.candidates:
            input_data = {
                "candidate_id": candidate_dict.get("candidate_id", ""),
                "resume_text": candidate_dict.get("resume_text", ""),
                "resume_format": candidate_dict.get("resume_format", "txt"),
            }
            
            # Skip if no resume text provided
            if not input_data["resume_text"]:
                self._log_event("resume_skipped", {
                    "candidate_id": input_data["candidate_id"],
                    "reason": "No resume text provided",
                })
                continue
            
            response = self.resume_parser.run(input_data)
            self._record_agent_response(response)
            
            if response.status == AgentStatus.SUCCESS and response.data:
                parsed_resumes.append({
                    "candidate_id": input_data["candidate_id"],
                    "parsed": response.data.to_dict(),
                })
            else:
                errors.append(f"Failed to parse resume for {input_data['candidate_id'][:8]}")
        
        # Store parsed resumes in state (update candidate entries)
        parsed_map = {pr["candidate_id"]: pr["parsed"] for pr in parsed_resumes}
        self.state.candidates = [
            {**c, "parsed_resume": parsed_map.get(c.get("candidate_id", ""), {})}
            for c in self.state.candidates
        ]
        
        if errors:
            self.state.warnings.extend(errors)
        
        return OrchestratorDecision(
            action=OrchestratorAction.CONTINUE,
            next_stage=PipelineStage.RESUME_PARSING,
            reason=f"Parsed {len(parsed_resumes)} of {len(self.state.candidates)} resumes",
        )
    
    def _run_matching(self) -> OrchestratorDecision:
        """Run resume-JD matching for all parsed candidates."""
        from ..schemas.candidates import MatchResult
        
        match_results = []
        parsed_jd = self.state.parsed_jd or {}
        
        for candidate_dict in self.state.candidates:
            parsed_resume = candidate_dict.get("parsed_resume", {})
            if not parsed_resume:
                continue
            
            # Prepare matcher input
            matcher_input = MatcherInput(
                candidate_id=candidate_dict.get("candidate_id", ""),
                parsed_resume=parsed_resume,
                parsed_jd=parsed_jd,
            )
            
            response = self.matcher.run(matcher_input)
            self._record_agent_response(response)
            
            if response.status == AgentStatus.SUCCESS and response.data:
                match_results.append(response.data)
        
        self.state.match_results = [m.to_dict() for m in match_results]
        
        return OrchestratorDecision(
            action=OrchestratorAction.CONTINUE,
            next_stage=PipelineStage.MATCHING,
            reason=f"Matched {len(match_results)} candidates against JD",
        )
    
    def _run_shortlisting(self) -> OrchestratorDecision:
        """Run shortlisting decision gate."""
        from ..schemas.candidates import MatchResult
        
        threshold = self.config.get("shortlist_threshold", 0.7)
        max_candidates = self.config.get("max_shortlist", 50)
        
        # Reconstruct MatchResult objects from stored dicts
        match_results = [
            MatchResult(**{k: v for k, v in m.items() if k in MatchResult.__dataclass_fields__})
            for m in self.state.match_results
        ]
        
        # Prepare shortlister input
        shortlist_input = ShortlistInput(
            match_results=match_results,
            threshold=threshold,
            max_candidates=max_candidates,
        )
        
        response = self.shortlister.run(shortlist_input)
        self._record_agent_response(response)
        
        if response.status == AgentStatus.SUCCESS and response.data:
            self.state.shortlisted_candidates = response.data.shortlisted
            
            # Add decision gates to state
            for decision in response.data.decisions:
                self.state.add_decision_gate(decision)
            
            # Check if too many borderline cases
            borderline = [d for d in response.data.decisions 
                         if any("borderline" in str(f).lower() for f in d.bias_flags)]
            if len(borderline) > len(match_results) * 0.25:
                return OrchestratorDecision(
                    action=OrchestratorAction.PAUSE,
                    reason=f"High borderline rate ({len(borderline)}/{len(match_results)}) - human review recommended",
                    requires_human_approval=True,
                )
        
        return OrchestratorDecision(
            action=OrchestratorAction.CONTINUE,
            next_stage=PipelineStage.SHORTLISTING,
            reason=f"Shortlisted {len(self.state.shortlisted_candidates)} candidates (threshold: {threshold})",
        )
    
    def _run_test_generation(self) -> OrchestratorDecision:
        """Generate assessment test."""
        from ..schemas.job import ParsedJD
        
        parsed_jd = self.state.parsed_jd or {}
        job_id = self.state.job_id
        
        # Number of questions based on config
        num_questions = self.config.get("test_questions", 10)
        difficulty = self.config.get("test_difficulty", "mixed")
        
        # Prepare test generator input
        test_input = TestGeneratorInput(
            job_id=job_id,
            parsed_jd=parsed_jd,
            num_questions=num_questions,
            difficulty=difficulty,
        )
        
        response = self.test_generator.run(test_input)
        self._record_agent_response(response)
        
        if response.status == AgentStatus.SUCCESS and response.data:
            # Store generated test
            self.state.test_questions = [q.to_dict() for q in response.data.questions]
            
            # Log test metadata
            self._log_event("test_generated", {
                "job_id": job_id,
                "test_id": response.data.test_id,
                "num_questions": len(response.data.questions),
                "categories": list(response.data.questions_by_category.keys()),
            })
        else:
            self.state.errors.append(f"Test generation failed: {response.errors}")
            return OrchestratorDecision(
                action=OrchestratorAction.PAUSE,
                reason="Test generation failed - manual test creation may be required",
                requires_human_approval=True,
            )
        
        return OrchestratorDecision(
            action=OrchestratorAction.CONTINUE,
            next_stage=PipelineStage.TEST_GENERATION,
            reason=f"Generated {len(self.state.test_questions)} test questions",
        )
    
    def _run_test_evaluation(self) -> OrchestratorDecision:
        """Evaluate test responses for all shortlisted candidates who completed tests."""
        from ..schemas.job import TestQuestion
        
        test_results = []
        job_id = self.state.job_id
        
        # Get test questions
        questions = [
            TestQuestion(**{k: v for k, v in q.items() if k in TestQuestion.__dataclass_fields__})
            for q in self.state.test_questions
        ]
        
        # Get candidate test responses from state
        # In real implementation, this would come from the test-taking service
        candidate_responses = self.state.candidate_test_responses or {}
        
        for candidate_id in self.state.shortlisted_candidates:
            responses = candidate_responses.get(candidate_id, [])
            
            if not responses:
                self._log_event("test_not_taken", {"candidate_id": candidate_id[:8]})
                continue
            
            # Prepare evaluator input
            eval_input = TestEvaluatorInput(
                candidate_id=candidate_id,
                job_id=job_id,
                test_id=f"test_{job_id}",
                questions=questions,
                responses=responses,
            )
            
            response = self.test_evaluator.run(eval_input)
            self._record_agent_response(response)
            
            if response.status == AgentStatus.SUCCESS and response.data:
                test_results.append(response.data)
        
        self.state.test_results = [r.to_dict() for r in test_results]
        
        # Apply test passing threshold
        test_threshold = self.config.get("test_passing_score", 0.6)
        passed = [r for r in test_results if r.total_score >= test_threshold]
        
        # Create decision gate
        gate = DecisionGate(
            gate_name="test_evaluation_gate",
            condition=f"Test score >= {test_threshold}",
            threshold=test_threshold,
            passed=len(passed) > 0,
            explanation=f"{len(passed)} of {len(test_results)} candidates passed the test threshold",
        )
        self.state.add_decision_gate(gate)
        
        return OrchestratorDecision(
            action=OrchestratorAction.CONTINUE,
            next_stage=PipelineStage.TEST_EVALUATION,
            reason=f"Evaluated {len(test_results)} tests, {len(passed)} passed threshold ({test_threshold})",
        )
    
    def _run_ranking(self) -> OrchestratorDecision:
        """Produce final rankings."""
        from ..schemas.candidates import MatchResult, TestResult
        
        # Reconstruct MatchResult objects
        match_results = [
            MatchResult(**{k: v for k, v in m.items() if k in MatchResult.__dataclass_fields__})
            for m in self.state.match_results
        ]
        
        # Reconstruct TestResult objects
        test_results = [
            TestResult(**{k: v for k, v in t.items() if k in TestResult.__dataclass_fields__})
            for t in self.state.test_results
        ]
        
        # Filter to only shortlisted candidates who took tests
        shortlisted_ids = set(self.state.shortlisted_candidates)
        test_taker_ids = {t.candidate_id for t in test_results}
        
        match_results = [m for m in match_results if m.candidate_id in shortlisted_ids]
        test_results = [t for t in test_results if t.candidate_id in test_taker_ids]
        
        # Get ranking config
        top_k = self.config.get("top_k_candidates", 10)
        weights = self.config.get("ranking_weights", {"resume": 0.5, "test": 0.5})
        
        # Prepare ranker input
        ranker_input = RankerInput(
            job_id=self.state.job_id,
            match_results=match_results,
            test_results=test_results,
            weights=weights,
            top_k=top_k,
        )
        
        response = self.ranker.run(ranker_input)
        self._record_agent_response(response)
        
        if response.status == AgentStatus.SUCCESS and response.data:
            self.state.final_rankings = [r.to_dict() for r in response.data.rankings]
            
            # Log ranking summary
            self._log_event("ranking_complete", {
                "job_id": self.state.job_id,
                "total_ranked": response.data.total_candidates,
                "top_k": len(response.data.rankings),
                "methodology": response.data.ranking_methodology,
            })
            
            # Check if top candidates need human review
            needs_review = [r for r in response.data.rankings if r.human_review_required]
            if needs_review:
                self.state.warnings.append(
                    f"{len(needs_review)} top candidates flagged for human review"
                )
        else:
            self.state.errors.append("Ranking failed")
        
        return OrchestratorDecision(
            action=OrchestratorAction.CONTINUE,
            next_stage=PipelineStage.RANKING,
            reason=f"Ranked top {len(self.state.final_rankings)} candidates",
        )
    
    def _run_bias_audit(self) -> OrchestratorDecision:
        """Run final bias audit."""
        from ..schemas.candidates import FinalRanking, MatchResult
        
        # Reconstruct ranking objects
        rankings = [
            FinalRanking(**{k: v for k, v in r.items() if k in FinalRanking.__dataclass_fields__})
            for r in self.state.final_rankings
        ]
        
        # Reconstruct match results for deeper analysis
        match_results = [
            MatchResult(**{k: v for k, v in m.items() if k in MatchResult.__dataclass_fields__})
            for m in self.state.match_results
        ]
        
        # Prepare bias audit input
        audit_input = BiasAuditInput(
            pipeline_state=self.state,
            rankings=rankings,
            match_results=match_results,
        )
        
        response = self.bias_auditor.run(audit_input)
        self._record_agent_response(response)
        
        if response.status == AgentStatus.SUCCESS and response.data:
            self.state.bias_audit_results = {
                "audit_passed": response.data.audit_passed,
                "fairness_score": response.data.overall_fairness_score,
                "findings": response.data.findings,
                "recommendations": response.data.recommendations,
                "requires_human_review": response.data.requires_human_review,
                "compliance_notes": response.data.compliance_notes,
            }
            
            # Log audit results
            self._log_event("bias_audit_complete", {
                "job_id": self.state.job_id,
                "audit_passed": response.data.audit_passed,
                "fairness_score": response.data.overall_fairness_score,
                "findings_count": len(response.data.findings),
            })
            
            # If audit fails, require human review
            if not response.data.audit_passed:
                return OrchestratorDecision(
                    action=OrchestratorAction.PAUSE,
                    reason=f"Bias audit failed (fairness score: {response.data.overall_fairness_score:.0%}). "
                           f"Found {len(response.data.findings)} issues requiring review.",
                    requires_human_approval=True,
                )
        else:
            self.state.errors.append("Bias audit failed to execute")
            return OrchestratorDecision(
                action=OrchestratorAction.PAUSE,
                reason="Bias audit could not be completed - human review required",
                requires_human_approval=True,
            )
        
        return OrchestratorDecision(
            action=OrchestratorAction.CONTINUE,
            next_stage=PipelineStage.BIAS_AUDIT,
            reason=f"Bias audit {'PASSED' if response.data.audit_passed else 'REQUIRES REVIEW'} "
                   f"(fairness: {response.data.overall_fairness_score:.0%})",
        )
    
    def _record_agent_response(self, response: AgentResponse) -> None:
        """Record an agent's response in the pipeline state."""
        self.state.add_agent_response(response.to_dict())
        
        if response.warnings:
            self.state.warnings.extend(response.warnings)
        
        self._log_event("agent_response", {
            "agent_type": response.agent_type,
            "status": response.status.value,
            "confidence": response.confidence,
            "requires_review": response.requires_human_review,
        })
    
    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log an orchestrator event for auditing."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "pipeline_id": self.state.pipeline_id if self.state else None,
            "data": data,
        }
        self._audit_log.append(event)
        
        # Emit to registered handlers
        for handler in self._event_handlers.get(event_type, []):
            handler(event)
    
    def on_event(self, event_type: str, handler: Callable) -> None:
        """Register an event handler for framework integration."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
    
    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get the complete audit log."""
        return self._audit_log.copy()
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get a summary of the current pipeline state."""
        if not self.state:
            return {"status": "not_initialized"}
        
        return {
            "pipeline_id": self.state.pipeline_id,
            "job_id": self.state.job_id,
            "current_stage": self.state.current_stage.value,
            "candidate_count": len(self.state.candidates),
            "shortlisted_count": len(self.state.shortlisted_candidates),
            "final_rankings_count": len(self.state.final_rankings),
            "error_count": len(self.state.errors),
            "warning_count": len(self.state.warnings),
            "decision_gates_passed": sum(
                1 for g in self.state.decision_gates if g.get("passed", False)
            ),
            "requires_human_review": any(
                r.get("requires_human_review", False) 
                for r in self.state.agent_responses
            ),
        }
