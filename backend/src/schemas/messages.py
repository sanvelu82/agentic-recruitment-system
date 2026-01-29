"""
Core message schemas for inter-agent communication.

All agents communicate using these standardized message formats.
This ensures consistency, traceability, and framework-agnostic operation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4


class MessageType(str, Enum):
    """Types of messages in the agent pipeline."""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    DECISION_GATE = "decision_gate"
    HUMAN_REVIEW_REQUEST = "human_review_request"
    PIPELINE_STATE = "pipeline_state"
    AUDIT_EVENT = "audit_event"


class PipelineStage(str, Enum):
    """Stages in the recruitment pipeline."""
    INITIALIZED = "initialized"
    JD_ANALYSIS = "jd_analysis"
    RESUME_PARSING = "resume_parsing"
    MATCHING = "matching"
    SHORTLISTING = "shortlisting"
    TEST_GENERATION = "test_generation"
    TEST_EVALUATION = "test_evaluation"
    RANKING = "ranking"
    BIAS_AUDIT = "bias_audit"
    COMPLETED = "completed"
    FAILED = "failed"
    AWAITING_HUMAN_REVIEW = "awaiting_human_review"


@dataclass
class AgentMessage:
    """
    Base message structure for all inter-agent communication.
    
    This is the envelope that wraps all data flowing between agents.
    """
    message_id: str = field(default_factory=lambda: uuid4().hex)
    message_type: MessageType = MessageType.TASK_REQUEST
    source_agent: str = ""
    target_agent: str = ""
    correlation_id: str = ""  # Links related messages across the pipeline
    timestamp: datetime = field(default_factory=datetime.utcnow)
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
            "metadata": self.metadata,
        }


@dataclass
class TaskRequest:
    """
    Request sent to an agent to perform a task.
    
    Attributes:
        task_id: Unique task identifier
        job_id: The job posting this task relates to
        task_type: What operation to perform
        input_data: The data to process
        context: Additional context from previous stages
        priority: Task priority (1=highest, 5=lowest)
        timeout_seconds: Max time for task completion
    """
    task_id: str = field(default_factory=lambda: uuid4().hex)
    job_id: str = ""
    task_type: str = ""
    input_data: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    priority: int = 3
    timeout_seconds: int = 300

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "job_id": self.job_id,
            "task_type": self.task_type,
            "input_data": self.input_data,
            "context": self.context,
            "priority": self.priority,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass
class TaskResponse:
    """
    Response from an agent after completing a task.
    
    Attributes:
        task_id: The task this responds to
        success: Whether the task completed successfully
        output_data: The result of the task
        confidence: How confident the agent is in the result [0.0-1.0]
        explanation: Human-readable explanation
        next_action: Suggested next step in the pipeline
        requires_human_review: Whether a human should review this
        audit_entries: Detailed audit trail
    """
    task_id: str = ""
    success: bool = False
    output_data: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    explanation: str = ""
    next_action: Optional[str] = None
    requires_human_review: bool = False
    audit_entries: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "output_data": self.output_data,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "next_action": self.next_action,
            "requires_human_review": self.requires_human_review,
            "audit_entries": self.audit_entries,
            "errors": self.errors,
            "warnings": self.warnings,
            "processing_time_ms": self.processing_time_ms,
        }


@dataclass
class DecisionGate:
    """
    A decision point in the pipeline where conditions are evaluated.
    
    Decision gates ensure that candidates meet criteria before proceeding.
    They also flag potential bias issues for review.
    """
    gate_id: str = field(default_factory=lambda: uuid4().hex)
    gate_name: str = ""
    condition: str = ""  # Human-readable condition
    threshold: float = 0.0
    actual_value: float = 0.0
    passed: bool = False
    explanation: str = ""
    bias_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "gate_name": self.gate_name,
            "condition": self.condition,
            "threshold": self.threshold,
            "actual_value": self.actual_value,
            "passed": self.passed,
            "explanation": self.explanation,
            "bias_flags": self.bias_flags,
        }


@dataclass
class PipelineState:
    """
    Complete state of the recruitment pipeline for a job posting.
    
    This is the central state object that flows through the system.
    It maintains the complete history of all agent interactions.
    """
    pipeline_id: str = field(default_factory=lambda: uuid4().hex)
    job_id: str = ""
    current_stage: PipelineStage = PipelineStage.INITIALIZED
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Accumulated data from each stage
    job_description: Dict[str, Any] = field(default_factory=dict)
    parsed_jd: Dict[str, Any] = field(default_factory=dict)
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    match_results: List[Dict[str, Any]] = field(default_factory=list)
    shortlisted_candidates: List[str] = field(default_factory=list)
    test_questions: List[Dict[str, Any]] = field(default_factory=list)
    candidate_test_responses: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)  # candidate_id -> responses
    test_results: List[Dict[str, Any]] = field(default_factory=list)
    final_rankings: List[Dict[str, Any]] = field(default_factory=list)
    
    # Audit and compliance
    decision_gates: List[Dict[str, Any]] = field(default_factory=list)
    bias_audit_results: Dict[str, Any] = field(default_factory=dict)
    human_review_notes: List[str] = field(default_factory=list)
    
    # Agent responses history
    agent_responses: List[Dict[str, Any]] = field(default_factory=list)
    
    # Error tracking
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "job_id": self.job_id,
            "current_stage": self.current_stage.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "job_description": self.job_description,
            "parsed_jd": self.parsed_jd,
            "candidates": self.candidates,
            "match_results": self.match_results,
            "shortlisted_candidates": self.shortlisted_candidates,
            "test_questions": self.test_questions,
            "candidate_test_responses": self.candidate_test_responses,
            "test_results": self.test_results,
            "final_rankings": self.final_rankings,
            "decision_gates": self.decision_gates,
            "bias_audit_results": self.bias_audit_results,
            "human_review_notes": self.human_review_notes,
            "agent_responses": self.agent_responses,
            "errors": self.errors,
            "warnings": self.warnings,
        }

    def add_agent_response(self, response: Dict[str, Any]) -> None:
        """Record an agent's response to the pipeline history."""
        self.agent_responses.append(response)
        self.updated_at = datetime.utcnow()

    def add_decision_gate(self, gate: DecisionGate) -> None:
        """Record a decision gate evaluation."""
        self.decision_gates.append(gate.to_dict())
        self.updated_at = datetime.utcnow()
