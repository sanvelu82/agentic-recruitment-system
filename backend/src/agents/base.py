"""
Base Agent Contracts for the Agentic Recruitment System.

This module defines the foundational abstractions that ALL agents must follow.
It establishes a consistent interface for agent communication, state management,
and pipeline coordination.

Design Principles:
    1. Single Responsibility: Each agent does exactly one thing well
    2. Strong Typing: All inputs/outputs are typed dataclasses, not raw strings
    3. Immutability: PipelineState is immutable; updates return new instances
    4. Framework-Agnostic: No LangChain/CrewAI/etc. dependencies
    5. Explainability: Every decision includes confidence and explanation
    6. Auditability: All operations are traceable via metadata

Usage:
    >>> class MyAgent(BaseAgent[MyInput, MyOutput]):
    ...     name = "my_agent"
    ...     description = "Does something specific"
    ...     
    ...     def run(self, input_data: MyInput, state: PipelineState) -> AgentResult[MyOutput]:
    ...         # Process input, return result with updated state
    ...         pass
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar
from uuid import uuid4
from uuid import uuid4


# =============================================================================
# TYPE VARIABLES
# =============================================================================

InputT = TypeVar("InputT")   # Agent input type
OutputT = TypeVar("OutputT") # Agent output type


# =============================================================================
# ENUMS
# =============================================================================

class AgentStatus(str, Enum):
    """
    Outcome status of an agent's execution.
    
    Using str, Enum for JSON serialization compatibility.
    """
    SUCCESS = "success"   # Agent completed successfully
    FAILURE = "failure"   # Agent failed, cannot proceed
    RETRY = "retry"       # Agent failed but retry may succeed (transient error)


# =============================================================================
# CORE DATA CONTRACTS
# =============================================================================

@dataclass(frozen=True)
class AgentResponse(Generic[OutputT]):
    """
    Standardized response returned by every agent.
    
    This is the universal output contract. All agents return this structure,
    enabling consistent handling by the orchestrator and downstream consumers.
    
    Attributes:
        agent_name: Identifier of the agent that produced this response
        status: Execution outcome (success/failure/retry)
        output: The actual result data, strongly typed per agent
        confidence_score: Agent's confidence in the output [0.0, 1.0]
            - 1.0 = Fully confident (deterministic result)
            - 0.7+ = High confidence (proceed automatically)
            - 0.5-0.7 = Medium confidence (may need review)
            - <0.5 = Low confidence (requires human review)
        explanation: Human-readable reasoning for the output
        metadata: Optional diagnostics (timing, model info, debug data)
    
    Design Notes:
        - frozen=True makes this immutable (hashable, thread-safe)
        - Generic[OutputT] enables type-safe output access
        - All fields have sensible defaults except agent_name and status
    
    Example:
        >>> response = AgentResponse(
        ...     agent_name="resume_parser",
        ...     status=AgentStatus.SUCCESS,
        ...     output=ParsedResume(...),
        ...     confidence_score=0.92,
        ...     explanation="Successfully extracted 12 skills and 3 experiences"
        ... )
    """
    agent_name: str
    status: AgentStatus
    output: Optional[OutputT] = None
    confidence_score: float = 0.0
    explanation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate invariants after initialization."""
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError(
                f"confidence_score must be in [0.0, 1.0], got {self.confidence_score}"
            )
    
    def is_successful(self) -> bool:
        """Check if the agent completed successfully."""
        return self.status == AgentStatus.SUCCESS
    
    def should_retry(self) -> bool:
        """Check if the agent suggests retrying."""
        return self.status == AgentStatus.RETRY
    
    def needs_human_review(self, threshold: float = 0.7) -> bool:
        """Check if confidence is below the review threshold."""
        return self.is_successful() and self.confidence_score < threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dictionary."""
        return {
            "agent_name": self.agent_name,
            "status": self.status.value,
            "output": self.output if not hasattr(self.output, "to_dict") 
                      else self.output.to_dict(),  # type: ignore
            "confidence_score": self.confidence_score,
            "explanation": self.explanation,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class PipelineState:
    """
    Immutable state object that flows through the recruitment pipeline.
    
    Each agent receives the current state, processes it, and returns a NEW
    state instance with updates. This immutability ensures:
        - Thread safety in concurrent execution
        - Easy state history tracking (each step is a snapshot)
        - No hidden side effects between agents
    
    Attributes:
        pipeline_id: Unique identifier for this pipeline execution
        job_id: The job posting being recruited for
        candidate_ids: List of candidates being processed
        created_at: When this pipeline was initiated
        current_stage: Human-readable current processing stage
        agent_outputs: Accumulated outputs from each agent (keyed by agent_name)
        decision_log: Audit trail of key decisions made
        errors: Any errors encountered (non-fatal)
        metadata: Additional context (config snapshots, etc.)
    
    Design Notes:
        - frozen=True ensures immutability
        - Use with_*() methods to create modified copies
        - agent_outputs stores raw dicts for serialization; type safety
          is enforced at the agent level
    
    Example:
        >>> state = PipelineState(job_id="job_123", candidate_ids=["c1", "c2"])
        >>> new_state = state.with_agent_output("parser", {"skills": [...]})
        >>> assert state is not new_state  # Original unchanged
    """
    job_id: str
    candidate_ids: tuple[str, ...] = field(default_factory=tuple)  # Immutable sequence
    pipeline_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    current_stage: str = "initialized"
    agent_outputs: Dict[str, Any] = field(default_factory=dict)
    decision_log: tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    errors: tuple[str, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # -------------------------------------------------------------------------
    # Immutable Update Methods
    # -------------------------------------------------------------------------
    
    def with_agent_output(
        self, 
        agent_name: str, 
        output: Any,
        stage: Optional[str] = None
    ) -> "PipelineState":
        """
        Return a new state with an agent's output recorded.
        
        Args:
            agent_name: The agent that produced the output
            output: The output data (should be JSON-serializable)
            stage: Optional stage name update
        
        Returns:
            New PipelineState with the output added
        """
        new_outputs = {**self.agent_outputs, agent_name: output}
        return replace(
            self,
            agent_outputs=new_outputs,
            current_stage=stage if stage else self.current_stage,
        )
    
    def with_decision(
        self,
        decision_type: str,
        decision: str,
        reasoning: str,
        confidence: float,
        agent_name: str,
    ) -> "PipelineState":
        """
        Return a new state with a decision logged for audit purposes.
        
        Args:
            decision_type: Category of decision (e.g., "shortlist", "score")
            decision: The actual decision made
            reasoning: Why this decision was made
            confidence: Confidence in the decision [0.0, 1.0]
            agent_name: Which agent made this decision
        
        Returns:
            New PipelineState with the decision logged
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": agent_name,
            "type": decision_type,
            "decision": decision,
            "reasoning": reasoning,
            "confidence": confidence,
        }
        return replace(self, decision_log=self.decision_log + (entry,))
    
    def with_error(self, error: str) -> "PipelineState":
        """Return a new state with an error recorded."""
        return replace(self, errors=self.errors + (error,))
    
    def with_stage(self, stage: str) -> "PipelineState":
        """Return a new state with updated current stage."""
        return replace(self, current_stage=stage)
    
    def with_candidates(self, candidate_ids: List[str]) -> "PipelineState":
        """Return a new state with updated candidate list."""
        return replace(self, candidate_ids=tuple(candidate_ids))
    
    # -------------------------------------------------------------------------
    # Query Methods
    # -------------------------------------------------------------------------
    
    def get_agent_output(self, agent_name: str) -> Optional[Any]:
        """Retrieve a specific agent's output, or None if not present."""
        return self.agent_outputs.get(agent_name)
    
    def has_agent_run(self, agent_name: str) -> bool:
        """Check if a specific agent has already contributed output."""
        return agent_name in self.agent_outputs
    
    def get_decisions_by_type(self, decision_type: str) -> List[Dict[str, Any]]:
        """Filter decision log by decision type."""
        return [d for d in self.decision_log if d.get("type") == decision_type]
    
    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to JSON-compatible dictionary."""
        return {
            "pipeline_id": self.pipeline_id,
            "job_id": self.job_id,
            "candidate_ids": list(self.candidate_ids),
            "created_at": self.created_at,
            "current_stage": self.current_stage,
            "agent_outputs": self.agent_outputs,
            "decision_log": list(self.decision_log),
            "errors": list(self.errors),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineState":
        """Deserialize state from a dictionary."""
        return cls(
            pipeline_id=data.get("pipeline_id", uuid4().hex),
            job_id=data["job_id"],
            candidate_ids=tuple(data.get("candidate_ids", [])),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            current_stage=data.get("current_stage", "initialized"),
            agent_outputs=data.get("agent_outputs", {}),
            decision_log=tuple(data.get("decision_log", [])),
            errors=tuple(data.get("errors", [])),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class AgentResult(Generic[OutputT]):
    """
    Combined result of an agent execution: response + updated state.
    
    This is what agents actually return from their run() method.
    It bundles the agent's response with any state mutations.
    
    Attributes:
        response: The agent's output and metadata
        state: The (potentially) updated pipeline state
    
    Example:
        >>> result = AgentResult(
        ...     response=AgentResponse(...),
        ...     state=new_state
        ... )
        >>> if result.response.is_successful():
        ...     next_agent.run(next_input, result.state)
    """
    response: AgentResponse[OutputT]
    state: PipelineState


# =============================================================================
# BASE AGENT ABSTRACT CLASS
# =============================================================================

class BaseAgent(ABC, Generic[InputT, OutputT]):
    """
    Abstract base class for all agents in the recruitment system.
    
    Every agent MUST inherit from this class and implement the run() method.
    This ensures consistent interfaces across all agents, enabling:
        - Uniform orchestration
        - Consistent logging and monitoring
        - Easy testing and mocking
        - Framework-agnostic design
    
    Class Attributes:
        name: Unique identifier for this agent type (must be overridden)
        description: Human-readable description of agent's responsibility
    
    Design Contract:
        1. Agents are stateless - all state flows through PipelineState
        2. Agents are pure functions - same input always produces same output
        3. Agents are single-purpose - each does exactly one thing
        4. Agents are honest - confidence scores reflect actual certainty
    
    Example:
        >>> class SkillExtractor(BaseAgent[ResumeText, ExtractedSkills]):
        ...     name = "skill_extractor"
        ...     description = "Extracts technical skills from resume text"
        ...     
        ...     def run(self, input_data: ResumeText, state: PipelineState) -> AgentResult[ExtractedSkills]:
        ...         skills = self._extract_skills(input_data.text)
        ...         response = AgentResponse(
        ...             agent_name=self.name,
        ...             status=AgentStatus.SUCCESS,
        ...             output=ExtractedSkills(skills=skills),
        ...             confidence_score=0.85,
        ...             explanation=f"Extracted {len(skills)} skills"
        ...         )
        ...         new_state = state.with_agent_output(self.name, response.output)
        ...         return AgentResult(response=response, state=new_state)
    """
    
    # -------------------------------------------------------------------------
    # Class Attributes (must be overridden by subclasses)
    # -------------------------------------------------------------------------
    
    name: str = "base_agent"  # Unique agent identifier
    description: str = "Base agent - must be overridden"  # What this agent does
    
    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    
    def __init__(self, agent_id: Optional[str] = None):
        """
        Initialize the agent with optional ID.
        
        Args:
            agent_id: Unique identifier for this agent instance.
                     If not provided, generates a UUID.
        """
        self.agent_id = agent_id or uuid4().hex[:12]
        self._reasoning_log: List[str] = []
    
    def log_reasoning(self, message: str) -> None:
        """Add a step to the reasoning trace for auditability."""
        self._reasoning_log.append(message)
    
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    
    @property
    def confidence_threshold(self) -> float:
        """
        Minimum confidence for auto-proceeding without human review.
        
        Override in subclasses for different thresholds.
        Decision-making agents should have higher thresholds.
        """
        return 0.7
    
    @property
    def max_retries(self) -> int:
        """Maximum retry attempts for transient failures."""
        return 3
    
    # -------------------------------------------------------------------------
    # Abstract Method
    # -------------------------------------------------------------------------
    
    @abstractmethod
    def run(
        self, 
        input_data: InputT, 
        state: PipelineState
    ) -> AgentResult[OutputT]:
        """
        Execute the agent's core logic.
        
        This is the ONLY public method agents expose. It receives typed input,
        processes it, and returns a typed response with updated state.
        
        Args:
            input_data: Strongly-typed input specific to this agent
            state: Current immutable pipeline state
        
        Returns:
            AgentResult containing:
                - AgentResponse with output, confidence, explanation
                - Updated PipelineState (new instance)
        
        Raises:
            Should NOT raise exceptions. Errors should be captured in
            AgentResponse with status=FAILURE or RETRY.
        
        Implementation Guidelines:
            1. Validate input_data at the start
            2. Perform core logic
            3. Construct response with honest confidence score
            4. Return new state (never mutate input state)
            5. Catch exceptions and convert to FAILURE/RETRY status
        """
        pass
    
    # -------------------------------------------------------------------------
    # Helper Methods (for subclasses)
    # -------------------------------------------------------------------------
    
    def _success(
        self,
        output: OutputT,
        state: PipelineState,
        confidence: float,
        explanation: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentResult[OutputT]:
        """
        Helper to construct a successful result.
        
        Args:
            output: The computed output
            state: Current state (will be updated with agent output)
            confidence: Confidence score [0.0, 1.0]
            explanation: Human-readable explanation
            metadata: Optional diagnostic info
        
        Returns:
            AgentResult with SUCCESS status and updated state
        """
        response = AgentResponse(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            output=output,
            confidence_score=confidence,
            explanation=explanation,
            metadata=metadata or {},
        )
        
        # Serialize output for state storage
        output_dict = output.to_dict() if hasattr(output, "to_dict") else output
        new_state = state.with_agent_output(self.name, output_dict)
        
        return AgentResult(response=response, state=new_state)
    
    def _failure(
        self,
        state: PipelineState,
        error: str,
        explanation: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentResult[OutputT]:
        """
        Helper to construct a failure result.
        
        Args:
            state: Current state (will have error recorded)
            error: Technical error message
            explanation: Human-readable explanation
            metadata: Optional diagnostic info
        
        Returns:
            AgentResult with FAILURE status and error logged
        """
        response: AgentResponse[OutputT] = AgentResponse(
            agent_name=self.name,
            status=AgentStatus.FAILURE,
            output=None,
            confidence_score=0.0,
            explanation=explanation,
            metadata={**(metadata or {}), "error": error},
        )
        new_state = state.with_error(f"[{self.name}] {error}")
        
        return AgentResult(response=response, state=new_state)
    
    def _retry(
        self,
        state: PipelineState,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentResult[OutputT]:
        """
        Helper to construct a retry result.
        
        Args:
            state: Current state (unchanged)
            reason: Why retry is suggested
            metadata: Optional diagnostic info
        
        Returns:
            AgentResult with RETRY status
        """
        response: AgentResponse[OutputT] = AgentResponse(
            agent_name=self.name,
            status=AgentStatus.RETRY,
            output=None,
            confidence_score=0.0,
            explanation=f"Retry suggested: {reason}",
            metadata=metadata or {},
        )
        return AgentResult(response=response, state=state)
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

# -----------------------------------------------------------------------------
# Example: Strongly-typed input/output dataclasses
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class EchoInput:
    """Example input: a message to echo."""
    message: str
    repeat_count: int = 1


@dataclass(frozen=True)
class EchoOutput:
    """Example output: the echoed message."""
    echoed_message: str
    char_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "echoed_message": self.echoed_message,
            "char_count": self.char_count,
        }


# -----------------------------------------------------------------------------
# Example: Dummy agent implementing the contract
# -----------------------------------------------------------------------------

class EchoAgent(BaseAgent[EchoInput, EchoOutput]):
    """
    A minimal example agent that echoes input messages.
    
    This demonstrates the complete agent contract:
        - Typed input (EchoInput)
        - Typed output (EchoOutput)
        - Proper error handling
        - State updates
        - Confidence scoring
    
    Not for production use - this is a reference implementation.
    """
    
    name = "echo_agent"
    description = "Echoes input messages for testing the agent framework"
    
    def run(
        self, 
        input_data: EchoInput, 
        state: PipelineState
    ) -> AgentResult[EchoOutput]:
        """
        Echo the input message.
        
        Args:
            input_data: Message to echo with repeat count
            state: Current pipeline state
        
        Returns:
            AgentResult with echoed message and updated state
        """
        # ---------------------------------------------------------------------
        # Step 1: Validate input
        # ---------------------------------------------------------------------
        if not input_data.message:
            return self._failure(
                state=state,
                error="Empty message provided",
                explanation="Cannot echo an empty message",
            )
        
        if input_data.repeat_count < 1:
            return self._failure(
                state=state,
                error=f"Invalid repeat_count: {input_data.repeat_count}",
                explanation="Repeat count must be at least 1",
            )
        
        # ---------------------------------------------------------------------
        # Step 2: Perform core logic
        # ---------------------------------------------------------------------
        try:
            echoed = (input_data.message + " ") * input_data.repeat_count
            echoed = echoed.strip()
            
            output = EchoOutput(
                echoed_message=echoed,
                char_count=len(echoed),
            )
            
        except Exception as e:
            # Catch unexpected errors and convert to failure
            return self._failure(
                state=state,
                error=str(e),
                explanation=f"Unexpected error during echo: {e}",
            )
        
        # ---------------------------------------------------------------------
        # Step 3: Construct response with confidence
        # ---------------------------------------------------------------------
        # Echo is deterministic, so confidence is 1.0
        confidence = 1.0
        
        explanation = (
            f"Successfully echoed message {input_data.repeat_count} time(s). "
            f"Output length: {output.char_count} characters."
        )
        
        # ---------------------------------------------------------------------
        # Step 4: Return result with updated state
        # ---------------------------------------------------------------------
        return self._success(
            output=output,
            state=state,
            confidence=confidence,
            explanation=explanation,
            metadata={
                "input_length": len(input_data.message),
                "repeat_count": input_data.repeat_count,
            },
        )


# -----------------------------------------------------------------------------
# Example: Usage demonstration
# -----------------------------------------------------------------------------

def example_usage() -> None:
    """
    Demonstrates the complete agent workflow.
    
    This shows:
        1. Creating initial pipeline state
        2. Running an agent with typed input
        3. Handling the response
        4. Passing updated state to next agent
    """
    # Create initial state
    initial_state = PipelineState(
        job_id="job_senior_engineer_2026",
        candidate_ids=("candidate_001", "candidate_002"),
        metadata={"source": "example_usage"},
    )
    
    print("=" * 60)
    print("AGENT CONTRACT EXAMPLE")
    print("=" * 60)
    
    # Instantiate agent
    agent = EchoAgent()
    print(f"\nAgent: {agent}")
    print(f"Description: {agent.description}")
    
    # Prepare typed input
    input_data = EchoInput(message="Hello, recruitment system!", repeat_count=2)
    print(f"\nInput: {input_data}")
    
    # Run agent
    result = agent.run(input_data, initial_state)
    
    # Inspect response
    print(f"\n--- Response ---")
    print(f"Status: {result.response.status.value}")
    print(f"Confidence: {result.response.confidence_score}")
    print(f"Explanation: {result.response.explanation}")
    print(f"Output: {result.response.output}")
    print(f"Needs human review: {result.response.needs_human_review()}")
    
    # Inspect state changes
    print(f"\n--- State Changes ---")
    print(f"Original state ID: {initial_state.pipeline_id}")
    print(f"Updated state ID: {result.state.pipeline_id}")  # Same ID, immutable copy
    print(f"Agent outputs recorded: {list(result.state.agent_outputs.keys())}")
    print(f"State is new object: {initial_state is not result.state}")
    
    # Demonstrate failure case
    print(f"\n--- Failure Case ---")
    bad_input = EchoInput(message="", repeat_count=1)
    bad_result = agent.run(bad_input, initial_state)
    print(f"Status: {bad_result.response.status.value}")
    print(f"Explanation: {bad_result.response.explanation}")
    print(f"Errors in state: {bad_result.state.errors}")
    
    # Demonstrate state chaining
    print(f"\n--- State Chaining ---")
    state_v1 = initial_state.with_stage("parsing")
    state_v2 = state_v1.with_decision(
        decision_type="shortlist",
        decision="include",
        reasoning="Score above threshold",
        confidence=0.85,
        agent_name="shortlister",
    )
    state_v3 = state_v2.with_candidates(["c1", "c2", "c3"])
    
    print(f"v1 stage: {state_v1.current_stage}")
    print(f"v2 decisions: {len(state_v2.decision_log)}")
    print(f"v3 candidates: {state_v3.candidate_ids}")
    print(f"Original unchanged: {initial_state.current_stage}")


# Run example if executed directly
if __name__ == "__main__":
    example_usage()
