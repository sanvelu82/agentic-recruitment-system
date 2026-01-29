"""
Test Generator Agent

Responsibility: Generate MCQ assessments based on job requirements.
Single purpose: Create fair, relevant test questions aligned to the JD.

This agent does NOT evaluate responses - only generates questions.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .base import BaseAgent
from ..schemas.job import ParsedJD, TestQuestion

# LangChain imports for LLM integration (using Groq)
try:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


# Predefined question templates as fallback
FALLBACK_QUESTIONS = {
    "Python": [
        {
            "question_text": "What is the output of the following Python code?\n\n```python\nx = [1, 2, 3]\ny = x\ny.append(4)\nprint(x)\n```",
            "options": {
                "A": "[1, 2, 3]",
                "B": "[1, 2, 3, 4]",
                "C": "[4, 1, 2, 3]",
                "D": "Error"
            },
            "correct_option": "B",
            "explanation": "Lists in Python are mutable and assigned by reference. When y = x, both variables point to the same list object. Appending to y also modifies x.",
            "difficulty": "medium"
        },
        {
            "question_text": "Which of the following is the correct way to create a virtual environment in Python 3?",
            "options": {
                "A": "python -m venv myenv",
                "B": "virtualenv create myenv",
                "C": "python create venv myenv",
                "D": "pip install venv myenv"
            },
            "correct_option": "A",
            "explanation": "The python -m venv command is the standard way to create virtual environments in Python 3. It uses the built-in venv module.",
            "difficulty": "easy"
        },
        {
            "question_text": "What is the time complexity of searching for an element in a Python dictionary?",
            "options": {
                "A": "O(n)",
                "B": "O(log n)",
                "C": "O(1) average case",
                "D": "O(n²)"
            },
            "correct_option": "C",
            "explanation": "Python dictionaries are implemented as hash tables, providing O(1) average-case time complexity for lookups, insertions, and deletions.",
            "difficulty": "medium"
        }
    ],
    "JavaScript": [
        {
            "question_text": "What will be the output of the following JavaScript code?\n\n```javascript\nconsole.log(typeof null);\n```",
            "options": {
                "A": "null",
                "B": "undefined",
                "C": "object",
                "D": "NaN"
            },
            "correct_option": "C",
            "explanation": "This is a well-known JavaScript quirk. typeof null returns 'object' due to a historical bug in the language that has been kept for backward compatibility.",
            "difficulty": "medium"
        },
        {
            "question_text": "Which method is used to convert a JSON string to a JavaScript object?",
            "options": {
                "A": "JSON.stringify()",
                "B": "JSON.parse()",
                "C": "JSON.convert()",
                "D": "JSON.toObject()"
            },
            "correct_option": "B",
            "explanation": "JSON.parse() converts a JSON string into a JavaScript object. JSON.stringify() does the opposite - converts an object to a JSON string.",
            "difficulty": "easy"
        }
    ],
    "SQL": [
        {
            "question_text": "Which SQL clause is used to filter grouped results?",
            "options": {
                "A": "WHERE",
                "B": "HAVING",
                "C": "FILTER",
                "D": "GROUP BY"
            },
            "correct_option": "B",
            "explanation": "HAVING is used to filter results after GROUP BY aggregation. WHERE filters individual rows before grouping, while HAVING filters grouped results.",
            "difficulty": "medium"
        },
        {
            "question_text": "What is the difference between INNER JOIN and LEFT JOIN?",
            "options": {
                "A": "There is no difference",
                "B": "INNER JOIN returns all rows from both tables",
                "C": "LEFT JOIN returns all rows from the left table and matched rows from the right table",
                "D": "LEFT JOIN is faster than INNER JOIN"
            },
            "correct_option": "C",
            "explanation": "LEFT JOIN returns all rows from the left table, with NULL values for non-matching rows from the right table. INNER JOIN only returns rows where there is a match in both tables.",
            "difficulty": "easy"
        }
    ],
    "REST API": [
        {
            "question_text": "Which HTTP status code indicates that a resource was successfully created?",
            "options": {
                "A": "200 OK",
                "B": "201 Created",
                "C": "204 No Content",
                "D": "202 Accepted"
            },
            "correct_option": "B",
            "explanation": "201 Created is the standard response for successful resource creation. 200 OK is for general success, 204 for success with no content, and 202 for accepted but not yet processed.",
            "difficulty": "easy"
        },
        {
            "question_text": "Which HTTP method should be used to partially update a resource?",
            "options": {
                "A": "PUT",
                "B": "POST",
                "C": "PATCH",
                "D": "UPDATE"
            },
            "correct_option": "C",
            "explanation": "PATCH is designed for partial updates to a resource. PUT is for complete replacement of a resource. There is no standard HTTP UPDATE method.",
            "difficulty": "medium"
        }
    ],
    "Docker": [
        {
            "question_text": "What is the purpose of a Dockerfile?",
            "options": {
                "A": "To run Docker containers",
                "B": "To define instructions for building a Docker image",
                "C": "To manage Docker networks",
                "D": "To store Docker volumes"
            },
            "correct_option": "B",
            "explanation": "A Dockerfile contains instructions for building a Docker image. It specifies the base image, commands to run, files to copy, and other configuration.",
            "difficulty": "easy"
        }
    ],
    "Git": [
        {
            "question_text": "What does 'git rebase' do?",
            "options": {
                "A": "Deletes the current branch",
                "B": "Moves or combines a sequence of commits to a new base commit",
                "C": "Creates a new branch",
                "D": "Reverts all changes"
            },
            "correct_option": "B",
            "explanation": "Git rebase re-applies commits on top of another base tip, creating a linear history. It's often used to maintain a clean commit history.",
            "difficulty": "medium"
        }
    ],
    "AWS": [
        {
            "question_text": "Which AWS service is used for serverless function execution?",
            "options": {
                "A": "EC2",
                "B": "Lambda",
                "C": "ECS",
                "D": "Elastic Beanstalk"
            },
            "correct_option": "B",
            "explanation": "AWS Lambda is a serverless compute service that runs code in response to events without provisioning or managing servers.",
            "difficulty": "easy"
        }
    ],
    "React": [
        {
            "question_text": "What is the purpose of the useEffect hook in React?",
            "options": {
                "A": "To manage component state",
                "B": "To perform side effects in function components",
                "C": "To create new components",
                "D": "To handle form submissions"
            },
            "correct_option": "B",
            "explanation": "useEffect is used to perform side effects in function components, such as data fetching, subscriptions, or DOM manipulation. It replaces lifecycle methods like componentDidMount.",
            "difficulty": "medium"
        }
    ],
    "Machine Learning": [
        {
            "question_text": "What is overfitting in machine learning?",
            "options": {
                "A": "When a model performs poorly on training data",
                "B": "When a model learns the training data too well and fails to generalize",
                "C": "When a model is too simple",
                "D": "When training takes too long"
            },
            "correct_option": "B",
            "explanation": "Overfitting occurs when a model learns the noise and details of training data to the extent that it negatively impacts performance on new data. The model essentially memorizes rather than learns patterns.",
            "difficulty": "medium"
        }
    ]
}


@dataclass
class TestGeneratorInput:
    """Input for test generation."""
    job_id: str = ""
    parsed_jd: Any = None  # ParsedJD or dict
    num_questions: int = 10
    difficulty: str = "mixed"  # easy, medium, hard, mixed
    
    def __post_init__(self):
        if isinstance(self.parsed_jd, dict):
            # Convert dict to ParsedJD-like object if needed
            pass
    
    @property
    def difficulty_distribution(self) -> Dict[str, float]:
        """Calculate difficulty distribution based on difficulty setting."""
        if self.difficulty == "easy":
            return {"easy": 0.7, "medium": 0.3, "hard": 0.0}
        elif self.difficulty == "medium":
            return {"easy": 0.2, "medium": 0.6, "hard": 0.2}
        elif self.difficulty == "hard":
            return {"easy": 0.0, "medium": 0.3, "hard": 0.7}
        else:  # mixed
            return {"easy": 0.33, "medium": 0.34, "hard": 0.33}


@dataclass
class TestGeneratorOutput:
    """Output from test generation."""
    test_id: str
    job_id: str
    questions: List[TestQuestion]
    total_time_minutes: int
    topics_covered: List[str]
    difficulty_breakdown: Dict[str, int]
    questions_by_category: Dict[str, List[str]] = None
    
    def __post_init__(self):
        if self.questions_by_category is None:
            self.questions_by_category = {}


class TestGeneratorAgent(BaseAgent[TestGeneratorInput, TestGeneratorOutput]):
    """
    Generates MCQ assessments based on job description requirements using LLM.
    
    Input: ParsedJD + generation parameters
    Output: List of test questions with answers and explanations
    
    Key responsibilities:
    - Generate questions aligned to required skills using LLM
    - Ensure fair difficulty distribution
    - Provide clear explanations for answers
    - Avoid culturally biased questions
    
    Does NOT:
    - Evaluate candidate responses
    - Score tests
    - Make hiring decisions
    """
    
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
                    temperature=0.7,  # Higher temperature for variety
                    api_key=api_key,
                )
        return self._llm
    
    @property
    def description(self) -> str:
        return (
            "Generates fair, job-relevant MCQ assessments from parsed job "
            "descriptions, ensuring coverage of required skills and topics."
        )
    
    @property
    def required_confidence_threshold(self) -> float:
        return 0.75  # Questions need review before use
    
    def run(
        self,
        input_data: TestGeneratorInput,
        state: Optional["PipelineState"] = None
    ) -> "AgentResult[TestGeneratorOutput]":
        """
        Execute test generation and return result.
        
        Args:
            input_data: TestGeneratorInput with JD and parameters
            state: Optional pipeline state
        
        Returns:
            AgentResult with TestGeneratorOutput and updated state
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
                explanation=f"Test generation failed: {str(e)}",
            )
            return AgentResult(response=response, state=state)
    
    def _process(
        self, 
        input_data: TestGeneratorInput
    ) -> tuple[TestGeneratorOutput, float, str]:
        """
        Generate test questions for a job using LLM or fallback templates.
        
        Args:
            input_data: TestGeneratorInput with JD and parameters
        
        Returns:
            TestGeneratorOutput, confidence_score, explanation
        """
        jd = input_data.parsed_jd
        num_questions = min(input_data.num_questions, 10)  # Cap at 10 questions
        
        self.log_reasoning(f"Generating {num_questions} questions for job {jd.job_id[:8]}")
        self.log_reasoning(f"Topics to cover: {jd.technical_topics}")
        
        test_id = uuid4().hex
        
        # Calculate questions per difficulty
        diff_dist = input_data.difficulty_distribution
        difficulty_breakdown = {
            level: max(1, int(num_questions * ratio))
            for level, ratio in diff_dist.items()
        }
        
        # Try LLM generation first
        llm = self._get_llm()
        questions = []
        topics_covered = []
        
        if llm and LANGCHAIN_AVAILABLE and jd.technical_topics:
            try:
                questions, topics_covered = self._generate_with_llm(
                    jd, num_questions, difficulty_breakdown, llm
                )
                self.log_reasoning(f"LLM generated {len(questions)} questions")
            except Exception as e:
                self.log_reasoning(f"LLM generation failed: {e}, using fallback")
                questions, topics_covered = self._generate_from_templates(jd, num_questions)
        else:
            self.log_reasoning("LLM not available or no topics, using template fallback")
            questions, topics_covered = self._generate_from_templates(jd, num_questions)
        
        # Ensure we have at least some questions
        if not questions:
            self.log_reasoning("No questions generated, using generic questions")
            questions, topics_covered = self._generate_generic_questions(num_questions)
        
        # Set job_id on all questions
        for q in questions:
            q.job_id = jd.job_id
        
        output = TestGeneratorOutput(
            test_id=test_id,
            job_id=jd.job_id,
            questions=questions,
            total_time_minutes=len(questions) * 2,  # 2 min per question
            topics_covered=topics_covered,
            difficulty_breakdown=difficulty_breakdown,
        )
        
        confidence = 0.85 if llm else 0.75
        explanation = (
            f"Generated test with {len(questions)} questions. "
            f"Covers {len(topics_covered)} topics: {', '.join(topics_covered[:5])}. "
            f"Estimated time: {output.total_time_minutes} minutes."
        )
        
        self.log_reasoning("Test generation completed")
        
        return output, confidence, explanation
    
    def _generate_with_llm(
        self, 
        jd: ParsedJD, 
        num_questions: int,
        difficulty_breakdown: Dict[str, int],
        llm
    ) -> tuple[List[TestQuestion], List[str]]:
        """Generate questions using LLM."""
        
        questions = []
        topics_covered = set()
        
        # Distribute questions across topics
        topics = jd.technical_topics[:5] if jd.technical_topics else ["General Programming"]
        questions_per_topic = max(1, num_questions // len(topics))
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert technical interviewer creating assessment questions.
Generate high-quality multiple choice questions that:
1. Test practical knowledge, not just memorization
2. Have clear, unambiguous answers
3. Include plausible distractors (wrong options)
4. Avoid cultural bias or region-specific references
5. Are appropriate for the specified difficulty level

Return ONLY valid JSON with no additional text."""),
            ("user", """Generate {num_questions} multiple choice questions about {topic} for a {seniority} level {job_function} position.

Difficulty distribution: {difficulty_dist}

Job context: {job_title}
Related skills: {related_skills}

Return a JSON array with this exact structure for each question:
[
    {{
        "question_text": "Clear question text. Can include code snippets formatted with markdown.",
        "options": {{
            "A": "First option",
            "B": "Second option",
            "C": "Third option",
            "D": "Fourth option"
        }},
        "correct_option": "A, B, C, or D",
        "explanation": "Detailed explanation of why this answer is correct and why others are wrong.",
        "difficulty": "easy, medium, or hard",
        "topic": "specific sub-topic tested"
    }}
]

Ensure questions are practical and test real-world understanding.""")
        ])
        
        parser = JsonOutputParser()
        chain = prompt | llm | parser
        
        for topic in topics:
            try:
                # Find related skills for context
                related_skills = [
                    s.skill_name for s in jd.skills 
                    if topic.lower() in s.skill_name.lower() or s.skill_name.lower() in topic.lower()
                ][:3]
                
                result = chain.invoke({
                    "num_questions": questions_per_topic,
                    "topic": topic,
                    "seniority": jd.seniority_level,
                    "job_function": jd.job_function,
                    "difficulty_dist": str(difficulty_breakdown),
                    "job_title": jd.job_title_normalized,
                    "related_skills": ", ".join(related_skills) if related_skills else topic,
                })
                
                # Parse LLM response into TestQuestion objects
                for q_data in result:
                    question = TestQuestion(
                        job_id=jd.job_id,
                        question_text=q_data.get("question_text", ""),
                        options=q_data.get("options", {}),
                        correct_option=q_data.get("correct_option", "A"),
                        explanation=q_data.get("explanation", ""),
                        skill_tested=topic,
                        topic=q_data.get("topic", topic),
                        difficulty=q_data.get("difficulty", "medium"),
                        time_limit_seconds=120,  # 2 minutes
                    )
                    questions.append(question)
                    topics_covered.add(topic)
                    
            except Exception as e:
                self.log_reasoning(f"Failed to generate questions for {topic}: {e}")
                continue
        
        return questions[:num_questions], list(topics_covered)
    
    def _generate_from_templates(
        self, 
        jd: ParsedJD, 
        num_questions: int
    ) -> tuple[List[TestQuestion], List[str]]:
        """Generate questions from predefined templates based on JD topics."""
        
        questions = []
        topics_covered = set()
        
        # Map JD topics to available templates
        topics = jd.technical_topics if jd.technical_topics else list(FALLBACK_QUESTIONS.keys())[:3]
        
        for topic in topics:
            # Find matching template category
            template_key = None
            topic_lower = topic.lower()
            
            for key in FALLBACK_QUESTIONS.keys():
                if key.lower() in topic_lower or topic_lower in key.lower():
                    template_key = key
                    break
            
            if template_key and template_key in FALLBACK_QUESTIONS:
                for q_data in FALLBACK_QUESTIONS[template_key]:
                    if len(questions) >= num_questions:
                        break
                    
                    question = TestQuestion(
                        job_id=jd.job_id,
                        question_text=q_data["question_text"],
                        options=q_data["options"],
                        correct_option=q_data["correct_option"],
                        explanation=q_data["explanation"],
                        skill_tested=template_key,
                        topic=template_key,
                        difficulty=q_data["difficulty"],
                        time_limit_seconds=120,
                    )
                    questions.append(question)
                    topics_covered.add(template_key)
            
            if len(questions) >= num_questions:
                break
        
        # Fill remaining with general questions if needed
        if len(questions) < num_questions:
            for key, q_list in FALLBACK_QUESTIONS.items():
                for q_data in q_list:
                    if len(questions) >= num_questions:
                        break
                    if key not in topics_covered:
                        question = TestQuestion(
                            job_id=jd.job_id,
                            question_text=q_data["question_text"],
                            options=q_data["options"],
                            correct_option=q_data["correct_option"],
                            explanation=q_data["explanation"],
                            skill_tested=key,
                            topic=key,
                            difficulty=q_data["difficulty"],
                            time_limit_seconds=120,
                        )
                        questions.append(question)
                        topics_covered.add(key)
                if len(questions) >= num_questions:
                    break
        
        return questions[:num_questions], list(topics_covered)
    
    def _generate_generic_questions(
        self, 
        num_questions: int
    ) -> tuple[List[TestQuestion], List[str]]:
        """Generate generic programming questions as last resort."""
        
        questions = []
        topics_covered = set()
        
        # Collect questions from all categories
        all_questions = []
        for key, q_list in FALLBACK_QUESTIONS.items():
            for q_data in q_list:
                all_questions.append((key, q_data))
        
        for i, (topic, q_data) in enumerate(all_questions[:num_questions]):
            question = TestQuestion(
                job_id="",
                question_text=q_data["question_text"],
                options=q_data["options"],
                correct_option=q_data["correct_option"],
                explanation=q_data["explanation"],
                skill_tested=topic,
                topic=topic,
                difficulty=q_data["difficulty"],
                time_limit_seconds=120,
            )
            questions.append(question)
            topics_covered.add(topic)
        
        return questions, list(topics_covered)
