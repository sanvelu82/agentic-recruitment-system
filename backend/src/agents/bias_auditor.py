"""
Bias Auditor Agent

Responsibility: Monitor and flag potential bias in the recruitment process.
Single purpose: Ensure fairness and compliance throughout the pipeline.

This is a CRITICAL COMPLIANCE agent that reviews all decisions.
"""

import re
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseAgent
from ..schemas.candidates import FinalRanking, MatchResult
from ..schemas.messages import PipelineState


# Comprehensive bias detection dictionaries
GENDERED_TERMS = {
    # Masculine-coded terms (may discourage female applicants)
    "rockstar": {"severity": "medium", "suggestion": "high performer"},
    "ninja": {"severity": "medium", "suggestion": "expert"},
    "guru": {"severity": "low", "suggestion": "specialist"},
    "hacker": {"severity": "low", "suggestion": "developer"},
    "manpower": {"severity": "high", "suggestion": "workforce"},
    "manning": {"severity": "high", "suggestion": "staffing"},
    "chairman": {"severity": "high", "suggestion": "chairperson"},
    "fireman": {"severity": "high", "suggestion": "firefighter"},
    "salesman": {"severity": "high", "suggestion": "salesperson"},
    "businessman": {"severity": "medium", "suggestion": "business professional"},
    "spokesman": {"severity": "high", "suggestion": "spokesperson"},
    "workmanship": {"severity": "low", "suggestion": "quality of work"},
    "man-hours": {"severity": "medium", "suggestion": "work hours"},
    "aggressive": {"severity": "medium", "suggestion": "ambitious"},
    "dominant": {"severity": "medium", "suggestion": "leading"},
    "fearless": {"severity": "low", "suggestion": "confident"},
    "headstrong": {"severity": "low", "suggestion": "determined"},
}

AGE_BIASED_TERMS = {
    "young": {"severity": "high", "suggestion": "motivated"},
    "energetic": {"severity": "low", "suggestion": "enthusiastic"},
    "digital native": {"severity": "high", "suggestion": "tech-savvy"},
    "recent graduate only": {"severity": "high", "suggestion": "entry-level candidates"},
    "fresh graduate": {"severity": "medium", "suggestion": "new graduate"},
    "youthful": {"severity": "high", "suggestion": "dynamic"},
    "mature": {"severity": "medium", "suggestion": "experienced"},
    "seasoned veteran": {"severity": "low", "suggestion": "experienced professional"},
    "overqualified": {"severity": "medium", "suggestion": None},
    "digital immigrant": {"severity": "high", "suggestion": None},
    "old school": {"severity": "medium", "suggestion": "traditional approach"},
}

DISABILITY_BIASED_TERMS = {
    "must be able to stand for long periods": {"severity": "medium", "suggestion": "position may require standing (accommodations available)"},
    "must have valid driver's license": {"severity": "low", "suggestion": "transportation required (accommodations available)"},
    "physically fit": {"severity": "medium", "suggestion": "able to perform job duties"},
    "able-bodied": {"severity": "high", "suggestion": "able to perform essential functions"},
    "clean health record": {"severity": "high", "suggestion": None},
    "no disabilities": {"severity": "critical", "suggestion": None},
}

ETHNICITY_NATIONALITY_BIASED_TERMS = {
    "native english speaker": {"severity": "high", "suggestion": "fluent in English"},
    "american accent": {"severity": "high", "suggestion": "clear communication skills"},
    "native speaker": {"severity": "medium", "suggestion": "fluent speaker"},
    "local candidates only": {"severity": "medium", "suggestion": "candidates in [location] preferred"},
    "must be citizen": {"severity": "low", "suggestion": "must be authorized to work"},
}

# Patterns that may indicate discriminatory intent
DISCRIMINATORY_PATTERNS = [
    (r"must be (\d+)-(\d+) years old", "age_restriction", "critical", "Age restrictions are illegal in most jurisdictions"),
    (r"no older than \d+", "age_restriction", "critical", "Maximum age requirements are discriminatory"),
    (r"preferably (male|female)", "gender_preference", "critical", "Gender preferences are discriminatory"),
    (r"looking for (a |)(male|female)", "gender_preference", "critical", "Gender-based hiring is discriminatory"),
    (r"(men|women) only", "gender_restriction", "critical", "Gender restrictions are illegal"),
    (r"must be married", "marital_status", "critical", "Marital status requirements are discriminatory"),
    (r"single (only|preferred)", "marital_status", "critical", "Marital status requirements are discriminatory"),
    (r"no children", "family_status", "critical", "Family status discrimination"),
    (r"childless", "family_status", "high", "Family status discrimination"),
]


@dataclass
class BiasAuditInput:
    """Input for bias auditing."""
    pipeline_state: PipelineState
    rankings: List[FinalRanking]
    match_results: Optional[List[MatchResult]] = None


@dataclass
class BiasFinding:
    """A single bias finding."""
    category: str
    severity: str  # critical, high, medium, low
    description: str
    affected_items: List[str]  # candidate IDs, gate IDs, or "all"
    recommendation: str
    evidence: Optional[str] = None


@dataclass
class BiasAuditResult:
    """Results of bias audit."""
    audit_passed: bool
    overall_fairness_score: float  # [0.0-1.0]
    findings: List[Dict[str, Any]]
    recommendations: List[str]
    requires_human_review: bool
    compliance_notes: List[str]
    detailed_analysis: Dict[str, Any] = None


class BiasAuditorAgent(BaseAgent[BiasAuditInput, BiasAuditResult]):
    """
    Audits the recruitment pipeline for potential bias and fairness issues.
    
    Input: Complete pipeline state with all decisions
    Output: Bias audit results with findings and recommendations
    
    Key responsibilities:
    - Analyze JD for biased language (gendered, age, disability, ethnicity)
    - Check decision patterns for disparate impact
    - Analyze score distributions for anomalies
    - Review borderline decision consistency
    - Ensure explanation quality
    - Provide compliance documentation
    
    This agent has VETO POWER:
    - Can require human review for any decision
    - Can flag the entire pipeline for review
    - Findings must be addressed before proceeding
    """
    
    @property
    def description(self) -> str:
        return (
            "Audits recruitment decisions for bias, ensuring fairness "
            "and regulatory compliance throughout the pipeline."
        )
    
    @property
    def required_confidence_threshold(self) -> float:
        return 0.9  # Compliance requires high confidence
    
    def run(
        self,
        input_data: BiasAuditInput,
        state: Optional["PipelineState"] = None
    ) -> "AgentResult[BiasAuditResult]":
        """
        Execute bias audit and return result.
        
        Args:
            input_data: BiasAuditInput with pipeline state and rankings
            state: Optional pipeline state
        
        Returns:
            AgentResult with BiasAuditResult and updated state
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
                explanation=f"Bias audit failed: {str(e)}",
            )
            return AgentResult(response=response, state=state)
    
    def _process(
        self, 
        input_data: BiasAuditInput
    ) -> tuple[BiasAuditResult, float, str]:
        """
        Comprehensive audit of pipeline for bias and fairness issues.
        
        Args:
            input_data: BiasAuditInput with pipeline state and rankings
        
        Returns:
            BiasAuditResult, confidence_score, explanation
        """
        self.log_reasoning("Starting comprehensive bias audit")
        
        findings: List[BiasFinding] = []
        recommendations: List[str] = []
        compliance_notes: List[str] = []
        detailed_analysis: Dict[str, Any] = {}
        
        # Audit 1: Comprehensive JD language analysis
        jd_findings = self._audit_jd_language(input_data.pipeline_state)
        findings.extend(jd_findings)
        if jd_findings:
            self.log_reasoning(f"JD language audit: {len(jd_findings)} issues found")
            detailed_analysis["jd_language_issues"] = len(jd_findings)
        
        # Audit 2: Decision gate analysis
        gate_findings = self._audit_decision_gates(input_data.pipeline_state)
        findings.extend(gate_findings)
        if gate_findings:
            self.log_reasoning(f"Decision gate audit: {len(gate_findings)} issues found")
        
        # Audit 3: Score distribution analysis
        score_findings, score_analysis = self._audit_score_distribution(input_data.rankings)
        findings.extend(score_findings)
        detailed_analysis["score_distribution"] = score_analysis
        
        # Audit 4: Ranking fairness analysis
        ranking_findings = self._audit_ranking_fairness(input_data.rankings)
        findings.extend(ranking_findings)
        
        # Audit 5: Explanation quality check
        explanation_findings = self._audit_explanation_quality(input_data.rankings)
        findings.extend(explanation_findings)
        
        # Audit 6: Match result consistency (if available)
        if input_data.match_results:
            match_findings = self._audit_match_consistency(input_data.match_results)
            findings.extend(match_findings)
        
        # Audit 7: Resume matching bias patterns
        matching_bias = self._audit_matching_bias_patterns(input_data.pipeline_state)
        findings.extend(matching_bias)
        
        # Calculate overall fairness score
        fairness_score = self._calculate_fairness_score(findings)
        detailed_analysis["fairness_score_breakdown"] = self._get_score_breakdown(findings)
        
        # Generate recommendations based on findings
        recommendations = self._generate_recommendations(findings)
        
        # Determine if audit passes
        has_critical = any(f.severity == "critical" for f in findings)
        has_high = any(f.severity == "high" for f in findings)
        audit_passed = fairness_score >= 0.7 and not has_critical
        
        # Generate compliance notes
        compliance_notes = self._generate_compliance_notes(
            input_data.pipeline_state, 
            findings, 
            fairness_score
        )
        
        # Convert findings to dict format
        findings_dicts = [
            {
                "category": f.category,
                "severity": f.severity,
                "description": f.description,
                "affected_items": f.affected_items,
                "recommendation": f.recommendation,
                "evidence": f.evidence,
            }
            for f in findings
        ]
        
        result = BiasAuditResult(
            audit_passed=audit_passed,
            overall_fairness_score=fairness_score,
            findings=findings_dicts,
            recommendations=recommendations,
            requires_human_review=not audit_passed or has_critical or has_high,
            compliance_notes=compliance_notes,
            detailed_analysis=detailed_analysis,
        )
        
        self.log_reasoning(
            f"Audit {'PASSED' if audit_passed else 'FAILED'} "
            f"with fairness score {fairness_score:.2f}"
        )
        
        confidence = 0.95 if len(findings) < 3 else 0.85
        explanation = (
            f"Bias audit {'PASSED' if audit_passed else 'REQUIRES REVIEW'}. "
            f"Fairness score: {fairness_score:.0%}. "
            f"Found {len(findings)} issues "
            f"(Critical: {sum(1 for f in findings if f.severity == 'critical')}, "
            f"High: {sum(1 for f in findings if f.severity == 'high')}, "
            f"Medium: {sum(1 for f in findings if f.severity == 'medium')}, "
            f"Low: {sum(1 for f in findings if f.severity == 'low')}). "
            f"{len(recommendations)} recommendations provided."
        )
        
        return result, confidence, explanation
    
    def _audit_jd_language(self, state: PipelineState) -> List[BiasFinding]:
        """Comprehensive audit of job description language for bias."""
        findings = []
        
        # Get JD text from parsed_jd or job_description
        jd_text = ""
        if state.parsed_jd:
            # Check if there are already bias flags from JD analyzer
            # Handle both dataclass and dict formats
            if hasattr(state.parsed_jd, 'potential_bias_flags'):
                existing_flags = state.parsed_jd.potential_bias_flags or []
            elif isinstance(state.parsed_jd, dict):
                existing_flags = state.parsed_jd.get("potential_bias_flags", [])
            else:
                existing_flags = []
            
            for flag in existing_flags:
                findings.append(BiasFinding(
                    category="jd_language_bias",
                    severity="medium",
                    description=flag,
                    affected_items=["all_candidates"],
                    recommendation="Review and revise job description language",
                ))
        
        if state.job_description:
            # Handle both dataclass and dict formats
            if hasattr(state.job_description, 'raw_description'):
                jd_text = state.job_description.raw_description or ""
            elif isinstance(state.job_description, dict):
                jd_text = state.job_description.get("raw_description", "")
            else:
                jd_text = ""
        
        if not jd_text:
            return findings
        
        text_lower = jd_text.lower()
        
        # Check for gendered terms
        for term, info in GENDERED_TERMS.items():
            if term in text_lower:
                suggestion = info["suggestion"]
                findings.append(BiasFinding(
                    category="gendered_language",
                    severity=info["severity"],
                    description=f"Gendered term '{term}' detected in job description",
                    affected_items=["all_candidates"],
                    recommendation=f"Replace '{term}' with '{suggestion}'" if suggestion else f"Remove or rephrase '{term}'",
                    evidence=self._extract_context(jd_text, term),
                ))
        
        # Check for age-biased terms
        for term, info in AGE_BIASED_TERMS.items():
            if term in text_lower:
                suggestion = info["suggestion"]
                findings.append(BiasFinding(
                    category="age_bias",
                    severity=info["severity"],
                    description=f"Age-biased term '{term}' detected",
                    affected_items=["all_candidates"],
                    recommendation=f"Replace '{term}' with '{suggestion}'" if suggestion else f"Remove '{term}'",
                    evidence=self._extract_context(jd_text, term),
                ))
        
        # Check for disability-biased terms
        for term, info in DISABILITY_BIASED_TERMS.items():
            if term in text_lower:
                suggestion = info["suggestion"]
                findings.append(BiasFinding(
                    category="disability_bias",
                    severity=info["severity"],
                    description=f"Potential disability bias: '{term}'",
                    affected_items=["all_candidates"],
                    recommendation=f"Consider: '{suggestion}'" if suggestion else "Review for ADA/disability compliance",
                    evidence=self._extract_context(jd_text, term),
                ))
        
        # Check for ethnicity/nationality bias
        for term, info in ETHNICITY_NATIONALITY_BIASED_TERMS.items():
            if term in text_lower:
                suggestion = info["suggestion"]
                findings.append(BiasFinding(
                    category="ethnicity_nationality_bias",
                    severity=info["severity"],
                    description=f"Potential ethnicity/nationality bias: '{term}'",
                    affected_items=["all_candidates"],
                    recommendation=f"Replace with: '{suggestion}'" if suggestion else "Remove discriminatory language",
                    evidence=self._extract_context(jd_text, term),
                ))
        
        # Check for discriminatory patterns using regex
        for pattern, bias_type, severity, message in DISCRIMINATORY_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                findings.append(BiasFinding(
                    category=bias_type,
                    severity=severity,
                    description=message,
                    affected_items=["all_candidates"],
                    recommendation="Remove discriminatory requirement immediately",
                    evidence=match.group(0),
                ))
        
        return findings
    
    def _extract_context(self, text: str, term: str, context_chars: int = 50) -> str:
        """Extract context around a found term."""
        text_lower = text.lower()
        idx = text_lower.find(term.lower())
        if idx == -1:
            return ""
        
        start = max(0, idx - context_chars)
        end = min(len(text), idx + len(term) + context_chars)
        
        context = text[start:end]
        if start > 0:
            context = "..." + context
        if end < len(text):
            context = context + "..."
        
        return context
    
    def _audit_decision_gates(self, state: PipelineState) -> List[BiasFinding]:
        """Audit decision gate consistency."""
        findings = []
        decision_gates = state.decision_gates
        
        if not decision_gates:
            return findings
        
        # Check for high borderline rate
        borderline_cases = []
        for gate in decision_gates:
            bias_flags = gate.get("bias_flags", [])
            if any("borderline" in str(flag).lower() for flag in bias_flags):
                borderline_cases.append(gate.get("gate_id", "unknown"))
        
        if len(borderline_cases) > len(decision_gates) * 0.3:
            findings.append(BiasFinding(
                category="threshold_calibration",
                severity="high",
                description=f"{len(borderline_cases)} of {len(decision_gates)} decisions ({len(borderline_cases)/len(decision_gates):.0%}) are borderline cases",
                affected_items=borderline_cases,
                recommendation="Review threshold settings - high borderline rate indicates poorly calibrated thresholds",
            ))
        
        # Check for inconsistent decision patterns
        passed_gates = [g for g in decision_gates if g.get("passed", False)]
        failed_gates = [g for g in decision_gates if not g.get("passed", False)]
        
        if passed_gates and failed_gates:
            # Check if pass/fail margins are inconsistent
            pass_margins = [
                abs(g.get("actual_value", 0) - g.get("threshold", 0))
                for g in passed_gates if g.get("actual_value") and g.get("threshold")
            ]
            fail_margins = [
                abs(g.get("actual_value", 0) - g.get("threshold", 0))
                for g in failed_gates if g.get("actual_value") and g.get("threshold")
            ]
            
            if pass_margins and fail_margins:
                avg_pass_margin = sum(pass_margins) / len(pass_margins)
                avg_fail_margin = sum(fail_margins) / len(fail_margins)
                
                # If failures are very close to passing, may indicate overly strict threshold
                if avg_fail_margin < 0.05:
                    findings.append(BiasFinding(
                        category="threshold_strictness",
                        severity="medium",
                        description=f"Failed candidates are very close to threshold (avg margin: {avg_fail_margin:.2%})",
                        affected_items=[g.get("gate_id", "") for g in failed_gates],
                        recommendation="Consider reviewing borderline rejections for potential false negatives",
                    ))
        
        return findings
    
    def _audit_score_distribution(
        self, 
        rankings: List[FinalRanking]
    ) -> Tuple[List[BiasFinding], Dict[str, Any]]:
        """Audit score distribution for anomalies."""
        findings = []
        analysis = {}
        
        if not rankings or len(rankings) < 3:
            return findings, {"insufficient_data": True}
        
        scores = [r.final_composite_score for r in rankings]
        
        # Calculate statistics
        mean_score = statistics.mean(scores)
        if len(scores) >= 2:
            stdev_score = statistics.stdev(scores)
        else:
            stdev_score = 0
        
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score
        
        analysis = {
            "mean": mean_score,
            "stdev": stdev_score,
            "min": min_score,
            "max": max_score,
            "range": score_range,
            "count": len(scores),
        }
        
        # Check for clustering (all scores very similar)
        if score_range < 0.1 and len(scores) > 5:
            findings.append(BiasFinding(
                category="score_clustering",
                severity="medium",
                description=f"Scores are tightly clustered (range: {score_range:.2%}, stdev: {stdev_score:.2%})",
                affected_items=["all_ranked"],
                recommendation="Score clustering may indicate: (1) evaluation not discriminating well, (2) all candidates truly similar, or (3) scoring algorithm issue",
            ))
        
        # Check for bimodal distribution (potential group discrimination)
        if len(scores) >= 10:
            # Simple bimodality check - are there two distinct clusters?
            sorted_scores = sorted(scores)
            mid = len(sorted_scores) // 2
            lower_half = sorted_scores[:mid]
            upper_half = sorted_scores[mid:]
            
            if lower_half and upper_half:
                lower_mean = statistics.mean(lower_half)
                upper_mean = statistics.mean(upper_half)
                gap = upper_mean - lower_mean
                
                if gap > 0.3:  # Large gap between groups
                    analysis["potential_bimodal"] = True
                    analysis["group_gap"] = gap
                    findings.append(BiasFinding(
                        category="score_bimodality",
                        severity="medium",
                        description=f"Scores show two distinct groups with {gap:.0%} gap",
                        affected_items=["all_ranked"],
                        recommendation="Investigate whether the score gap correlates with any protected characteristics",
                    ))
        
        # Check for outliers
        if stdev_score > 0:
            outliers_low = [r.candidate_id for r in rankings if r.final_composite_score < mean_score - 2 * stdev_score]
            outliers_high = [r.candidate_id for r in rankings if r.final_composite_score > mean_score + 2 * stdev_score]
            
            if outliers_low or outliers_high:
                analysis["outliers"] = {
                    "low": outliers_low,
                    "high": outliers_high,
                }
        
        return findings, analysis
    
    def _audit_ranking_fairness(self, rankings: List[FinalRanking]) -> List[BiasFinding]:
        """Audit rankings for fairness issues."""
        findings = []
        
        if not rankings:
            return findings
        
        # Check for suspicious patterns in top rankings
        top_rankings = rankings[:5] if len(rankings) >= 5 else rankings
        
        # Check if all top candidates are marked for human review (might indicate bias concerns)
        review_required_count = sum(1 for r in top_rankings if r.human_review_required)
        if review_required_count == len(top_rankings) and len(top_rankings) >= 3:
            findings.append(BiasFinding(
                category="review_required_pattern",
                severity="low",
                description="All top candidates flagged for human review",
                affected_items=[r.candidate_id for r in top_rankings],
                recommendation="High review rate may indicate uncertainty in evaluations",
            ))
        
        # Check for score-rank inconsistency
        for i, ranking in enumerate(rankings[:-1]):
            next_ranking = rankings[i + 1]
            if ranking.final_composite_score < next_ranking.final_composite_score:
                findings.append(BiasFinding(
                    category="rank_score_inconsistency",
                    severity="high",
                    description=f"Rank #{ranking.rank} has lower score than rank #{next_ranking.rank}",
                    affected_items=[ranking.candidate_id, next_ranking.candidate_id],
                    recommendation="Rankings should be strictly ordered by score unless justified",
                ))
        
        return findings
    
    def _audit_explanation_quality(self, rankings: List[FinalRanking]) -> List[BiasFinding]:
        """Audit quality of explanations for compliance."""
        findings = []
        
        if not rankings:
            return findings
        
        poor_explanations = []
        for r in rankings:
            # Check if explanation exists and is meaningful
            explanation = r.ranking_explanation or ""
            if len(explanation) < 20:
                poor_explanations.append(r.candidate_id)
            elif not any(word in explanation.lower() for word in ["score", "skill", "experience", "match", "qualification"]):
                poor_explanations.append(r.candidate_id)
        
        if poor_explanations:
            findings.append(BiasFinding(
                category="explanation_quality",
                severity="medium",
                description=f"{len(poor_explanations)} rankings lack adequate explanation",
                affected_items=poor_explanations,
                recommendation="All decisions must have clear, documented explanations for compliance",
            ))
        
        # Check for missing concerns documentation
        no_concerns_documented = [
            r.candidate_id for r in rankings
            if r.recommendation in ["consider", "not_recommended"] and not r.key_concerns
        ]
        
        if no_concerns_documented:
            findings.append(BiasFinding(
                category="missing_rejection_rationale",
                severity="high",
                description=f"{len(no_concerns_documented)} non-recommended candidates lack documented concerns",
                affected_items=no_concerns_documented,
                recommendation="All rejections or lower recommendations must document specific concerns",
            ))
        
        return findings
    
    def _audit_match_consistency(self, match_results: List[MatchResult]) -> List[BiasFinding]:
        """Audit match results for consistency."""
        findings = []
        
        if len(match_results) < 3:
            return findings
        
        # Check for high variance in confidence scores
        confidences = [m.confidence for m in match_results]
        if len(confidences) >= 2:
            conf_stdev = statistics.stdev(confidences)
            if conf_stdev > 0.2:
                findings.append(BiasFinding(
                    category="match_confidence_variance",
                    severity="low",
                    description=f"High variance in match confidence scores (stdev: {conf_stdev:.2%})",
                    affected_items=["all_matched"],
                    recommendation="Variable confidence may indicate inconsistent data quality across candidates",
                ))
        
        # Check for candidates with many bias flags
        high_bias_candidates = [
            m.candidate_id for m in match_results
            if len(m.bias_flags) >= 2
        ]
        
        if high_bias_candidates:
            findings.append(BiasFinding(
                category="multiple_bias_flags",
                severity="medium",
                description=f"{len(high_bias_candidates)} candidates have multiple bias flags in matching",
                affected_items=high_bias_candidates,
                recommendation="Review matches with multiple bias flags for potential systematic issues",
            ))
        
        return findings
    
    def _audit_matching_bias_patterns(self, state: PipelineState) -> List[BiasFinding]:
        """Check for bias patterns in the matching process."""
        findings = []
        
        # Check if certain skills are being weighted unfairly
        match_results = state.match_results
        if not match_results or len(match_results) < 5:
            return findings
        
        # Analyze skill match patterns
        skill_match_rates = {}
        for match in match_results:
            skill_matches = match.get("skill_matches", [])
            for sm in skill_matches:
                skill = sm.get("required_skill", "")
                if skill:
                    if skill not in skill_match_rates:
                        skill_match_rates[skill] = {"matched": 0, "total": 0}
                    skill_match_rates[skill]["total"] += 1
                    if sm.get("match_score", 0) >= 0.7:
                        skill_match_rates[skill]["matched"] += 1
        
        # Check for skills with very low match rates
        for skill, rates in skill_match_rates.items():
            if rates["total"] >= 5:
                match_rate = rates["matched"] / rates["total"]
                if match_rate < 0.1:
                    findings.append(BiasFinding(
                        category="skill_match_pattern",
                        severity="low",
                        description=f"Very low match rate ({match_rate:.0%}) for skill '{skill}'",
                        affected_items=["skill_requirement"],
                        recommendation=f"Skill '{skill}' may be too specific or niche. Consider broadening requirement.",
                    ))
        
        return findings
    
    def _calculate_fairness_score(self, findings: List[BiasFinding]) -> float:
        """Calculate overall fairness score based on findings."""
        severity_weights = {
            "critical": 0.4,
            "high": 0.25,
            "medium": 0.1,
            "low": 0.03,
        }
        
        penalty = sum(
            severity_weights.get(f.severity, 0.1)
            for f in findings
        )
        
        # Cap penalty at 1.0
        penalty = min(1.0, penalty)
        
        return max(0.0, 1.0 - penalty)
    
    def _get_score_breakdown(self, findings: List[BiasFinding]) -> Dict[str, int]:
        """Get breakdown of findings by severity."""
        breakdown = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
        }
        
        for f in findings:
            if f.severity in breakdown:
                breakdown[f.severity] += 1
        
        return breakdown
    
    def _generate_recommendations(self, findings: List[BiasFinding]) -> List[str]:
        """Generate actionable recommendations from findings."""
        recommendations = []
        
        # Deduplicate recommendations
        seen = set()
        for f in findings:
            if f.recommendation and f.recommendation not in seen:
                # Prioritize by severity
                if f.severity in ["critical", "high"]:
                    recommendations.insert(0, f"[{f.severity.upper()}] {f.recommendation}")
                else:
                    recommendations.append(f.recommendation)
                seen.add(f.recommendation)
        
        # Add general recommendations based on finding patterns
        categories = set(f.category for f in findings)
        
        if "gendered_language" in categories or "age_bias" in categories:
            if "Review job description with HR/Legal for inclusive language" not in seen:
                recommendations.append("Review job description with HR/Legal for inclusive language")
        
        if "threshold_calibration" in categories:
            if "Consider A/B testing different threshold values" not in seen:
                recommendations.append("Consider A/B testing different threshold values")
        
        if any(f.category == "explanation_quality" for f in findings):
            recommendations.append("Implement mandatory explanation templates for all decisions")
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def _generate_compliance_notes(
        self, 
        state: PipelineState, 
        findings: List[BiasFinding],
        fairness_score: float
    ) -> List[str]:
        """Generate compliance documentation notes."""
        notes = []
        
        # Basic audit info
        notes.append(f"Audit completed at pipeline stage: {state.current_stage.value}")
        notes.append(f"Total candidates processed: {len(state.candidates)}")
        notes.append(f"Overall fairness score: {fairness_score:.2%}")
        
        # Finding summary
        breakdown = self._get_score_breakdown(findings)
        notes.append(
            f"Total findings: {len(findings)} "
            f"(Critical: {breakdown['critical']}, High: {breakdown['high']}, "
            f"Medium: {breakdown['medium']}, Low: {breakdown['low']})"
        )
        
        # Critical issues
        critical_findings = [f for f in findings if f.severity == "critical"]
        if critical_findings:
            notes.append(f"CRITICAL ISSUES REQUIRING IMMEDIATE ACTION: {len(critical_findings)}")
            for f in critical_findings:
                notes.append(f"  - {f.description}")
        
        # Compliance status
        if fairness_score >= 0.9 and not critical_findings:
            notes.append("COMPLIANCE STATUS: PASS - No significant bias concerns identified")
        elif fairness_score >= 0.7 and not critical_findings:
            notes.append("COMPLIANCE STATUS: CONDITIONAL PASS - Minor issues to address")
        else:
            notes.append("COMPLIANCE STATUS: FAIL - Significant issues require resolution before proceeding")
        
        return notes
