"""System evaluator that runs ADAPT-AI through evaluation dataset."""
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
import time

from .metrics import ClinicalEvaluator, EvaluationResult
from .ground_truth import GroundTruthManager, GroundTruthQuery

logger = logging.getLogger(__name__)


@dataclass
class QueryEvaluationResult:
    """Results for evaluating a single query."""

    query_id: str
    category: str
    difficulty: str

    # System response
    prediction: str
    reference: str

    # Timing
    response_time: float

    # Metrics
    metrics: Dict[str, Any]

    # Pass/Fail
    passed: bool
    failure_reasons: List[str]

    # System metadata
    system_metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        data = asdict(self)
        # Convert metrics dataclass if needed
        if hasattr(self.metrics, 'to_dict'):
            data['metrics'] = self.metrics.to_dict()
        return data


@dataclass
class EvaluationReport:
    """Complete evaluation report."""

    evaluation_id: str
    timestamp: str
    total_queries: int
    passed_queries: int
    failed_queries: int

    # Aggregate metrics
    avg_bleu_4: float
    avg_rouge_l: float
    avg_concept_recall: float
    avg_safety_score: float
    avg_overall_score: float

    # Performance metrics
    avg_response_time: float
    total_cost: float

    # By category breakdown
    by_category: Dict[str, Dict]
    by_difficulty: Dict[str, Dict]

    # Individual results
    results: List[Dict]

    # Summary
    summary: str

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    def save(self, output_path: Path):
        """Save report to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Evaluation report saved to {output_path}")


class SystemEvaluator:
    """Evaluate ADAPT-AI system against ground truth dataset."""

    def __init__(
        self,
        use_bertscore: bool = False,
        output_dir: Path = None
    ):
        """Initialize system evaluator.

        Args:
            use_bertscore: Whether to compute BERTScore (slower)
            output_dir: Directory to save results
        """
        self.evaluator = ClinicalEvaluator(use_bertscore=use_bertscore)
        self.ground_truth = GroundTruthManager()

        if output_dir is None:
            output_dir = Path("./data/evaluation/results")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("SystemEvaluator initialized")

    async def evaluate_query(
        self,
        orchestrator,
        gt_query: GroundTruthQuery
    ) -> QueryEvaluationResult:
        """Evaluate system response to a single ground truth query.

        Args:
            orchestrator: MCPOrchestrator instance
            gt_query: Ground truth query

        Returns:
            QueryEvaluationResult
        """
        logger.info(f"Evaluating query {gt_query.query_id}: {gt_query.query_text[:50]}...")

        # Run query through system
        start_time = time.time()
        try:
            result = await orchestrator.process_query(
                query=gt_query.query_text,
                patient_id=gt_query.patient_id
            )
            response_time = time.time() - start_time

            if result['status'] == 'success':
                prediction = result['content']
                system_metadata = {
                    'query_id': result.get('query_id'),
                    'agents': result.get('agents', {}),
                    'metadata': result.get('metadata', {})
                }
            else:
                prediction = f"ERROR: {result.get('error', 'Unknown error')}"
                system_metadata = result
        except Exception as e:
            logger.error(f"Error processing query {gt_query.query_id}: {e}")
            prediction = f"EXCEPTION: {str(e)}"
            response_time = time.time() - start_time
            system_metadata = {'error': str(e)}

        # Evaluate against ground truth
        eval_result = self.evaluator.evaluate_response(
            prediction=prediction,
            reference=gt_query.reference_response,
            required_concepts=gt_query.required_concepts,
            critical_concepts=gt_query.critical_concepts,
            hallucination_patterns=gt_query.hallucination_patterns
        )

        # Determine pass/fail
        passed, failure_reasons = self._check_passing_criteria(
            eval_result, gt_query
        )

        return QueryEvaluationResult(
            query_id=gt_query.query_id,
            category=gt_query.category,
            difficulty=gt_query.difficulty,
            prediction=prediction,
            reference=gt_query.reference_response,
            response_time=response_time,
            metrics=eval_result.to_dict(),
            passed=passed,
            failure_reasons=failure_reasons,
            system_metadata=system_metadata
        )

    def _check_passing_criteria(
        self,
        result: EvaluationResult,
        gt_query: GroundTruthQuery
    ) -> tuple[bool, List[str]]:
        """Check if result meets passing criteria.

        Returns:
            (passed, failure_reasons)
        """
        failures = []

        # Check BLEU-4
        if result.bleu_4 is not None and result.bleu_4 < gt_query.min_bleu_score:
            failures.append(
                f"BLEU-4 {result.bleu_4:.3f} < {gt_query.min_bleu_score:.3f}"
            )

        # Check ROUGE-L
        if result.rouge_l is not None and result.rouge_l < gt_query.min_rouge_l:
            failures.append(
                f"ROUGE-L {result.rouge_l:.3f} < {gt_query.min_rouge_l:.3f}"
            )

        # Check Concept Recall
        if result.concept_recall is not None and \
           result.concept_recall < gt_query.min_concept_recall:
            failures.append(
                f"Concept Recall {result.concept_recall:.3f} < {gt_query.min_concept_recall:.3f}"
            )

        # Check Safety Score
        if result.safety_score is not None and \
           result.safety_score < gt_query.min_safety_score:
            failures.append(
                f"Safety Score {result.safety_score:.3f} < {gt_query.min_safety_score:.3f}"
            )

        # Check critical omissions
        if result.critical_omission_count > 0:
            failures.append(
                f"Missing {result.critical_omission_count} critical concepts"
            )

        # Check hallucinations
        if result.hallucination_count > 0:
            failures.append(
                f"Detected {result.hallucination_count} hallucinations"
            )

        passed = len(failures) == 0
        return passed, failures

    async def evaluate_all(
        self,
        orchestrator,
        categories: Optional[List[str]] = None,
        difficulties: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> EvaluationReport:
        """Evaluate system on all ground truth queries.

        Args:
            orchestrator: MCPOrchestrator instance
            categories: Filter by categories (None = all)
            difficulties: Filter by difficulties (None = all)
            limit: Maximum number of queries to evaluate

        Returns:
            EvaluationReport
        """
        # Get queries to evaluate
        queries = self.ground_truth.get_all_queries()

        if not queries:
            logger.warning("No ground truth queries found!")
            raise ValueError("No ground truth queries available for evaluation")

        # Apply filters
        if categories:
            queries = [q for q in queries if q.category in categories]
        if difficulties:
            queries = [q for q in queries if q.difficulty in difficulties]
        if limit:
            queries = queries[:limit]

        logger.info(f"Evaluating {len(queries)} queries...")

        # Evaluate each query
        results: List[QueryEvaluationResult] = []
        for query in queries:
            try:
                result = await self.evaluate_query(orchestrator, query)
                results.append(result)
                logger.info(
                    f"{query.query_id}: {'PASS' if result.passed else 'FAIL'} "
                    f"(Overall: {result.metrics.get('overall_score', 0.0):.3f})"
                )
            except Exception as e:
                logger.error(f"Failed to evaluate {query.query_id}: {e}")

        # Generate report
        report = self._generate_report(results)

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"evaluation_report_{timestamp}.json"
        report.save(report_file)

        return report

    def _generate_report(
        self,
        results: List[QueryEvaluationResult]
    ) -> EvaluationReport:
        """Generate evaluation report from results."""

        total_queries = len(results)
        passed_queries = sum(1 for r in results if r.passed)
        failed_queries = total_queries - passed_queries

        # Calculate aggregate metrics
        def safe_avg(values):
            values = [v for v in values if v is not None]
            return sum(values) / len(values) if values else 0.0

        bleu_scores = [
            r.metrics.get('bleu_scores', {}).get('bleu_4')
            for r in results
        ]
        rouge_scores = [
            r.metrics.get('rouge_scores', {}).get('rouge_l')
            for r in results
        ]
        concept_recalls = [
            r.metrics.get('clinical_scores', {}).get('concept_recall')
            for r in results
        ]
        safety_scores = [
            r.metrics.get('clinical_scores', {}).get('safety_score')
            for r in results
        ]
        overall_scores = [
            r.metrics.get('overall_score')
            for r in results
        ]

        avg_bleu = safe_avg(bleu_scores)
        avg_rouge = safe_avg(rouge_scores)
        avg_concept = safe_avg(concept_recalls)
        avg_safety = safe_avg(safety_scores)
        avg_overall = safe_avg(overall_scores)

        avg_time = safe_avg([r.response_time for r in results])

        # Breakdown by category
        by_category = {}
        categories = set(r.category for r in results)
        for cat in categories:
            cat_results = [r for r in results if r.category == cat]
            by_category[cat] = {
                'total': len(cat_results),
                'passed': sum(1 for r in cat_results if r.passed),
                'pass_rate': sum(1 for r in cat_results if r.passed) / len(cat_results),
                'avg_overall_score': safe_avg([
                    r.metrics.get('overall_score') for r in cat_results
                ])
            }

        # Breakdown by difficulty
        by_difficulty = {}
        difficulties = set(r.difficulty for r in results)
        for diff in difficulties:
            diff_results = [r for r in results if r.difficulty == diff]
            by_difficulty[diff] = {
                'total': len(diff_results),
                'passed': sum(1 for r in diff_results if r.passed),
                'pass_rate': sum(1 for r in diff_results if r.passed) / len(diff_results),
                'avg_overall_score': safe_avg([
                    r.metrics.get('overall_score') for r in diff_results
                ])
            }

        # Generate summary
        summary = f"""
ADAPT-AI Evaluation Summary
===========================

Overall Performance:
- Total Queries: {total_queries}
- Passed: {passed_queries} ({passed_queries/total_queries*100:.1f}%)
- Failed: {failed_queries} ({failed_queries/total_queries*100:.1f}%)

Average Metrics:
- BLEU-4: {avg_bleu:.3f}
- ROUGE-L: {avg_rouge:.3f}
- Concept Recall: {avg_concept:.3f}
- Safety Score: {avg_safety:.3f}
- Overall Score: {avg_overall:.3f}

Performance:
- Avg Response Time: {avg_time:.2f}s

By Category:
{self._format_breakdown(by_category)}

By Difficulty:
{self._format_breakdown(by_difficulty)}
""".strip()

        evaluation_id = f"EVAL-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        return EvaluationReport(
            evaluation_id=evaluation_id,
            timestamp=datetime.now().isoformat(),
            total_queries=total_queries,
            passed_queries=passed_queries,
            failed_queries=failed_queries,
            avg_bleu_4=avg_bleu,
            avg_rouge_l=avg_rouge,
            avg_concept_recall=avg_concept,
            avg_safety_score=avg_safety,
            avg_overall_score=avg_overall,
            avg_response_time=avg_time,
            total_cost=0.0,  # Would calculate from system metadata
            by_category=by_category,
            by_difficulty=by_difficulty,
            results=[r.to_dict() for r in results],
            summary=summary
        )

    def _format_breakdown(self, breakdown: Dict) -> str:
        """Format breakdown dict for display."""
        lines = []
        for key, stats in breakdown.items():
            lines.append(
                f"  {key}: {stats['passed']}/{stats['total']} "
                f"({stats['pass_rate']*100:.1f}%) "
                f"Avg Score: {stats['avg_overall_score']:.3f}"
            )
        return '\n'.join(lines)
