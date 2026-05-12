#!/usr/bin/env python
"""Run comprehensive evaluation of ADAPT-AI system.

This script:
1. Loads ground truth queries
2. Runs each query through the ADAPT-AI system
3. Evaluates responses using BLEU, ROUGE, and clinical metrics
4. Generates detailed evaluation report
"""
import asyncio
import argparse
import sys
from pathlib import Path
import logging

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mcp.orchestrator import MCPOrchestrator
from evaluation.evaluator import SystemEvaluator
from evaluation.ground_truth import ground_truth_manager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main evaluation runner."""
    parser = argparse.ArgumentParser(
        description="Evaluate ADAPT-AI clinical decision support system"
    )
    parser.add_argument(
        '--categories',
        nargs='+',
        help='Filter by categories (e.g., simple_knowledge differential_diagnosis)'
    )
    parser.add_argument(
        '--difficulties',
        nargs='+',
        help='Filter by difficulty (easy, medium, hard)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of queries to evaluate'
    )
    parser.add_argument(
        '--use-bertscore',
        action='store_true',
        help='Compute BERTScore (slower but more accurate)'
    )
    parser.add_argument(
        '--load-samples',
        action='store_true',
        help='Load sample ground truth queries before evaluation'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('./data/evaluation/results'),
        help='Output directory for results'
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("ADAPT-AI System Evaluation")
    logger.info("=" * 80)

    # Load sample queries if requested
    if args.load_samples:
        logger.info("Loading sample ground truth queries...")
        ground_truth_manager.load_sample_queries()

    # Check if we have queries
    queries = ground_truth_manager.get_all_queries()
    if not queries:
        logger.error("No ground truth queries found!")
        logger.error("Run with --load-samples to generate sample queries")
        return 1

    logger.info(f"Found {len(queries)} ground truth queries")

    # Initialize system
    logger.info("Initializing ADAPT-AI system...")
    orchestrator = MCPOrchestrator()

    # Initialize evaluator
    logger.info("Initializing evaluator...")
    evaluator = SystemEvaluator(
        use_bertscore=args.use_bertscore,
        output_dir=args.output_dir
    )

    # Run evaluation
    logger.info("Starting evaluation...")
    logger.info(f"Filters: categories={args.categories}, difficulties={args.difficulties}, limit={args.limit}")

    report = await evaluator.evaluate_all(
        orchestrator=orchestrator,
        categories=args.categories,
        difficulties=args.difficulties,
        limit=args.limit
    )

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 80)
    print("\n" + report.summary)

    logger.info("\n" + "=" * 80)
    logger.info(f"Detailed results saved to: {args.output_dir}")
    logger.info("=" * 80)

    # Return exit code based on pass rate
    pass_rate = report.passed_queries / report.total_queries
    if pass_rate >= 0.8:
        return 0  # Success
    elif pass_rate >= 0.6:
        logger.warning(f"Pass rate {pass_rate*100:.1f}% is below 80%")
        return 1
    else:
        logger.error(f"Pass rate {pass_rate*100:.1f}% is critically low!")
        return 2


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
