#!/usr/bin/env python3
"""Expand RAES public agent-facing questions from 400 to 1K.

This script intentionally creates public questions only. It does not invent
gold answers or gold sources, because weak synthetic gold would contaminate
evaluation. The output is suitable for teacher trajectory collection and SFT
data generation, not for gold-backed scoring unless gold is added later.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


PUBLIC_FIELDS = ("id", "category", "domain", "task_type", "difficulty", "question", "split")
GOLD_FIELDS = {
    "answerability",
    "gold_answer",
    "required_facets",
    "source_policy",
    "minimum_evidence_policy",
    "gold_sources",
    "facet_evidence",
    "known_conflicts",
    "conflict_evidence",
    "misleading_sources",
    "common_wrong_answers",
    "failure_modes",
    "stop_condition",
    "uncertainty_source",
    "verification_mode",
    "verification_mode_label",
    "metadata",
    "process_diagnostics",
    "scoring_rubric",
}


CATEGORY_TARGETS = {
    "academic": 175,
    "deep_search": 300,
    "technical": 200,
    "fact_check": 150,
    "data_analysis": 175,
}

DIFFICULTY_TARGETS = {
    "easy": 150,
    "medium": 500,
    "hard": 350,
}

SPLIT_TARGETS = {
    "dev": 200,
    "test": 600,
    "stress": 200,
}


ACADEMIC_SUBJECTS = [
    "Search-R1",
    "WebGPT",
    "Self-RAG",
    "ReAct",
    "RAGAS faithfulness evaluation",
    "FEVER evidence annotation",
    "AIS attribution evaluation",
    "BEIR retrieval benchmarking",
    "BrowseComp",
    "GAIA",
    "WebArena",
    "SWE-bench",
    "STORM",
    "DSPy teleprompters",
    "Tree of Thoughts",
    "Reflexion",
    "Voyager",
    "Toolformer",
    "Let us Verify Step by Step",
    "process supervision",
    "retrieval-augmented generation",
    "agentic RAG",
    "long-context retrieval",
    "citation faithfulness",
    "uncertainty calibration",
    "source attribution",
    "multi-hop web QA",
    "retriever learning",
    "answer abstention",
    "evidence sufficiency",
]

DEEP_SEARCH_SUBJECTS = [
    "OpenAI Deep Research",
    "GPT Researcher",
    "Open Deep Research",
    "SearchClaw",
    "Storm-MM",
    "STORM",
    "Perplexity-style answer engines",
    "LangChain Open Deep Research",
    "LangGraph research agents",
    "AutoGen research workflows",
    "CrewAI research workflows",
    "OpenHands browsing agents",
    "WebArena",
    "MiniWoB++",
    "OSWorld",
    "GAIA",
    "BrowseComp",
    "SimpleQA",
    "FreshQA",
    "FRAMES",
    "HotpotQA",
    "Natural Questions",
    "ELI5",
    "ASQA",
    "ALCE",
    "Qasper",
    "MuSiQue",
    "2WikiMultihopQA",
    "KILT",
    "HaluBench",
    "LongBench",
    "RULER",
    "HELM",
    "OpenCompass",
    "Chatbot Arena",
    "BigCodeBench",
    "LiveCodeBench",
    "HumanEval",
    "MMLU-Pro",
    "Arena-Hard",
]

TECHNICAL_SUBJECTS = [
    "LangGraph",
    "LangChain",
    "LlamaIndex",
    "AutoGen",
    "OpenAI Agents SDK",
    "OpenAI Responses API",
    "OpenAI tracing",
    "Anthropic MCP",
    "Haystack",
    "RAGAS",
    "DSPy",
    "CrewAI",
    "OpenHands",
    "SWE-agent",
    "Aider",
    "Pydantic AI",
    "Semantic Kernel",
    "Vercel AI SDK",
    "Transformers Agents",
    "Hugging Face smolagents",
    "Qdrant",
    "Milvus",
    "Weaviate",
    "Chroma",
    "FAISS",
    "Elasticsearch vector search",
    "Postgres pgvector",
    "Cohere rerank",
    "Jina embeddings",
    "ColBERT",
    "vLLM",
    "SGLang",
    "llama.cpp",
    "Ollama",
    "LiteLLM",
    "Ray Serve",
    "FastAPI",
    "DuckDB",
    "Polars",
]

FACT_CHECK_SUBJECTS = [
    "FEVER",
    "Climate-FEVER",
    "COVID-Fact",
    "LIAR",
    "ClaimBuster",
    "IFCN Code of Principles",
    "PolitiFact",
    "Snopes",
    "Full Fact",
    "AP Fact Check",
    "Reuters Fact Check",
    "AFP Fact Check",
    "Logically Facts",
    "EUvsDisinfo",
    "Media Bias/Fact Check",
    "Google Fact Check Tools",
    "ClaimReview markup",
    "Poynter fact-checking guidelines",
    "WHO mythbusters pages",
    "CDC health misinformation pages",
    "NASA climate evidence pages",
    "NOAA climate summaries",
    "IPCC reports",
    "Our World in Data",
    "World Bank data",
    "UN data portal",
    "OECD data",
    "BLS statistics",
    "FRED economic data",
    "SEC company filings",
]

DATA_ANALYSIS_SUBJECTS = [
    "pandas",
    "NumPy",
    "SciPy",
    "scikit-learn",
    "statsmodels",
    "PyTorch",
    "TensorFlow",
    "JAX",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "Polars",
    "DuckDB",
    "Apache Arrow",
    "Dask",
    "Ray Data",
    "Spark MLlib",
    "Vega-Lite",
    "Altair",
    "Plotly",
    "Matplotlib",
    "Seaborn",
    "Great Expectations",
    "dbt-core",
    "OpenML Python",
    "Kaggle datasets",
    "Hugging Face Datasets",
    "UCI Machine Learning Repository",
    "BigQuery public datasets",
    "DataHub",
    "Evidently AI",
    "WhyLabs",
    "MLflow",
    "Weights and Biases",
    "DVC",
]

FOCUSES = [
    "primary-source identity",
    "claim-scope boundary",
    "implementation locator",
    "evaluation-protocol locator",
    "temporal-validity boundary",
    "evidence-sufficiency boundary",
    "conflict-resolution boundary",
    "citation-faithfulness boundary",
    "source-authority hierarchy",
    "public-access boundary",
    "reproducibility-artifact boundary",
    "limitation-caveat boundary",
    "negative-evidence boundary",
    "version-specific documentation boundary",
    "benchmark-comparability boundary",
    "dataset-card boundary",
    "leaderboard-stability boundary",
    "license-and-usage boundary",
    "method-vs-implementation boundary",
    "ablation-claim boundary",
]


TEMPLATES = {
    "academic": [
        (
            "conceptual_methodology",
            "Compare {subject} with adjacent work on search-augmented reasoning: what claims are directly supported, and what should remain uncertain?",
        ),
        (
            "comparative_research",
            "As of {date}, compare the public evidence for {subject} and two related methods; identify source boundaries, shared assumptions, and unsupported extrapolations.",
        ),
        (
            "claim_decomposition_and_verification",
            "Decompose a claim about {subject} into verifiable subclaims, then state what primary academic or project sources would be sufficient for each subclaim.",
        ),
        (
            "negative_evidence_search",
            "For {subject}, search for public evidence that would refute a broad capability claim and explain what absence of evidence can and cannot prove.",
        ),
        (
            "unanswerable_detection",
            "As of {date}, determine whether public sources are sufficient to answer a narrow question about {subject} with a {focus} focus, or whether uncertainty is required.",
        ),
    ],
    "deep_search": [
        (
            "leaderboard_verification",
            "For {subject}, verify the correct public source boundary for a leaderboard-style claim with a {focus} focus.",
        ),
        (
            "multi_source_timeline_reconstruction",
            "Reconstruct the public evidence timeline for {subject} as a deep-search or agent-evaluation artifact, and state what cannot be concluded from the sources.",
        ),
        (
            "comparative_research",
            "Compare {subject} with two other deep-search or browsing-agent benchmarks; focus on task design, evidence requirements, and comparability limits.",
        ),
        (
            "adversarial_conflict_resolution",
            "Resolve conflicting public claims about {subject}: which sources should an agent trust first, and what answer boundary should it report?",
        ),
        (
            "current_status_verification",
            "As of {date}, verify the current public status of {subject} and explain whether stale sources could mislead a research agent.",
        ),
    ],
    "technical": [
        (
            "documentation_change_tracking",
            "Track documentation/version evidence for {subject}: how should an agent verify whether a technical API claim applies to the current documentation?",
        ),
        (
            "repo_implementation_audit",
            "Audit {subject}: what public evidence is sufficient to verify its implementation boundary, and what remains uncertain?",
        ),
        (
            "implementation_verification",
            "As of {date}, verify whether a public implementation claim about {subject} is supported by official docs, repository evidence, or release notes.",
        ),
        (
            "adversarial_conflict_resolution",
            "Resolve conflicting documentation or repository evidence about {subject}; identify the source-authority hierarchy and the safest answer.",
        ),
        (
            "current_status_verification",
            "Verify the current status of {subject} with a {focus} focus, and explain how an agent should avoid relying on outdated tutorials.",
        ),
    ],
    "fact_check": [
        (
            "live_fact_verification",
            "As of {date}, verify a public claim involving {subject}; separate direct evidence, contextual evidence, and claims that require uncertainty.",
        ),
        (
            "claim_decomposition_and_verification",
            "Decompose a fact-checking claim about {subject} into evidence-bearing subclaims and identify which public sources would be authoritative.",
        ),
        (
            "adversarial_conflict_resolution",
            "Resolve conflicting public claims about {subject}; explain which sources should dominate and where uncertainty remains.",
        ),
        (
            "negative_evidence_search",
            "For {subject}, determine whether the absence of a public fact-check or official statement is enough to reject a claim.",
        ),
        (
            "unanswerable_detection",
            "As of {date}, decide whether public evidence is sufficient to answer a narrow claim about {subject} with a {focus} focus.",
        ),
    ],
    "data_analysis": [
        (
            "documentation_change_tracking",
            "Track documentation/version evidence for {subject}: how should an agent verify whether a data-analysis API or metric claim is current?",
        ),
        (
            "implementation_verification",
            "As of {date}, verify whether a public implementation or API claim about {subject} is supported by official docs, code, or release notes.",
        ),
        (
            "claim_decomposition_and_verification",
            "Decompose a data-analysis claim about {subject} into verifiable subclaims, including dataset, metric, and version boundaries.",
        ),
        (
            "comparative_research",
            "Compare {subject} with two adjacent data-analysis tools or datasets; focus on evidence boundaries and unsupported performance claims.",
        ),
        (
            "unanswerable_detection",
            "Determine whether public sources are sufficient to answer a narrow question about {subject} with a {focus} focus, or whether uncertainty is required.",
        ),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="raes-bench/raes-bench-v12_process/public_questions.jsonl",
        help="Original 400-row public questions JSONL.",
    )
    parser.add_argument(
        "--output",
        default="raes-bench/raes-bench-v12_process/public_questions_1k.jsonl",
        help="Merged 1K public questions JSONL.",
    )
    parser.add_argument(
        "--additions-output",
        default="raes-bench/raes-bench-v12_process/public_questions_1k_synthetic_additions.jsonl",
        help="Synthetic additions only.",
    )
    parser.add_argument(
        "--report",
        default="raes-bench/raes-bench-v12_process/public_questions_1k_report.json",
        help="Expansion report path.",
    )
    parser.add_argument("--target-size", type=int, default=1000)
    parser.add_argument("--date", default="2026-06-02")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def next_values(current: Counter, targets: dict[str, int]) -> list[str]:
    values: list[str] = []
    for key, target in targets.items():
        needed = target - current.get(key, 0)
        if needed < 0:
            raise ValueError(f"target for {key} is below existing count")
        values.extend([key] * needed)
    return values


def ordered_assignments(values: list[str]) -> list[str]:
    """Spread labels deterministically instead of grouping them in blocks."""
    counts = Counter(values)
    order = sorted(counts, key=lambda key: (-counts[key], key))
    out: list[str] = []
    while counts:
        for key in order:
            if counts.get(key, 0) <= 0:
                continue
            out.append(key)
            counts[key] -= 1
            if counts[key] == 0:
                del counts[key]
        order = [key for key in order if key in counts]
    return out


def category_subjects(category: str) -> list[str]:
    return {
        "academic": ACADEMIC_SUBJECTS,
        "deep_search": DEEP_SEARCH_SUBJECTS,
        "technical": TECHNICAL_SUBJECTS,
        "fact_check": FACT_CHECK_SUBJECTS,
        "data_analysis": DATA_ANALYSIS_SUBJECTS,
    }[category]


def generate_additions(
    existing: list[dict],
    *,
    target_size: int,
    date: str,
) -> list[dict]:
    current_category = Counter(row["category"] for row in existing)
    current_difficulty = Counter(row["difficulty"] for row in existing)
    current_split = Counter(row["split"] for row in existing)
    category_values = ordered_assignments(next_values(current_category, CATEGORY_TARGETS))
    difficulty_values = ordered_assignments(next_values(current_difficulty, DIFFICULTY_TARGETS))
    split_values = ordered_assignments(next_values(current_split, SPLIT_TARGETS))
    needed = target_size - len(existing)
    if needed != len(category_values):
        raise ValueError(f"category target yields {len(category_values)} additions, expected {needed}")
    if needed != len(difficulty_values) or needed != len(split_values):
        raise ValueError("difficulty or split targets do not match requested target size")

    existing_questions = {row["question"] for row in existing}
    existing_ids = {row["id"] for row in existing}
    per_category_index: Counter = Counter()
    additions: list[dict] = []
    next_numeric_id = len(existing) + 1

    for i, category in enumerate(category_values):
        subjects = category_subjects(category)
        templates = TEMPLATES[category]
        local_idx = per_category_index[category]
        per_category_index[category] += 1
        task_type, template = templates[local_idx % len(templates)]
        subject = subjects[(local_idx + local_idx // len(templates)) % len(subjects)]
        focus = FOCUSES[(local_idx * 3 + i) % len(FOCUSES)]
        question = template.format(subject=subject, focus=focus, date=date)

        base_question = question
        # If a generated wording collides exactly, rotate the focus and then add
        # a secondary evidence angle. This keeps the task natural while making
        # the public prompt unique.
        collision_steps = 0
        while question in existing_questions:
            collision_steps += 1
            focus = FOCUSES[(local_idx * 3 + i + collision_steps) % len(FOCUSES)]
            question = template.format(subject=subject, focus=focus, date=date)
            if question == base_question or question in existing_questions:
                secondary_focus = FOCUSES[(local_idx * 5 + i + collision_steps) % len(FOCUSES)]
                question = (
                    template.format(subject=subject, focus=focus, date=date)
                    + f" Include a secondary check on the {secondary_focus}."
                )
            if collision_steps > len(FOCUSES) * 2:
                question = (
                    template.format(subject=subject, focus=focus, date=date)
                    + f" Treat this as synthetic expansion item {len(additions) + 1}."
                )

        row_id = f"raes-v12-synth-{next_numeric_id:03d}"
        while row_id in existing_ids:
            next_numeric_id += 1
            row_id = f"raes-v12-synth-{next_numeric_id:03d}"
        next_numeric_id += 1

        row = {
            "id": row_id,
            "category": category,
            "domain": category,
            "task_type": task_type,
            "difficulty": difficulty_values[i],
            "question": question,
            "split": split_values[i],
        }
        additions.append(row)
        existing_questions.add(question)
        existing_ids.add(row_id)

    return additions


def validate_public_rows(rows: list[dict], target_size: int) -> None:
    if len(rows) != target_size:
        raise ValueError(f"expected {target_size} rows, found {len(rows)}")
    ids = [row.get("id") for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate ids found")
    questions = [row.get("question") for row in rows]
    if len(questions) != len(set(questions)):
        raise ValueError("duplicate questions found")
    for row in rows:
        missing = [field for field in PUBLIC_FIELDS if field not in row]
        if missing:
            raise ValueError(f"{row.get('id')} missing fields: {missing}")
        leaked = GOLD_FIELDS.intersection(row)
        if leaked:
            raise ValueError(f"{row.get('id')} leaks gold fields: {sorted(leaked)}")


def distribution(rows: list[dict], field: str) -> dict[str, int]:
    return dict(Counter(str(row.get(field, "")) for row in rows))


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    additions_path = Path(args.additions_output)
    report_path = Path(args.report)

    existing = load_jsonl(input_path)
    additions = generate_additions(existing, target_size=args.target_size, date=args.date)
    merged = existing + additions
    validate_public_rows(merged, args.target_size)

    write_jsonl(output_path, merged)
    write_jsonl(additions_path, additions)

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "additions_output": str(additions_path),
        "target_size": args.target_size,
        "original_rows": len(existing),
        "synthetic_additions": len(additions),
        "date_in_prompts": args.date,
        "note": (
            "This is an agent-facing public-question expansion for trajectory collection. "
            "No gold answers or gold sources were generated."
        ),
        "distributions": {
            "category": distribution(merged, "category"),
            "difficulty": distribution(merged, "difficulty"),
            "split": distribution(merged, "split"),
            "task_type": distribution(merged, "task_type"),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md_path = report_path.with_suffix(".md")
    md_path.write_text(
        "# RAES Public Questions 1K Expansion\n\n"
        f"- Input: `{input_path}`\n"
        f"- Output: `{output_path}`\n"
        f"- Additions only: `{additions_path}`\n"
        f"- Original rows: `{len(existing)}`\n"
        f"- Synthetic additions: `{len(additions)}`\n"
        f"- Target size: `{args.target_size}`\n"
        f"- Date used in prompts: `{args.date}`\n\n"
        "This file is intended for teacher trajectory collection and SFT data generation. "
        "It does not include gold answers or gold sources.\n\n"
        "## Distributions\n\n"
        "```json\n"
        + json.dumps(report["distributions"], ensure_ascii=False, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
