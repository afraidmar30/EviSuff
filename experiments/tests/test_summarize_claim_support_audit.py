import random
import unittest

from experiments.summarize_claim_support_audit import (
    claim_statistics,
    direct_statistics,
    paired_direct_contrasts,
)


class PairedDirectContrastTest(unittest.TestCase):
    @staticmethod
    def output_labels():
        return {
            "source_policy_compliant": "yes",
            "conflict_disclosure_correct": "not_applicable",
            "uncertainty_present": "no",
            "uncertainty_appropriate": "yes",
            "stop_decision": "appropriate",
        }

    def test_full_minus_comparator_preserves_task_pairing_and_direction(self):
        groups = (
            "answerable",
            "uncertainty",
            "conflict",
            "insufficient-evidence",
        )
        systems = ("base", "answer-sft", "no-gate", "full-evisuff")
        outputs = {}
        labels = {}
        claim_counts = {}
        supported_by_system = {
            "base": 0,
            "answer-sft": 1,
            "no-gate": 2,
            "full-evisuff": 4,
        }
        for task_index, group in enumerate(groups):
            task_id = f"task-{task_index}"
            for system in systems:
                output_id = f"{system}::{task_id}"
                outputs[output_id] = {
                    "system": system,
                    "sample_id": task_id,
                    "group": group,
                }
                is_full = system == "full-evisuff"
                labels[output_id] = {
                    "source_policy_compliant": "yes" if is_full else "no",
                    "conflict_disclosure_correct": "yes" if is_full else "no",
                    "uncertainty_present": (
                        "no" if group == "answerable" and is_full else "yes"
                    ),
                    "uncertainty_appropriate": "yes" if is_full else "no",
                    "stop_decision": "appropriate" if is_full else "premature",
                }
                claim_counts[output_id] = (
                    int(task_index < supported_by_system[system]),
                    1,
                    1,
                )

        report = paired_direct_contrasts(
            labels,
            outputs,
            claim_counts,
            iterations=200,
            rng=random.Random(17),
        )

        base = report["base"]["overall"]
        self.assertEqual(base["tasks"], 4)
        self.assertEqual(base["cluster_unit"], "paired task")
        self.assertAlmostEqual(
            base["metrics"]["claim_support_rate"]["estimate"], 1.0
        )
        self.assertAlmostEqual(
            base["metrics"]["source_policy_compliance"]["estimate"], 1.0
        )
        self.assertAlmostEqual(
            base["metrics"]["stop_decision_accuracy"]["estimate"], 1.0
        )
        self.assertAlmostEqual(
            base["metrics"]["premature_finalization_rate"]["estimate"], -1.0
        )
        self.assertAlmostEqual(
            base["metrics"]["uncertainty_precision"]["estimate"], 0.25
        )
        self.assertAlmostEqual(
            report["no-gate"]["overall"]["metrics"]["claim_support_rate"]["estimate"],
            0.5,
        )
        self.assertAlmostEqual(
            report["base"]["conflict"]["metrics"]["conflict_disclosure_accuracy"]["estimate"],
            1.0,
        )
        self.assertNotIn(
            "conflict_disclosure_accuracy",
            report["base"]["answerable"]["metrics"],
        )

    def test_unclear_pairs_remain_in_primary_support_denominator(self):
        outputs = {
            "base::task": {
                "system": "base", "sample_id": "task", "group": "answerable"
            },
            "full-evisuff::task": {
                "system": "full-evisuff", "sample_id": "task", "group": "answerable"
            },
        }
        labels = {
            output_id: self.output_labels() for output_id in outputs
        }
        # (supported, all pairs, non-unclear pairs)
        claim_counts = {
            "base::task": (0, 2, 2),
            "full-evisuff::task": (1, 2, 1),
        }
        report = paired_direct_contrasts(
            labels, outputs, claim_counts, iterations=100, rng=random.Random(5)
        )
        metrics = report["base"]["overall"]["metrics"]
        self.assertAlmostEqual(metrics["claim_support_rate"]["estimate"], 0.5)
        self.assertAlmostEqual(
            metrics["claim_support_rate_judgeable"]["estimate"], 1.0
        )

    def test_paired_support_excludes_tasks_missing_pairs_on_either_side(self):
        outputs = {}
        labels = {}
        for task_id in ("task-1", "task-2"):
            for system in ("base", "full-evisuff"):
                output_id = f"{system}::{task_id}"
                outputs[output_id] = {
                    "system": system,
                    "sample_id": task_id,
                    "group": "answerable",
                }
                labels[output_id] = self.output_labels()
        claim_counts = {
            "base::task-1": (0, 1, 1),
            "full-evisuff::task-1": (1, 1, 1),
            # task-2 has a Full pair but no Base pair and must not enter CSR.
            "full-evisuff::task-2": (0, 1, 1),
        }
        report = paired_direct_contrasts(
            labels, outputs, claim_counts, iterations=100, rng=random.Random(7)
        )
        overall = report["base"]["overall"]
        self.assertEqual(overall["tasks"], 2)
        self.assertEqual(overall["claim_support_paired_tasks"], 1)
        self.assertEqual(overall["metrics"]["claim_support_rate"]["n"], 1)
        self.assertAlmostEqual(
            overall["metrics"]["claim_support_rate"]["estimate"], 1.0
        )

    def test_direct_metrics_use_output_as_unit_and_expected_uncertainty_truth(self):
        groups = (
            "answerable", "uncertainty", "conflict", "insufficient-evidence"
        )
        outputs = {}
        labels = {}
        claim_counts = {}
        uncertainty_present = ("yes", "yes", "yes", "no")
        uncertainty_appropriate = ("no", "yes", "yes", "no")
        stop_decisions = (
            "appropriate", "premature", "unnecessary_search", "appropriate"
        )
        for index, group in enumerate(groups):
            output_id = f"base::{group}"
            outputs[output_id] = {
                "system": "base", "sample_id": f"task-{index}", "group": group,
            }
            labels[output_id] = {
                "source_policy_compliant": "no" if group == "conflict" else "yes",
                "conflict_disclosure_correct": "yes" if group == "conflict" else "not_applicable",
                "uncertainty_present": uncertainty_present[index],
                "uncertainty_appropriate": uncertainty_appropriate[index],
                "stop_decision": stop_decisions[index],
            }
            claim_counts[output_id] = (1 if index == 0 else 0, 1, 1)

        report = direct_statistics(
            labels, outputs, claim_counts, iterations=100, rng=random.Random(11)
        )
        overall = report["base"]["overall"]
        self.assertEqual(overall["outputs"], 4)
        metrics = overall["metrics"]
        self.assertAlmostEqual(metrics["source_policy_compliance"]["estimate"], 0.75)
        self.assertAlmostEqual(metrics["conflict_disclosure_accuracy"]["estimate"], 1.0)
        self.assertAlmostEqual(metrics["uncertainty_precision"]["estimate"], 2 / 3)
        self.assertAlmostEqual(metrics["uncertainty_recall"]["estimate"], 2 / 3)
        self.assertAlmostEqual(metrics["uncertainty_calibration_accuracy"]["estimate"], 0.5)
        self.assertAlmostEqual(metrics["stop_decision_accuracy"]["estimate"], 0.5)
        self.assertAlmostEqual(metrics["premature_finalization_rate"]["estimate"], 0.25)
        self.assertAlmostEqual(metrics["unnecessary_search_rate"]["estimate"], 0.25)
        self.assertAlmostEqual(metrics["claim_support_rate"]["estimate"], 0.25)
        self.assertEqual(report["base"]["conflict"]["outputs"], 1)
        self.assertNotIn(
            "conflict_disclosure_accuracy",
            report["base"]["insufficient-evidence"]["metrics"],
        )

    def test_claim_proportions_include_partial_unsupported_and_unclear(self):
        support_labels = (
            "supported", "partially_supported", "unsupported", "unclear"
        )
        masters = {}
        labels = {}
        outputs = {}
        for index, support_label in enumerate(support_labels):
            output_id = f"base::task-{index}"
            pair_id = f"pair-{index}"
            outputs[output_id] = {
                "system": "base", "sample_id": f"task-{index}",
                "group": "answerable", "automatic_E": index / 3,
            }
            masters[pair_id] = {"output_id": output_id}
            labels[pair_id] = {"support_label": support_label}

        claims, per_output = claim_statistics(
            masters, labels, outputs, iterations=100, rng=random.Random(13)
        )
        base = claims["by_system"]["base"]
        self.assertEqual(base["outputs_with_claims"], 4)
        self.assertEqual(base["claim_pairs"], 4)
        for support_label in support_labels:
            metric = base["proportions"][support_label]
            self.assertEqual(metric["n"], 4)
            self.assertAlmostEqual(metric["estimate"], 0.25)
        self.assertEqual(per_output["base::task-3"], (0, 1, 0))


if __name__ == "__main__":
    unittest.main()
