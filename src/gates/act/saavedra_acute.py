from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

EXPECTED_SHA256 = "b8eb6ede15a195ab59a81610623df7c836101a40c738431065be8e32ace30cfa"
SPECIES_SHEETS = {"rat": "Figure4_rat", "human": "Figure4_human"}
CONCENTRATIONS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
POLICIES = {
    "top3": (4, 5, 6),
    "maximin": (0, 3, 6),
    "low3": (0, 1, 2),
}
EFFECT_THRESHOLD = 0.40


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_wmfr(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    workbook = load_workbook(path, read_only=True, data_only=False)
    records: list[dict[str, object]] = []
    species_profiles = []
    for species, sheet_name in SPECIES_SHEETS.items():
        sheet = workbook[sheet_name]
        endpoint_headers = [
            {"column": column, "label": str(sheet.cell(1, column).value)}
            for column in range(1, sheet.max_column + 1)
            if sheet.cell(1, column).value is not None
        ]
        compound_rows = [
            row for row in range(2, sheet.max_row + 1) if isinstance(sheet.cell(row, 1).value, str)
        ]
        compounds = [str(sheet.cell(row, 1).value).strip() for row in compound_rows]
        replicate_counts = []
        vehicle_vectors = []
        for compound_row, compound in zip(compound_rows, compounds, strict=True):
            vehicle = [
                float(sheet.cell(compound_row, column).value)
                for column in range(3, 21)
                if isinstance(sheet.cell(compound_row, column).value, (int, float))
            ]
            vehicle_vectors.append(vehicle)
            for rank, concentration in enumerate(CONCENTRATIONS):
                row = compound_row + rank + 1
                observed_label = str(sheet.cell(row, 2).value).strip()
                values = []
                for column in range(3, 21):
                    value = sheet.cell(row, column).value
                    if isinstance(value, str) and value.startswith("="):
                        coordinate = sheet.cell(row, column).coordinate
                        raise ValueError(f"formula in {sheet_name}!{coordinate}")
                    if isinstance(value, (int, float)) and math.isfinite(float(value)):
                        values.append((column - 2, float(value)))
                replicate_counts.append(len(values))
                for replicate, value in values:
                    records.append(
                        {
                            "species": species,
                            "compound": compound,
                            "concentration_rank": rank,
                            "concentration_um": concentration,
                            "concentration_label": observed_label,
                            "replicate_column": replicate,
                            "wmfr_percent_baseline": value,
                        }
                    )
        species_profiles.append(
            {
                "species": species,
                "sheet": sheet_name,
                "rows": sheet.max_row,
                "columns": sheet.max_column,
                "endpoint_headers": endpoint_headers,
                "compounds": compounds,
                "treatment_replicate_count_distribution": {
                    str(count): int(replicate_counts.count(count))
                    for count in sorted(set(replicate_counts))
                },
                "vehicle_replicates_per_reused_row": sorted(
                    {len(values) for values in vehicle_vectors}
                ),
                "unique_vehicle_vectors": len({tuple(values) for values in vehicle_vectors}),
            }
        )
    frame = pd.DataFrame(records)
    audit = {
        "schema": "gates-act-saavedra-acute-audit-v1",
        "source": {
            "dataset_doi": "10.23719/1518437",
            "paper_doi": "10.1093/toxsci/kfab008",
            "download_url": (
                "https://pasteur.epa.gov/uploads/10.23719/1518437/"
                "MEA%20data_figures_normalized.xlsx"
            ),
            "sha256": _sha256(path),
        },
        "species_profiles": species_profiles,
        "extracted_rows": len(frame),
        "compound_species_units": int(frame.groupby(["compound", "species"]).ngroups),
        "structure": {
            "plate_or_well_identifiers_present": False,
            "stable_cross_concentration_replicate_identity": False,
            "vehicle_rows_are_reused_within_species": all(
                profile["unique_vehicle_vectors"] == 1 for profile in species_profiles
            ),
            "assay_type": "acute baseline-normalized concentration response",
            "longitudinal_dose_time_eligible": False,
        },
    }
    checks = {
        "official_file_hash_matches": audit["source"]["sha256"] == EXPECTED_SHA256,
        "twelve_compounds_per_species": all(
            len(profile["compounds"]) == 12 for profile in species_profiles
        ),
        "same_compounds_in_both_species": species_profiles[0]["compounds"]
        == species_profiles[1]["compounds"],
        "seven_concentrations_per_unit": bool(
            frame.groupby(["compound", "species"])["concentration_rank"].nunique().eq(7).all()
        ),
        "five_named_mea_endpoints": all(
            len(profile["endpoint_headers"]) == 5 for profile in species_profiles
        ),
        "treatment_rows_have_five_or_six_values": bool(
            all(
                set(profile["treatment_replicate_count_distribution"]).issubset({"5", "6"})
                for profile in species_profiles
            )
        ),
        "plate_or_well_identifiers_present": False,
        "stable_cross_concentration_replicate_identity": False,
    }
    audit["checks"] = checks
    audit["status"] = "PASS" if all(checks.values()) else "FAIL"
    audit["eligibility"] = (
        "REJECTED: the public workbook lacks plate/well identities and stable cross-concentration "
        "replicate identities, and its treatment replicate counts do not match the reported "
        "six-per-concentration design. Any numerical policy comparison is quarantined as "
        "preliminary descriptive analysis and is ineligible for independent confirmation."
    )
    return frame, audit


def _aggregate_curves(frame: pd.DataFrame) -> pd.DataFrame:
    curves = (
        frame.groupby(
            ["compound", "species", "concentration_rank", "concentration_um"],
            as_index=False,
        )
        .agg(
            mean_wmfr_percent=("wmfr_percent_baseline", "mean"),
            replicate_observations=("wmfr_percent_baseline", "size"),
        )
        .sort_values(["compound", "species", "concentration_rank"])
    )
    curves["absolute_log_effect"] = np.abs(np.log(curves["mean_wmfr_percent"] / 100.0))
    curves["reference_effect_cell"] = curves["absolute_log_effect"] >= EFFECT_THRESHOLD
    return curves


def _evaluate_policy(curves: pd.DataFrame, ranks: tuple[int, ...], name: str) -> pd.DataFrame:
    rows = []
    for (compound, species), group in curves.groupby(["compound", "species"]):
        selected = group.loc[group["concentration_rank"].isin(ranks)]
        reference = bool(group["reference_effect_cell"].any())
        confirmed = bool(selected["reference_effect_cell"].any())
        rows.append(
            {
                "policy": name,
                "compound": compound,
                "species": species,
                "reference_effect": reference,
                "confirmed_effect": confirmed,
                "unresolved": not confirmed,
                "selected_conditions": len(ranks),
                "replicate_observations": int(selected["replicate_observations"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _metrics(decisions: pd.DataFrame) -> dict[str, float | int]:
    reference = decisions["reference_effect"].to_numpy(dtype=bool)
    confirmed = decisions["confirmed_effect"].to_numpy(dtype=bool)
    tp = int((reference & confirmed).sum())
    fp = int((~reference & confirmed).sum())
    fn = int((reference & ~confirmed).sum())
    return {
        "units": len(decisions),
        "reference_effect_units": int(reference.sum()),
        "confirmed_effect_units": int(confirmed.sum()),
        "effect_precision": tp / max(tp + fp, 1),
        "discovery_recall": tp / max(int(reference.sum()), 1),
        "false_discoveries": fp,
        "missed_effect_units": fn,
        "unresolved_units": int((~confirmed).sum()),
        "selected_concentration_conditions": int(decisions["selected_conditions"].sum()),
        "actual_replicate_observations": int(decisions["replicate_observations"].sum()),
    }


def _cluster_bootstrap(
    decisions: pd.DataFrame,
    candidate: str = "top3",
    baseline: str = "maximin",
    seed: int = 20261026,
    replicates: int = 10000,
) -> dict[str, object]:
    by_policy = {name: part for name, part in decisions.groupby("policy")}
    compounds = sorted(decisions["compound"].unique())
    arrays = {}
    for policy in (candidate, baseline):
        part = by_policy[policy]
        grouped = (
            part.assign(
                reference=part["reference_effect"].astype(int),
                true_positive=(part["reference_effect"] & part["confirmed_effect"]).astype(int),
            )
            .groupby("compound")[["reference", "true_positive"]]
            .sum()
        )
        arrays[policy] = grouped.reindex(compounds).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(compounds), size=(replicates, len(compounds)))
    recalls = {}
    for policy in (candidate, baseline):
        totals = arrays[policy][sampled].sum(axis=1)
        recalls[policy] = np.divide(
            totals[:, 1], totals[:, 0], out=np.zeros(replicates), where=totals[:, 0] > 0
        )
    differences = recalls[candidate] - recalls[baseline]
    return {
        "candidate": candidate,
        "baseline": baseline,
        "compound_clusters": len(compounds),
        "replicates": replicates,
        "seed": seed,
        "difference": float(np.mean(differences)),
        "ci95": [
            float(np.quantile(differences, 0.025)),
            float(np.quantile(differences, 0.975)),
        ],
    }


def run_saavedra_benchmark(path: Path, output: Path) -> dict[str, object]:
    frame, audit = extract_wmfr(path)
    curves = _aggregate_curves(frame)
    decisions = pd.concat(
        [_evaluate_policy(curves, ranks, name) for name, ranks in POLICIES.items()],
        ignore_index=True,
    )
    random_rows = []
    random_decisions = []
    for index, ranks in enumerate(itertools.combinations(range(7), 3)):
        name = f"random_{index:02d}"
        part = _evaluate_policy(curves, ranks, name)
        random_decisions.append(part)
        random_rows.append({"policy": name, "ranks": list(ranks), **_metrics(part)})
    random_frame = pd.DataFrame(random_rows)

    aggregate = []
    by_species = []
    for policy, part in decisions.groupby("policy"):
        aggregate.append({"policy": policy, **_metrics(part)})
        for species, species_part in part.groupby("species"):
            by_species.append({"policy": policy, "species": species, **_metrics(species_part)})
    aggregate_frame = pd.DataFrame(aggregate)
    species_frame = pd.DataFrame(by_species)
    bootstrap = _cluster_bootstrap(decisions)
    top = aggregate_frame.set_index("policy").loc["top3"]
    maximin = aggregate_frame.set_index("policy").loc["maximin"]
    top_species = species_frame.loc[species_frame["policy"].eq("top3")].set_index("species")
    maximin_species = species_frame.loc[species_frame["policy"].eq("maximin")].set_index("species")
    numerical_gate = bool(
        top["discovery_recall"] > maximin["discovery_recall"]
        and (top_species["discovery_recall"] >= maximin_species["discovery_recall"]).all()
        and bootstrap["ci95"][0] > 0
    )
    gate = bool(audit["status"] == "PASS" and numerical_gate)

    output.mkdir(parents=True, exist_ok=True)
    (output / "saavedra_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    curves.to_csv(output / "saavedra_wmfr_curves.csv", index=False)
    decisions.to_csv(output / "saavedra_fixed_policy_decisions.csv", index=False)
    aggregate_frame.to_csv(output / "saavedra_fixed_policy_metrics.csv", index=False)
    species_frame.to_csv(output / "saavedra_fixed_policy_by_species.csv", index=False)
    random_frame.to_csv(output / "saavedra_random_subset_metrics.csv", index=False)
    pd.concat(random_decisions, ignore_index=True).to_csv(
        output / "saavedra_random_subset_decisions.csv", index=False
    )
    result = {
        "schema": "gates-act-saavedra-acute-transfer-v1",
        "analysis_status": "REJECTED_AFTER_AUDIT",
        "evidence_class": (
            "independently sourced but provenance-ineligible acute workbook; numerical outputs are "
            "quarantined preliminary description, not confirmation"
        ),
        "eligibility_blocker": audit["eligibility"],
        "effect_reference": (
            "full seven-concentration mean wMFR curve; |log(mean percent baseline / 100)| >= 0.40"
        ),
        "candidate": "top3",
        "baseline": "maximin",
        "matched_risk": "direct observed effect calls only; zero reference false discoveries",
        "aggregate": aggregate_frame.to_dict(orient="records"),
        "by_species": species_frame.to_dict(orient="records"),
        "random_subsets": {
            "count": len(random_frame),
            "mean_discovery_recall": float(random_frame["discovery_recall"].mean()),
            "min_discovery_recall": float(random_frame["discovery_recall"].min()),
            "max_discovery_recall": float(random_frame["discovery_recall"].max()),
        },
        "bootstrap_top3_minus_maximin": bootstrap,
        "numerical_gate_if_the_file_had_been_eligible": "GO" if numerical_gate else "NO-GO",
        "prespecified_gate": "GO" if gate else "NO-GO_DATA_INELIGIBLE",
        "canonical_integration": "NO-GO: failed provenance/identity audit and acute-only task",
        "early_stopping": "disabled",
    }
    (output / "saavedra_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result
