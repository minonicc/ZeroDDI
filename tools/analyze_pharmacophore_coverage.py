"""Compute S0 pharmacophore and fixed-prefix coverage diagnostics."""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from rdkit.Chem import AllChem

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.left.graph.pharmacophore import PharmacophoreExtractor


THRESHOLDS = (64, 128, 144, 256, 512)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-file", required=True)
    parser.add_argument("--pair-file", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def prefix_coverage(types_a, types_b, limit):
    pair_count = len(types_a) * len(types_b)
    selected = min(pair_count, limit)
    if pair_count == 0:
        return 0.0, 0.0
    flat_indices = np.arange(selected)
    selected_a = set((flat_indices // len(types_b)).tolist())
    selected_b = set((flat_indices % len(types_b)).tolist())
    node_coverage = (len(selected_a) + len(selected_b)) / (len(types_a) + len(types_b))
    all_types = set(types_a) | set(types_b)
    retained_types = {types_a[index] for index in selected_a}
    retained_types.update(types_b[index] for index in selected_b)
    type_coverage = len(retained_types) / max(len(all_types), 1)
    return node_coverage, type_coverage


def main():
    args = parse_args()
    all_data = pd.read_csv(args.all_file)
    pairs = pd.read_csv(args.pair_file)
    drug_smiles = {}
    for side in ("1", "2"):
        drug_smiles.update(zip(all_data[f"drug{side}"], all_data[f"smiles{side}"]))

    extractor = PharmacophoreExtractor()
    drug_types = {}
    invalid_drugs = []
    family_counts = Counter()
    for drug, smiles in drug_smiles.items():
        mol = AllChem.MolFromSmiles(smiles)
        if mol is None:
            invalid_drugs.append(drug)
            drug_types[drug] = []
            continue
        features = extractor(mol)
        types = [feature["family"] for feature in features]
        drug_types[drug] = types
        family_counts.update(types)

    drug_counts = [len(types) for types in drug_types.values()]
    pair_counts = []
    prefix_node_coverage = {str(limit): [] for limit in THRESHOLDS}
    prefix_type_coverage = {str(limit): [] for limit in THRESHOLDS}
    for row in pairs.itertuples(index=False):
        types_a = drug_types.get(row.drug1, [])
        types_b = drug_types.get(row.drug2, [])
        pair_count = len(types_a) * len(types_b)
        pair_counts.append(pair_count)
        for limit in THRESHOLDS:
            node_coverage, type_coverage = prefix_coverage(types_a, types_b, limit)
            prefix_node_coverage[str(limit)].append(node_coverage)
            prefix_type_coverage[str(limit)].append(type_coverage)

    report = {
        "all_file": args.all_file,
        "pair_file": args.pair_file,
        "num_unique_drugs": len(drug_types),
        "invalid_drugs": invalid_drugs,
        "pharmacophore_count_by_drug": {
            drug: len(types) for drug, types in sorted(drug_types.items())
        },
        "pharmacophores_per_drug": describe(drug_counts),
        "pharmacophore_family_counts": dict(sorted(family_counts.items())),
        "pairs_per_drug_pair": describe(pair_counts),
        "fraction_pair_count_over": {
            str(limit): float(np.mean(np.asarray(pair_counts) > limit))
            for limit in THRESHOLDS
        },
        "fixed_prefix_node_coverage": {
            limit: describe(values) for limit, values in prefix_node_coverage.items()
        },
        "fixed_prefix_type_coverage": {
            limit: describe(values) for limit, values in prefix_type_coverage.items()
        },
    }
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
