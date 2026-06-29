import argparse
import csv
import json
from pathlib import Path

try:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except ImportError:
    pass


def mol_from_smiles(smiles):
    try:
        from rdkit import Chem
    except ImportError:
        return None

    mol = Chem.MolFromSmiles(smiles)
    return mol


def canonical_smiles(smiles, isomeric=True):
    try:
        from rdkit import Chem
    except ImportError:
        return smiles.strip()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles.strip()
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)


def inchikey_layers(smiles):
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None, None
    try:
        from rdkit.Chem import inchi
    except ImportError:
        return None, None

    key = inchi.MolToInchiKey(mol)
    first_block = key.split("-")[0] if key else None
    return key, first_block


def load_zeroddi_drugs(path):
    drugs = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for drug_key, smiles_key in (("drug1", "smiles1"), ("drug2", "smiles2")):
                drug_id = row[drug_key]
                smiles = row[smiles_key]
                if drug_id and smiles and drug_id not in drugs:
                    drugs[drug_id] = smiles
    return drugs


def load_molecbionet_drugs(path):
    drugs = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("Type") != "Drug":
                continue
            node_id = row["ID"]
            smiles = row.get("SMILES", "")
            inchikey = row.get("Node", "")
            if smiles:
                drugs[node_id] = {"smiles": smiles, "inchikey": inchikey}
    return drugs


def inspect_coverage(zeroddi_drugs, molecbionet_drugs):
    exact_smiles_to_nodes = {}
    no_stereo_smiles_to_nodes = {}
    inchikey_to_nodes = {}
    inchikey_first_block_to_nodes = {}
    for node_id, info in molecbionet_drugs.items():
        smiles = info["smiles"]
        exact_smiles_to_nodes.setdefault(canonical_smiles(smiles), []).append(node_id)
        no_stereo_smiles_to_nodes.setdefault(canonical_smiles(smiles, isomeric=False), []).append(node_id)
        if info["inchikey"]:
            inchikey_to_nodes.setdefault(info["inchikey"], []).append(node_id)
            inchikey_first_block_to_nodes.setdefault(info["inchikey"].split("-")[0], []).append(node_id)
        key, first_block = inchikey_layers(smiles)
        if key:
            inchikey_to_nodes.setdefault(key, []).append(node_id)
        if first_block:
            inchikey_first_block_to_nodes.setdefault(first_block, []).append(node_id)

    mapping = {}
    methods = {}
    missing = {}
    for drug_id, smiles in zeroddi_drugs.items():
        key, first_block = inchikey_layers(smiles)
        candidates = [
            ("inchikey", inchikey_to_nodes.get(key, []) if key else []),
            (
                "inchikey_first_block",
                inchikey_first_block_to_nodes.get(first_block, []) if first_block else [],
            ),
            ("smiles_exact", exact_smiles_to_nodes.get(canonical_smiles(smiles), [])),
            (
                "smiles_no_stereo",
                no_stereo_smiles_to_nodes.get(canonical_smiles(smiles, isomeric=False), []),
            ),
        ]
        nodes = []
        method = None
        for method_name, method_nodes in candidates:
            if method_nodes:
                nodes = method_nodes
                method = method_name
                break
        if nodes:
            mapping[drug_id] = nodes[0]
            methods[drug_id] = method
        else:
            missing[drug_id] = smiles

    return mapping, methods, missing


def main():
    parser = argparse.ArgumentParser(
        description="Inspect SMILES coverage between ZeroDDI drugs and MolecBioNet DrugBank KG drugs."
    )
    parser.add_argument(
        "--zeroddi-csv",
        default="data/DrugBank5.1.9/DDI_final2.csv",
        help="ZeroDDI all-pair CSV with drug1/drug2/smiles1/smiles2 columns.",
    )
    parser.add_argument(
        "--molecbionet-drug-info",
        required=True,
        help="MolecBioNet data/DrugBank/DrugBank_Node_Information.txt path.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path for DB drug id to MolecBioNet node id mapping.",
    )
    args = parser.parse_args()

    zeroddi_drugs = load_zeroddi_drugs(args.zeroddi_csv)
    molecbionet_drugs = load_molecbionet_drugs(args.molecbionet_drug_info)
    mapping, methods, missing = inspect_coverage(zeroddi_drugs, molecbionet_drugs)

    total = len(zeroddi_drugs)
    covered = len(mapping)
    coverage = covered / total if total else 0.0
    print(f"ZeroDDI drugs: {total}")
    print(f"MolecBioNet KG drug nodes: {len(molecbionet_drugs)}")
    print(f"Mapped drugs: {covered}")
    print(f"Coverage: {coverage:.4f}")
    method_counts = {}
    for method in methods.values():
        method_counts[method] = method_counts.get(method, 0) + 1
    print(f"Mapping methods: {method_counts}")

    if missing:
        print("First missing drug ids:")
        for drug_id in list(missing)[:20]:
            print(drug_id)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump({"mapping": mapping, "methods": methods}, f, indent=2, sort_keys=True)
        print(f"Saved mapping to {output_path}")


if __name__ == "__main__":
    main()
