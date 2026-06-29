import argparse
import csv
import json
from collections import defaultdict
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


def pair_key(drug1, drug2):
    return f"{drug1}||{drug2}"


def add_vocab(vocab, value):
    if value not in vocab:
        vocab[value] = len(vocab) + 1
    return vocab[value]


def read_zeroddi_drugs(all_csv):
    drugs = {}
    with open(all_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for drug_key, smiles_key in (("drug1", "smiles1"), ("drug2", "smiles2")):
                drug_id = row[drug_key]
                smiles = row[smiles_key]
                if drug_id and smiles and drug_id not in drugs:
                    drugs[drug_id] = smiles
    return drugs


def read_split_pairs(files):
    pairs = []
    for file in files:
        with open(file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pairs.append((row["drug1"], row["drug2"]))
    return pairs


def read_molecbionet_drugs(node_info_file):
    drugs = {}
    with open(node_info_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("Type") != "Drug":
                continue
            smiles = row.get("SMILES", "")
            if not smiles:
                continue
            drugs[row["ID"]] = {
                "smiles": smiles,
                "inchikey": row.get("Node", ""),
            }
    return drugs


def read_node_types(node_type_file):
    types = {}
    with open(node_type_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            types[row["ID"]] = row["Type"]
    return types


def map_drugs_by_smiles(zeroddi_drugs, molecbionet_drugs):
    exact_smiles_to_nodes = defaultdict(list)
    no_stereo_smiles_to_nodes = defaultdict(list)
    inchikey_to_nodes = defaultdict(list)
    inchikey_first_block_to_nodes = defaultdict(list)
    for node_id, info in molecbionet_drugs.items():
        smiles = info["smiles"]
        exact_smiles_to_nodes[canonical_smiles(smiles)].append(node_id)
        no_stereo_smiles_to_nodes[canonical_smiles(smiles, isomeric=False)].append(node_id)
        if info["inchikey"]:
            inchikey_to_nodes[info["inchikey"]].append(node_id)
            inchikey_first_block_to_nodes[info["inchikey"].split("-")[0]].append(node_id)
        key, first_block = inchikey_layers(smiles)
        if key:
            inchikey_to_nodes[key].append(node_id)
        if first_block:
            inchikey_first_block_to_nodes[first_block].append(node_id)

    mapping = {}
    methods = {}
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
        for method, nodes in candidates:
            if nodes:
                mapping[drug_id] = nodes[0]
                methods[drug_id] = method
                break
    return mapping, methods


def read_biokg_edges(biokg_file, exclude_relations):
    adjacency = defaultdict(list)
    edge_lookup = defaultdict(list)
    with open(biokg_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            relation = row["Relation"]
            if relation in exclude_relations:
                continue
            head = row["Node 1"]
            tail = row["Node 2"]
            adjacency[head].append((tail, relation))
            adjacency[tail].append((head, relation))
            key = tuple(sorted((head, tail)))
            edge_lookup[key].append(relation)
    return adjacency, edge_lookup


def token_to_ids(token, vocabs, node_types):
    entity, relation, side, distance = token
    entity_type = node_types.get(entity, "Unknown")
    return [
        add_vocab(vocabs["entity"], entity),
        add_vocab(vocabs["type"], entity_type),
        add_vocab(vocabs["relation"], relation),
        add_vocab(vocabs["side"], side),
        add_vocab(vocabs["distance"], distance),
    ]


def build_pair_tokens(
    drug1_node,
    drug2_node,
    adjacency,
    edge_lookup,
    node_types,
    vocabs,
    max_tokens,
    max_neighbors_per_side,
    max_bridge_edges,
):
    drug1_neighbors = adjacency.get(drug1_node, [])[:max_neighbors_per_side]
    drug2_neighbors = adjacency.get(drug2_node, [])[:max_neighbors_per_side]
    drug1_entities = {entity for entity, _ in drug1_neighbors}
    drug2_entities = {entity for entity, _ in drug2_neighbors}
    common_entities = drug1_entities & drug2_entities

    raw_tokens = []
    seen = set()

    def append_token(entity, relation, side, distance, priority):
        key = (entity, relation, side, distance)
        if key in seen:
            return
        seen.add(key)
        raw_tokens.append((priority, (entity, relation, side, distance)))

    for entity, relation in drug1_neighbors:
        side = "shared" if entity in common_entities else "drug1"
        append_token(entity, relation, side, "1hop", 0 if side == "shared" else 2)

    for entity, relation in drug2_neighbors:
        side = "shared" if entity in common_entities else "drug2"
        append_token(entity, relation, side, "1hop", 0 if side == "shared" else 2)

    bridge_count = 0
    for entity1 in drug1_entities:
        if bridge_count >= max_bridge_edges:
            break
        for entity2 in drug2_entities:
            relations = edge_lookup.get(tuple(sorted((entity1, entity2))))
            if not relations:
                continue
            for relation in relations:
                append_token(entity1, relation, "drug1", "2hop", 1)
                append_token(entity2, relation, "drug2", "2hop", 1)
                bridge_count += 1
                if bridge_count >= max_bridge_edges:
                    break
            if bridge_count >= max_bridge_edges:
                break

    raw_tokens.sort(key=lambda item: item[0])
    return [
        token_to_ids(token, vocabs, node_types)
        for _, token in raw_tokens[:max_tokens]
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Build pair-level KG evidence tokens from MolecBioNet DrugBank KG."
    )
    parser.add_argument("--zeroddi-csv", default="data/DrugBank5.1.9/DDI_final2.csv")
    parser.add_argument(
        "--split-files",
        nargs="+",
        default=[
            "data/DrugBank5.1.9/seen_random/train.csv",
            "data/DrugBank5.1.9/seen_random/val.csv",
            "data/DrugBank5.1.9/seen_random/test.csv",
        ],
    )
    parser.add_argument("--molecbionet-dir", required=True)
    parser.add_argument(
        "--output",
        default="data/DrugBank5.1.9/kg/molecbionet_pair_kg_seen.json",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-neighbors-per-side", type=int, default=32)
    parser.add_argument("--max-bridge-edges", type=int, default=8)
    parser.add_argument("--include-ddi-edges", action="store_true")
    args = parser.parse_args()

    molecbionet_dir = Path(args.molecbionet_dir)
    node_info_file = molecbionet_dir / "DrugBank_Node_Information.txt"
    node_type_file = molecbionet_dir / "Node_Type.txt"
    biokg_file = molecbionet_dir / "BioKG_DrugBank.txt"

    zeroddi_drugs = read_zeroddi_drugs(args.zeroddi_csv)
    split_pairs = read_split_pairs(args.split_files)
    molecbionet_drugs = read_molecbionet_drugs(node_info_file)
    node_types = read_node_types(node_type_file)
    drug_mapping, drug_mapping_methods = map_drugs_by_smiles(zeroddi_drugs, molecbionet_drugs)

    exclude_relations = set() if args.include_ddi_edges else {"DDI"}
    adjacency, edge_lookup = read_biokg_edges(biokg_file, exclude_relations)

    vocabs = {
        "entity": {},
        "type": {},
        "relation": {},
        "side": {},
        "distance": {},
    }
    pairs = {}
    mapped_pairs = 0
    pairs_with_tokens = 0

    for drug1, drug2 in split_pairs:
        drug1_node = drug_mapping.get(drug1)
        drug2_node = drug_mapping.get(drug2)
        if drug1_node is None or drug2_node is None:
            continue
        mapped_pairs += 1
        tokens = build_pair_tokens(
            drug1_node=drug1_node,
            drug2_node=drug2_node,
            adjacency=adjacency,
            edge_lookup=edge_lookup,
            node_types=node_types,
            vocabs=vocabs,
            max_tokens=args.max_tokens,
            max_neighbors_per_side=args.max_neighbors_per_side,
            max_bridge_edges=args.max_bridge_edges,
        )
        if tokens:
            pairs_with_tokens += 1
        pairs[pair_key(drug1, drug2)] = tokens

    output = {
        "metadata": {
            "zeroddi_csv": args.zeroddi_csv,
            "split_files": args.split_files,
            "molecbionet_dir": str(molecbionet_dir),
            "exclude_relations": sorted(exclude_relations),
            "max_tokens": args.max_tokens,
            "max_neighbors_per_side": args.max_neighbors_per_side,
            "max_bridge_edges": args.max_bridge_edges,
            "zeroddi_drugs": len(zeroddi_drugs),
            "mapped_drugs": len(drug_mapping),
            "drug_mapping_methods": {
                method: list(drug_mapping_methods.values()).count(method)
                for method in sorted(set(drug_mapping_methods.values()))
            },
            "split_pairs": len(split_pairs),
            "mapped_pairs": mapped_pairs,
            "pairs_with_tokens": pairs_with_tokens,
        },
        "feature_vocab_sizes": {
            name: len(vocab) + 1 for name, vocab in vocabs.items()
        },
        "feature_vocabs": vocabs,
        "drug_mapping": drug_mapping,
        "pairs": pairs,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, separators=(",", ":"))
    tmp_path.replace(output_path)

    print(f"ZeroDDI drugs: {len(zeroddi_drugs)}")
    print(f"Mapped drugs: {len(drug_mapping)}")
    print(f"Split pairs: {len(split_pairs)}")
    print(f"Mapped pairs: {mapped_pairs}")
    print(f"Pairs with KG tokens: {pairs_with_tokens}")
    print(f"Saved KG evidence to {output_path}")


if __name__ == "__main__":
    main()
