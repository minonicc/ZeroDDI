import argparse
import csv
import json
import math
import sqlite3
import time
import zlib
from collections import Counter, defaultdict, deque
from pathlib import Path

from build_molecbionet_kg_evidence import (
    map_drugs_by_smiles,
    read_molecbionet_drugs,
    read_node_types,
    read_split_pairs,
    read_zeroddi_drugs,
)


PAD = 0
DISTANCE_TO_ID = {"0": 1, "1": 2, "2": 3, "3+": 4, "unreachable": 5}


def add_vocab(vocab, value):
    if value not in vocab:
        vocab[value] = len(vocab) + 1
    return vocab[value]


def read_directed_kg(path, exclude_relations):
    """Read the original KG without discarding edge endpoints or direction."""
    adjacency = defaultdict(list)
    edges = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            relation = row["Relation"]
            if relation in exclude_relations:
                continue
            source = row["Node 1"]
            target = row["Node 2"]
            edge_id = len(edges)
            edges.append((source, target, relation))
            # This incidence list is intentionally undirected for neighborhood lookup.
            adjacency[source].append((edge_id, target))
            adjacency[target].append((edge_id, source))
    return edges, adjacency


def percentile(sorted_values, q):
    if not sorted_values:
        return 0
    index = (len(sorted_values) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[lower]
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def describe(values):
    values = sorted(values)
    if not values:
        return {}
    return {
        "count": len(values),
        "min": values[0],
        "mean": sum(values) / len(values),
        "median": percentile(values, 0.5),
        "p90": percentile(values, 0.9),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": values[-1],
    }


def pair_candidate_nodes(drug_a, drug_b, adjacency):
    neighbors_a = {other for _, other in adjacency.get(drug_a, ()) if other != drug_b}
    neighbors_b = {other for _, other in adjacency.get(drug_b, ()) if other != drug_a}
    nodes = {drug_a, drug_b} | neighbors_a | neighbors_b
    return neighbors_a, neighbors_b, nodes


def induced_edge_ids(nodes, adjacency):
    """Find induced edges in O(sum of selected-node degrees), not O(|A|*|B|)."""
    edge_ids = set()
    for node in nodes:
        for edge_id, other in adjacency.get(node, ()):
            if other in nodes:
                edge_ids.add(edge_id)
    return edge_ids


def capped_distance(root, local_adjacency, max_distance=3):
    distances = {root: 0}
    queue = deque([root])
    while queue:
        node = queue.popleft()
        if distances[node] >= max_distance:
            continue
        for neighbor in local_adjacency.get(node, ()):
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    return distances


def distance_id(distance):
    if distance is None:
        return DISTANCE_TO_ID["unreachable"]
    if distance >= 3:
        return DISTANCE_TO_ID["3+"]
    return DISTANCE_TO_ID[str(distance)]


def find_bridge_edge_ids(neighbors_a, neighbors_b, adjacency):
    side_a = neighbors_a - neighbors_b
    side_b = neighbors_b - neighbors_a
    bridge_edge_ids = set()
    # Scan the side with fewer incident edges and test membership on the other side.
    scan_side, target_side = (side_a, side_b)
    if sum(len(adjacency.get(node, ())) for node in side_b) < sum(
        len(adjacency.get(node, ())) for node in side_a
    ):
        scan_side, target_side = side_b, side_a
    for node in scan_side:
        for edge_id, other in adjacency.get(node, ()):
            if other in target_side:
                bridge_edge_ids.add(edge_id)
    return bridge_edge_ids


def select_nodes(
    drug_a, drug_b, neighbors_a, neighbors_b, nodes, edges, bridge_edge_ids, max_nodes
):
    if max_nodes is None or len(nodes) <= max_nodes:
        return nodes
    common = neighbors_a & neighbors_b
    selected = [drug_a, drug_b]
    selected.extend(sorted(common, key=str)[: max_nodes - len(selected)])

    # Retain complete bridges: both endpoints must fit in the remaining budget.
    ordered_bridges = sorted(
        bridge_edge_ids,
        key=lambda edge_id: (
            str(edges[edge_id][0]), edges[edge_id][2], str(edges[edge_id][1])
        ),
    )
    selected_set = set(selected)
    for edge_id in ordered_bridges:
        source, target, _ = edges[edge_id]
        missing = [node for node in (source, target) if node not in selected_set]
        if len(selected) + len(missing) > max_nodes:
            continue
        selected.extend(missing)
        selected_set.update(missing)

    # Fill the remaining budget evenly from the two drug-specific neighborhoods.
    # This is deterministic and deliberately avoids a hand-crafted candidate score.
    side_a = iter(sorted(neighbors_a - selected_set, key=str))
    side_b = iter(sorted(neighbors_b - selected_set, key=str))
    exhausted_a = exhausted_b = False
    while len(selected) < max_nodes and not (exhausted_a and exhausted_b):
        if not exhausted_a and len(selected) < max_nodes:
            try:
                selected.append(next(side_a))
            except StopIteration:
                exhausted_a = True
        if not exhausted_b and len(selected) < max_nodes:
            try:
                selected.append(next(side_b))
            except StopIteration:
                exhausted_b = True
    return set(selected)


def build_pair_graph(
    drug_a,
    drug_b,
    edges,
    adjacency,
    node_types,
    vocabs,
    max_nodes=None,
    max_edges=None,
):
    neighbors_a, neighbors_b, nodes = pair_candidate_nodes(drug_a, drug_b, adjacency)
    bridge_edge_ids = find_bridge_edge_ids(neighbors_a, neighbors_b, adjacency)
    nodes = select_nodes(
        drug_a,
        drug_b,
        neighbors_a,
        neighbors_b,
        nodes,
        edges,
        bridge_edge_ids,
        max_nodes,
    )
    edge_ids = induced_edge_ids(nodes, adjacency)

    ordered_nodes = [drug_a, drug_b] + sorted(nodes - {drug_a, drug_b}, key=str)
    local_id = {node: index for index, node in enumerate(ordered_nodes)}
    local_adjacency = defaultdict(set)
    kept_edges = []
    prioritized_edge_ids = sorted(
        edge_ids,
        key=lambda edge_id: (
            not (
                edges[edge_id][0] in {drug_a, drug_b}
                or edges[edge_id][1] in {drug_a, drug_b}
            ),
            edge_id not in bridge_edge_ids,
            edge_id,
        ),
    )
    for edge_id in prioritized_edge_ids:
        source, target, relation = edges[edge_id]
        if source not in local_id or target not in local_id:
            continue
        kept_edges.append((source, target, relation))
        local_adjacency[source].add(target)
        local_adjacency[target].add(source)
        if max_edges is not None and len(kept_edges) >= max_edges:
            break

    distance_a = capped_distance(drug_a, local_adjacency)
    distance_b = capped_distance(drug_b, local_adjacency)
    node_ids = [add_vocab(vocabs["entity"], node) for node in ordered_nodes]
    node_type_ids = [
        add_vocab(vocabs["type"], node_types.get(node, "Unknown"))
        for node in ordered_nodes
    ]

    edge_sources = []
    edge_targets = []
    edge_relations = []
    for source, target, relation in kept_edges:
        forward_relation = add_vocab(vocabs["relation"], relation)
        inverse_relation = add_vocab(vocabs["relation"], f"inverse::{relation}")
        edge_sources.extend((local_id[source], local_id[target]))
        edge_targets.extend((local_id[target], local_id[source]))
        edge_relations.extend((forward_relation, inverse_relation))

    return {
        "node_ids": node_ids,
        "node_types": node_type_ids,
        "distance_to_a": [distance_id(distance_a.get(node)) for node in ordered_nodes],
        "distance_to_b": [distance_id(distance_b.get(node)) for node in ordered_nodes],
        "edge_index": [edge_sources, edge_targets],
        "edge_relations": edge_relations,
    }, {
        "neighbors_a": len(neighbors_a),
        "neighbors_b": len(neighbors_b),
        "common": len(neighbors_a & neighbors_b),
        "bridge_edges": len(bridge_edge_ids),
        "nodes_before_cap": len({drug_a, drug_b} | neighbors_a | neighbors_b),
        "nodes": len(ordered_nodes),
        "directed_edges_before_inverse": len(kept_edges),
        "message_edges": len(edge_relations),
    }


def main():
    parser = argparse.ArgumentParser(description="Build explicit pair-level MolecBioNet graphs.")
    parser.add_argument("--zeroddi-csv", required=True)
    parser.add_argument("--split-files", nargs="+", required=True)
    parser.add_argument("--molecbionet-dir", required=True)
    parser.add_argument("--output")
    parser.add_argument("--stats-output")
    parser.add_argument("--limit-pairs", type=int)
    parser.add_argument("--max-nodes", type=int)
    parser.add_argument("--max-edges", type=int, help="Original directed edges before inverse edges.")
    parser.add_argument("--include-ddi-edges", action="store_true")
    parser.add_argument("--stats-only", action="store_true")
    args = parser.parse_args()

    start = time.time()
    source_dir = Path(args.molecbionet_dir)
    zeroddi_drugs = read_zeroddi_drugs(args.zeroddi_csv)
    split_pairs = read_split_pairs(args.split_files)
    if args.limit_pairs is not None:
        split_pairs = split_pairs[: args.limit_pairs]
    molecbionet_drugs = read_molecbionet_drugs(source_dir / "DrugBank_Node_Information.txt")
    node_types = read_node_types(source_dir / "Node_Type.txt")
    drug_mapping, mapping_methods = map_drugs_by_smiles(zeroddi_drugs, molecbionet_drugs)
    exclude_relations = set() if args.include_ddi_edges else {"DDI"}
    edges, adjacency = read_directed_kg(source_dir / "BioKG_DrugBank.txt", exclude_relations)

    relevant_nodes = set(drug_mapping.values())
    for drug_node in tuple(relevant_nodes):
        relevant_nodes.update(other for _, other in adjacency.get(drug_node, ()))
    relation_names = {relation for _, _, relation in edges}
    type_names = {node_types.get(node, "Unknown") for node in relevant_nodes}
    vocabs = {
        "entity": {node: index + 1 for index, node in enumerate(sorted(relevant_nodes, key=str))},
        "type": {name: index + 1 for index, name in enumerate(sorted(type_names))},
        "relation": {
            name: index + 1
            for index, name in enumerate(
                sorted(relation_names | {f"inverse::{name}" for name in relation_names})
            )
        },
    }
    pair_graphs = {}
    sqlite_connection = None
    sqlite_path = None
    if not args.stats_only and args.output and args.output.endswith(".sqlite"):
        sqlite_path = Path(args.output)
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = sqlite_path.with_suffix(sqlite_path.suffix + ".tmp")
        if temporary_path.exists():
            temporary_path.unlink()
        sqlite_connection = sqlite3.connect(str(temporary_path))
        sqlite_connection.execute("PRAGMA journal_mode=OFF")
        sqlite_connection.execute("PRAGMA synchronous=OFF")
        sqlite_connection.execute(
            "CREATE TABLE pairs (pair_key TEXT PRIMARY KEY, graph BLOB NOT NULL)"
        )
        sqlite_connection.execute(
            "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
    stats = defaultdict(list)
    mapped_pairs = 0
    build_start = time.time()
    for index, (drug1, drug2) in enumerate(split_pairs, 1):
        drug_a = drug_mapping.get(drug1)
        drug_b = drug_mapping.get(drug2)
        if drug_a is None or drug_b is None:
            continue
        mapped_pairs += 1
        graph, graph_stats = build_pair_graph(
            drug_a,
            drug_b,
            edges,
            adjacency,
            node_types,
            vocabs,
            max_nodes=args.max_nodes,
            max_edges=args.max_edges,
        )
        for name, value in graph_stats.items():
            stats[name].append(value)
        if not args.stats_only:
            key = f"{drug1}||{drug2}"
            if sqlite_connection is None:
                pair_graphs[key] = graph
            else:
                payload = zlib.compress(
                    json.dumps(graph, separators=(",", ":")).encode("utf-8"), level=3
                )
                sqlite_connection.execute(
                    "INSERT OR REPLACE INTO pairs(pair_key, graph) VALUES (?, ?)",
                    (key, payload),
                )
                if index % 1000 == 0:
                    sqlite_connection.commit()
        if index % 10000 == 0:
            elapsed = time.time() - build_start
            print(f"Processed {index}/{len(split_pairs)} pairs in {elapsed:.1f}s", flush=True)

    summary = {
        "zeroddi_drugs": len(zeroddi_drugs),
        "mapped_drugs": len(drug_mapping),
        "mapping_methods": dict(Counter(mapping_methods.values())),
        "kg_edges_excluding_ddi": len(edges),
        "requested_pairs": len(split_pairs),
        "mapped_pairs": mapped_pairs,
        "max_nodes": args.max_nodes,
        "max_edges_before_inverse": args.max_edges,
        "distributions": {name: describe(values) for name, values in stats.items()},
        "elapsed_seconds": time.time() - start,
        "pair_build_seconds": time.time() - build_start,
    }
    print(json.dumps(summary, indent=2), flush=True)

    if args.stats_output:
        stats_path = Path(args.stats_output)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    if not args.stats_only:
        if not args.output:
            parser.error("--output is required unless --stats-only is used")
        output = {
            "format": "molecbionet_pair_graph_v2",
            "metadata": summary,
            "distance_vocab_size": max(DISTANCE_TO_ID.values()) + 1,
            "feature_vocab_sizes": {
                name: len(vocab) + 1 for name, vocab in vocabs.items()
            },
            "feature_vocabs": vocabs,
            "drug_mapping": drug_mapping,
            "pairs": pair_graphs,
        }
        output_path = Path(args.output)
        if sqlite_connection is not None:
            sqlite_metadata = dict(output)
            sqlite_metadata.pop("pairs")
            for key, value in sqlite_metadata.items():
                sqlite_connection.execute(
                    "INSERT INTO metadata(key, value) VALUES (?, ?)",
                    (key, json.dumps(value, separators=(",", ":"))),
                )
            sqlite_connection.commit()
            sqlite_connection.execute("CREATE INDEX pair_key_index ON pairs(pair_key)")
            sqlite_connection.close()
            temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
            temporary_path.replace(output_path)
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, separators=(",", ":"))
        temporary_path.replace(output_path)


if __name__ == "__main__":
    main()
