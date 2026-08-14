"""Assert controlled differences among pharmacophore experiment configs."""

from collections.abc import Mapping
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.config import Config  # noqa: E402


CONFIG_DIR = ROOT / "configs"


def load(name):
    return Config.fromfile(str(CONFIG_DIR / name))


def flatten(value, prefix=""):
    flattened = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            flattened.update(flatten(child, child_prefix))
    else:
        flattened[prefix] = value
    return flattened


def differing_fields(left, right):
    left = flatten(left._cfg_dict)
    right = flatten(right._cfg_dict)
    return {
        key
        for key in set(left) | set(right)
        if left.get(key) != right.get(key)
    }


def require_differences(left, right, expected, label):
    actual = differing_fields(left, right)
    if actual != expected:
        raise AssertionError(
            f"{label} differs in unexpected fields: "
            f"actual={sorted(actual)}, expected={sorted(expected)}"
        )


def main():
    p1 = load("new_s0_reverse_kg_pharmacophore_p1_128_mean.py")
    p2 = load("new_s0_reverse_kg_pharmacophore_p2_512_sum.py")
    p3 = load("new_s0_reverse_kg_pharmacophore_p3_128_sum_gate.py")
    p6 = load("new_s0_reverse_kg_pharmacophore_p6_128_mean_gate.py")
    p7 = load("new_s0_reverse_kg_pharmacophore_p7_512_sum_gate.py")
    p8 = load("new_s0_reverse_kg_pharmacophore_p8_256_sum.py")
    struct_s1 = load("new_s0_struct_s1_pharmacophore_replaces_substructure.py")
    struct_s3 = load("new_s0_struct_s3_fixed_substructure_pharmacophore.py")
    struct_s4 = load("new_s0_struct_s4_fixed_substructure.py")
    top64 = load("new_s0_pharmacophore_t2_top64.py")
    top128 = load("new_s0_pharmacophore_t3_top128.py")
    top256 = load("new_s0_pharmacophore_t2_top256.py")
    drug12 = load("new_s0_pharmacophore_d3_drug_top12.py")
    drug16 = load("new_s0_pharmacophore_d3_drug_top16.py")

    require_differences(
        p1,
        p2,
        {
            "model.leftmodel.pharmacophore_max_pairs",
            "model.leftmodel.pharmacophore_pooling",
            "work_dir",
        },
        "P1 versus P2",
    )
    require_differences(
        p2,
        p3,
        {
            "model.leftmodel.pharmacophore_max_pairs",
            "model.matching_use_pharmacophore_gate",
            "work_dir",
        },
        "P2 versus P3",
    )
    require_differences(
        p1,
        p6,
        {"model.matching_use_pharmacophore_gate", "work_dir"},
        "P1 versus P6",
    )
    require_differences(
        p2,
        p7,
        {"model.matching_use_pharmacophore_gate", "work_dir"},
        "P2 versus P7",
    )
    require_differences(
        p2,
        p8,
        {"model.leftmodel.pharmacophore_max_pairs", "work_dir"},
        "P2 versus P8",
    )
    assert p8.model.leftmodel.pharmacophore_max_pairs == 256
    assert p8.model.leftmodel.pharmacophore_pooling == "sum"
    assert not p8.model.matching_use_pharmacophore_gate
    require_differences(
        struct_s1,
        struct_s3,
        {
            "model.leftmodel.fixed_substructure_alpha_init",
            "model.leftmodel.use_fixed_substructure_base",
            "model.leftmodel.use_sub",
            "work_dir",
        },
        "Struct-S1 versus Struct-S3",
    )
    require_differences(
        struct_s3,
        struct_s4,
        {
            "model.leftmodel.use_pharmacophore_pairs",
            "model.matching_use_pharmacophore_evidence",
            "work_dir",
        },
        "Struct-S3 versus Struct-S4",
    )
    require_differences(
        top64,
        top128,
        {"model.matching_pharmacophore_top_k", "work_dir"},
        "pair Top-64 versus Top-128",
    )
    require_differences(
        top128,
        top256,
        {"model.matching_pharmacophore_top_k", "work_dir"},
        "pair Top-128 versus Top-256",
    )
    require_differences(
        drug12,
        drug16,
        {"model.matching_pharmacophore_drug_top_k", "work_dir"},
        "per-drug Top-12 versus Top-16",
    )
    require_differences(
        top128,
        drug12,
        {
            "model.leftmodel.pharmacophore_selection_mode",
            "model.matching_pharmacophore_drug_top_k",
            "model.matching_pharmacophore_top_k",
            "work_dir",
        },
        "pair Top-128 versus per-drug Top-12",
    )

    assert struct_s1.model.matching_use_kg_evidence
    assert not struct_s1.model.matching_use_substructure_evidence
    assert struct_s3.model.leftmodel.fixed_substructure_alpha_init == 0.0
    assert not struct_s3.model.leftmodel.use_query_substructure
    assert not struct_s4.model.matching_use_pharmacophore_evidence
    for config in (struct_s1, struct_s3):
        assert config.get("provisional_dependency", None) is None
        assert config.model.leftmodel.pharmacophore_max_pairs == 128
        assert config.model.leftmodel.pharmacophore_pooling == "sum"
        assert not config.model.matching_use_pharmacophore_gate
    assert struct_s4.get("provisional_dependency", None) is None
    for config in (top64, top128, top256, drug12, drug16):
        assert config.provisional_dependency == "stage2_validation_winner"
        assert config.model.leftmodel.pharmacophore_max_pairs is None
        assert config.model.matching_use_pharmacophore_evidence
    assert top64.model.matching_pharmacophore_top_k == 64
    assert top128.model.matching_pharmacophore_top_k == 128
    assert top256.model.matching_pharmacophore_top_k == 256
    assert drug12.model.matching_pharmacophore_drug_top_k == 12
    assert drug16.model.matching_pharmacophore_drug_top_k == 16
    print("pharmacophore controlled-config audit ok")


if __name__ == "__main__":
    main()
