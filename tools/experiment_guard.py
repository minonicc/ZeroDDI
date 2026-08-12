"""Guards that keep dependency-ordered experiments from launching early."""


def enforce_provisional_run_limit(
    cfg, max_debug_epochs=5, evaluation_requested=False
):
    dependency = cfg.get("provisional_dependency", None)
    if dependency and evaluation_requested:
        raise RuntimeError(
            f"Config is provisional on {dependency!r}; checkpoint evaluation "
            "is disabled until the upstream validation winner is locked and "
            "the dependency marker is removed."
        )
    if dependency and cfg.num_epochs > max_debug_epochs:
        raise RuntimeError(
            f"Config is provisional on {dependency!r} and may run at most "
            f"{max_debug_epochs} debug epochs; requested {cfg.num_epochs}. "
            "Lock and audit the upstream validation winner first."
        )
