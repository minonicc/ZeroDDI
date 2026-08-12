"""Guards that keep dependency-ordered experiments from launching early."""


def enforce_provisional_run_limit(cfg, max_debug_epochs=5):
    dependency = cfg.get("provisional_dependency", None)
    if dependency and cfg.num_epochs > max_debug_epochs:
        raise RuntimeError(
            f"Config is provisional on {dependency!r} and may run at most "
            f"{max_debug_epochs} debug epochs; requested {cfg.num_epochs}. "
            "Lock and audit the upstream validation winner first."
        )
