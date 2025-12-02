def format_if_elif_else(ruleset, feature_names, operator_set):
    """
    Format a ruleset into human-readable IF/ELIF/ELSE form.

    :param ruleset: Ruleset to format (The best individual (list of rules) from GA.
    :param feature_names: Feature names to use in conditions.
    :param operator_set: Set of available operators for converting op_idx into a symbol.
    :return: Formatted ruleset as a string.
    """
    # Order rules by number of conditions (longer first)
    ordered = sorted(ruleset, key=lambda r: len(r) - 1, reverse=True)

    lines = []
    for idx, r in enumerate(ordered):
        conds = " AND ".join(
            f"{feature_names[f]} {operator_set.get(op_idx).symbol} {thr:.2f}"
            for f, thr, op_idx in r[:-1]
        )
        prefix = "if" if idx == 0 else ("elif" if idx < len(ordered) - 1 else "else")
        if conds:
            lines.append(f"{prefix} {conds}: class = {r[-1][1]}")
        else:
            # Rule without conditions
            lines.append(f"{prefix}: class = {r[-1][1]}")

    return "\n".join(lines)
