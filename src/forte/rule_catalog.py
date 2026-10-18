"""Validate and publish rule catalogs without inventing normative provenance."""

from collections.abc import Mapping


def catalog_rules(catalog: dict, selected: list[str]) -> list[dict]:
    """Support legacy source-backed rules and explicit user-defined criteria."""
    names = [name for name in ("detection_rules", "verified_detection_rules") if name in catalog]
    if len(names) != 1 or not isinstance(catalog[names[0]], list):
        raise ValueError("Exactly one rule catalog list is required")
    if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected):
        raise ValueError("Unique selected rule identifiers are required")
    by_id = {}
    for rule in catalog[names[0]]:
        if not isinstance(rule, Mapping):
            raise ValueError("Invalid detection rule")
        rule_id = rule.get("rule_id")
        if not isinstance(rule_id, str) or not rule_id.strip() or len(rule_id) > 120:
            raise ValueError("Invalid rule identifier")
        if rule_id in by_id:
            raise ValueError("Duplicate rule identifier")
        by_id[rule_id] = rule
    result = []
    for rule_id in selected:
        rule = by_id.get(rule_id)
        if rule is None or rule.get("enabled") is not True:
            raise ValueError("Selected rule is missing or disabled")
        for field in ("title", "source"):
            if not isinstance(rule.get(field), str) or not rule[field].strip():
                raise ValueError("Every rule requires a title and an honest source")
        item = {key: rule[key] for key in ("rule_id", "title", "source")}
        item["clause"] = rule.get("clause", "")
        if not isinstance(item["clause"], str):
            raise ValueError("Rule clause must be text")
        if "source_kind" in rule:
            if rule["source_kind"] not in {"user_manual_annotation", "normative_source"}:
                raise ValueError("Unsupported source kind")
            item["source_kind"] = rule["source_kind"]
        result.append(item)
    return result


def manual_prompt(catalog: dict) -> str:
    """Render general criteria only; no video IDs, labels, comments or event times."""
    rules = catalog["detection_rules"]
    selected = [rule["rule_id"] for rule in rules if rule.get("enabled") is True]
    catalog_rules(catalog, selected)
    lines = [
        "Review chronological timestamped frames from one cashier video. Detect ONLY the "
        "enabled user-defined observable criteria below. These are annotation criteria for "
        "human review, not verified legal rules. Never invent law references or hidden facts.",
        "ENABLED CRITERIA:",
    ]
    for rule in rules:
        if rule["rule_id"] in selected:
            lines.append(
                f"{rule['rule_id']}: {rule['observable_predicate']} EXCLUDE: {rule['exclude']}"
            )
    lines.extend(
        [
            "END CRITERIA.",
            "Inspect all supplied frames. Report every distinct supported type, including two "
            "types in one scene when both are supported. Cash-event precedence is: departure -> "
            "cash_left_unattended; staying with an unrelated activity -> "
            "other_activity_with_unsecured_cash; otherwise a finished operation with cash still "
            "exposed -> cash_not_secured_after_operation. Do not duplicate one cash sequence "
            "under these three overlapping types.",
            "A missing action between sampled frames is not proof it never happened. For an "
            "omission criterion require visible start, end and sufficient intervening context. "
            "If an apparently relevant sequence cannot be resolved, use uncertain. Ordinary "
            "cash handling or an unrelated normal scene with no supported criterion is "
            "no_violation. Do not infer amounts, document contents, authenticity, digital checks, "
            "password/lock status, intent or guilt. Describe observable facts briefly in Russian.",
            'Return only JSON: {"verdict":"violation" or "no_violation" or "uncertain",'
            '"analysis_complete":boolean,"events":[{"time_seconds":number,'
            '"rule_id":"one enabled ID","description":"visible evidence in Russian"}],'
            '"uncertainty_reason":"short reason or empty string"}.',
            "A complete window has analysis_complete=true and uncertainty_reason=empty string. "
            "violation requires at least one supported event; no_violation requires events=[]. "
            "For unresolved relevant evidence or inability to complete: uncertain, "
            "analysis_complete=false, events=[], and a specific uncertainty_reason. "
            "Event times MUST be among the timestamps supplied with the images.",
            "Each request says LOCAL WINDOW or OVERVIEW. Local windows contain up to 16 frames "
            "at the planned 0.5-second interval, with 6-frame overlap. Overview frames are "
            "uniformly selected across the entire video and may have large gaps. Use the actual "
            "timestamps. You have no memory of other windows. Never infer omitted actions or "
            "continuous object visibility from gaps. The runner aggregates windows separately.",
        ]
    )
    return "\n\n".join(lines) + "\n"
