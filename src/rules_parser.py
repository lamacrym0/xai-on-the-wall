import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Optional: Graphviz visualization (install: pip install graphviz + system Graphviz)
try:
    from graphviz import Digraph
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False


# ─────────────────────────────────────────────
# 1. Data structure for a rule tree
# ─────────────────────────────────────────────

@dataclass
class RuleNode:
    condition: Optional[str] = None      # e.g. "worst concave points <= 0.218"
    children: Dict[str, "RuleNode"] = field(default_factory=dict)
    label: Optional[str] = None          # class at leaf, e.g. "benign"


def parse_rule(rule_str: str):
    """
    Parse one rule of the form:
      IF ((cond1) AND (cond2) AND ...) THEN class

    Returns: (list_of_conditions, class_label)
    """
    # Clean spaces at edges
    rule_str = rule_str.strip()

    # Split at THEN
    if "THEN" not in rule_str:
        raise ValueError(f"Rule has no THEN: {rule_str}")

    lhs, rhs = rule_str.split("THEN", 1)
    rhs = rhs.strip()

    # Extract class label (e.g. "benign")
    class_label = rhs

    # Extract conditions from LHS
    # Example LHS: "IF ((worst concave points <= 0.218) AND (mean concavity <= 0.062) AND ...)"
    # We grab anything like "(feature op value)".
    cond_pattern = r"\(([^()]+[<>]=?[^()]+)\)"
    conds = re.findall(cond_pattern, lhs)

    # Clean each condition: strip and normalize spaces
    conds = [c.strip() for c in conds]

    return conds, class_label


def build_rule_tree(rules: List[str]) -> RuleNode:
    """
    Build a prefix tree from a list of rule strings.
    Each path from root to leaf corresponds to one rule.
    """
    root = RuleNode(condition=None)

    for rule_str in rules:
        conds, label = parse_rule(rule_str)
        node = root
        for cond in conds:
            if cond not in node.children:
                node.children[cond] = RuleNode(condition=cond)
            node = node.children[cond]
        # At the end of the path, set the class label
        node.label = label

    return root


# ─────────────────────────────────────────────
# 2. ASCII visualization
# ─────────────────────────────────────────────

def print_rule_tree(node: RuleNode, indent: str = ""):
    """
    Print the tree in a simple text form.
    """
    # Root node has condition=None
    for cond, child in node.children.items():
        print(f"{indent}- {cond}")
        if child.label is not None and not child.children:
            print(f"{indent}    => {child.label}")
        print_rule_tree(child, indent + "    ")


# ─────────────────────────────────────────────
# 3. Graphviz visualization (optional)
# ─────────────────────────────────────────────

def export_tree_to_graphviz(root: RuleNode, filename: str = "rules_tree", view: bool = False):
    """
    Export the rule tree to a Graphviz .pdf/.png (requires graphviz installed).
    """
    if not HAS_GRAPHVIZ:
        raise ImportError("graphviz is not installed. Please pip install graphviz and install system Graphviz.")

    dot = Digraph(name="RuleTree", format="pdf")
    dot.attr("node", shape="box", style="rounded,filled", color="lightgrey", fontsize="10")

    # Assign integer IDs to nodes so we can reference them in edges
    node_counter = [0]  # mutable integer, wrapped in list

    def add_node_recursive(node: RuleNode, parent_id: Optional[str] = None):
        node_id = f"n{node_counter[0]}"
        node_counter[0] += 1

        if node.condition is None:
            # root node
            label = "ROOT"
        else:
            label = node.condition

        # If leaf with label, append "→ class"
        if node.label is not None and not node.children:
            label = f"{label}\n=> {node.label}"

        dot.node(node_id, label=label)

        # Edge from parent to here
        if parent_id is not None:
            dot.edge(parent_id, node_id)

        # Recurse on children
        for child in node.children.values():
            add_node_recursive(child, node_id)

    add_node_recursive(root, None)

    # Render to file
    dot.render(filename, view=view)
    print(f"Graphviz tree saved to {filename}.pdf")


# ─────────────────────────────────────────────
# 4. Example usage with your rules
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Your example rules as a list of strings
    rules = [
        "IF ((worst concave points <= 0.218) AND (mean concavity <= 0.062) AND (worst radius > 0.395)) THEN benign",
        "IF ((worst concave points > 0.033) AND (worst concavity <= -0.26)) THEN malignant",
        "IF ((worst concave points <= 0.033) AND (worst area <= 0.246) AND (concavity error > 0.049) AND (worst perimeter > -0.258)) THEN benign",
        "IF ((worst concave points > 0.033) AND (worst concavity > -0.26) AND (worst texture > -1.469) AND (concavity error > -0.468)) THEN benign",
        # ... put all your other rules here as separate strings ...
        "IF ((worst concave points <= 0.218) AND (mean concavity <= 0.062) AND (worst radius <= 0.395) AND (worst concavity > -0.143)) THEN malignant",
    ]

    # Build tree
    root = build_rule_tree(rules)

    # 1) ASCII visualization
    print("=== ASCII tree ===")
    print_rule_tree(root)

    # 2) Graphviz visualization (optional)
    if HAS_GRAPHVIZ:
        export_tree_to_graphviz(root, filename="rules_tree", view=False)
        # set view=True if you want it to open automatically
