"""
CreditPulse - Causal DAG
Phase 3: Defines the causal graph of credit default drivers.
"""

import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

# ── Output folder ──────────────────────────────────────────────────────────
REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)


def build_causal_dag():
    """
    Build the directed acyclic graph (DAG) encoding our causal assumptions.
    Each arrow means: left node CAUSES right node.
    """
    G = nx.DiGraph()

    # ── Nodes (variables) ─────────────────────────────────────────────────
    G.add_nodes_from([
        "Macro Shock\n(COVID/Drought)",
        "Income\nVolatility",
        "M-Pesa\nBalance",
        "Loan\nAmount",
        "Credit Bureau\nScore",
        "Days Past\nDue",
        "DEFAULT",
    ])

    # ── Edges (causal relationships) ──────────────────────────────────────
    G.add_edges_from([
        ("Macro Shock\n(COVID/Drought)", "Income\nVolatility"),
        ("Macro Shock\n(COVID/Drought)", "M-Pesa\nBalance"),
        ("Income\nVolatility",           "Days Past\nDue"),
        ("Income\nVolatility",           "M-Pesa\nBalance"),
        ("M-Pesa\nBalance",              "Days Past\nDue"),
        ("Loan\nAmount",                 "Days Past\nDue"),
        ("Credit Bureau\nScore",         "DEFAULT"),
        ("Days Past\nDue",               "DEFAULT"),
        ("Loan\nAmount",                 "DEFAULT"),
    ])

    return G


def visualise_dag(G):
    """Draw and save the causal DAG as a PNG."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Layout
    pos = {
        "Macro Shock\n(COVID/Drought)": (0, 2),
        "Income\nVolatility":           (2, 3),
        "M-Pesa\nBalance":              (2, 1),
        "Loan\nAmount":                 (4, 3),
        "Credit Bureau\nScore":         (4, 1),
        "Days Past\nDue":               (6, 2),
        "DEFAULT":                      (8, 2),
    }

    # Node colours — red for outcome, green for treatment, blue for others
    colours = {
        "DEFAULT":                      "#E74C3C",
        "Macro Shock\n(COVID/Drought)": "#2ECC71",
        "Days Past\nDue":               "#E67E22",
    }
    node_colours = [colours.get(n, "#3498DB") for n in G.nodes()]

    nx.draw_networkx(
        G, pos=pos, ax=ax,
        node_color=node_colours,
        node_size=3000,
        font_size=8,
        font_color="white",
        font_weight="bold",
        edge_color="#2C3E50",
        arrows=True,
        arrowsize=20,
        width=2,
    )

    ax.set_title("CreditPulse — Causal DAG of Credit Default Drivers",
                 fontsize=14, fontweight="bold", pad=20)
    ax.axis("off")

    output = REPORTS / "causal_dag.png"
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  DAG saved to {output}")


def main():
    print("Building causal DAG...")
    G = build_causal_dag()
    print(f"  Nodes: {list(G.nodes())}")
    print(f"  Edges: {G.number_of_edges()} causal relationships defined")
    visualise_dag(G)
    print("Causal DAG complete!")


if __name__ == "__main__":
    main()