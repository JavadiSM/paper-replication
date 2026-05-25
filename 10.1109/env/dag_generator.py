import random
import networkx as nx
import numpy as np


class DAGGenerator:
    def __init__(
        self,
        num_tasks: int = 10,
        num_nodes: int = 10,
        max_out_degree: int = 4,
        alpha: float = 1.0,
        beta: float = 1.0,
        cpu_range=(1e7, 1e8),
        data_range=(50, 500),      # KB
        edge_data_range=(100, 500) # KB

    ):
        self.num_tasks = num_tasks
        
        self.num_nodes = num_nodes
        self.max_out_degree = max_out_degree
        self.alpha = alpha
        self.beta = beta

        self.cpu_range = cpu_range
        self.data_range = data_range
        self.edge_data_range = edge_data_range

    def generate(self):
        """
        Generate a valid DAG with:
        - Start node
        - End node
        - Node attributes
        - Edge attributes
        """

        g = nx.DiGraph()

        # --------------------------------------------------
        # executable nodes only
        # --------------------------------------------------

        executable_nodes = list(range(1, self.num_nodes + 1))

        for task in range(self.num_tasks):
            for node in executable_nodes:

                g.add_node(
                    (task, node),
                    cpu_cycles=random.uniform(*self.cpu_range),
                    data_size=random.uniform(*self.data_range),
                    scheduled_location=-1,
                    available=0
                )

            # --------------------------------------------------
            # DAG edges
            # only forward edges to preserve acyclic property
            # --------------------------------------------------

            for src in executable_nodes:

                possible_targets = [
                    t for t in executable_nodes
                    if t > src
                ]

                if len(possible_targets) == 0:
                    continue

                out_degree = random.randint(
                    0,
                    min(self.max_out_degree, len(possible_targets))
                )

                targets = random.sample(
                    possible_targets,
                    out_degree
                )

                for dst in targets:

                    g.add_edge(
                        (task, src),
                        (task, dst),
                        data_size=random.uniform(*self.edge_data_range)
                    )

        # --------------------------------------------------
        # add Start and End nodes
        # --------------------------------------------------

        start_node = 0
        end_node = self.num_nodes + 1

        g.add_node(
            start_node,
            cpu_cycles=0,
            data_size=0,
            scheduled_location=-1,
            available=1
        )

        for task in range(self.num_tasks):
            g.add_node(
                (task, end_node),
                cpu_cycles=0,
                data_size=0,
                scheduled_location=-1,
                available=0
            )

        for task in range(self.num_tasks):
            # connect Start to source nodes
            source_nodes = [
                (task, n) for n in executable_nodes
                if g.in_degree((task,n)) == 0
            ]

            for node in source_nodes:
                g.add_edge(start_node, node, data_size=0)

        for task in range(self.num_tasks):
            # connect sink nodes to End
            sink_nodes = [
                (task, n) for n in executable_nodes
                if g.out_degree((task, n)) == 0
            ]
            
            for node in sink_nodes:# type: ignore
                g.add_edge(node, (task, end_node), data_size=0)

        # --------------------------------------------------
        # update static node features
        # --------------------------------------------------

        for node in g.nodes:

            g.nodes[node]["in_degree"] = g.in_degree(node)
            g.nodes[node]["out_degree"] = g.out_degree(node)

        # --------------------------------------------------
        # validate DAG
        # --------------------------------------------------

        assert nx.is_directed_acyclic_graph(g)

        return g

import matplotlib.pyplot as plt
import networkx as nx

def visualize_dag(
    g,
    num_tasks,
    num_nodes,
    show_edge_labels: bool = False,
    node_size: int = 900,
    figsize=(14, 6),
):
    """
    Visualize a multi-task DAG where nodes are:
      - 0: global Start
      - num_nodes + 1: global End
      - (task, n): internal task nodes

    Args:
        g: networkx.DiGraph
        num_tasks: int
        num_nodes: int (per task)
        show_edge_labels: bool, whether to show data_size on edges
        node_size: int, matplotlib node size
        figsize: tuple, figure size
    """

    start_node = 0
    end_node = num_nodes + 1

    # --------------------------------------------------
    # Compute positions
    # --------------------------------------------------
    pos = {}

    # فاصله‌ی عمودی بین taskها
    lane_gap = 1.5
    # مرکز عمودی را طوری تنظیم می‌کنیم که همه‌ی taskها وسط تصویر باشند
    y_center = (num_tasks - 1) * lane_gap / 2

    # Start / End در وسط عمودی
    pos[start_node] = (-1.5, y_center)
    pos[end_node] = (num_nodes + 2, y_center)

    # نودهای داخلی: (task, n)
    for node in g.nodes:
        if isinstance(node, tuple):
            task, n = node
            x = n                       # ستون = اندیس نود
            y = task * lane_gap        # هر task در lane جدا
            pos[node] = (x, y)

    # --------------------------------------------------
    # Draw figure
    # --------------------------------------------------
    plt.figure(figsize=figsize)

    # رنگ‌بندی نودها بر اساس نوع‌شان (Start / End / task)
    node_colors = []
    for node in g.nodes:
        if node == start_node:
            node_colors.append("#66c2a5")   # سبز ملایم
        elif node == end_node:
            node_colors.append("#fc8d62")   # نارنجی ملایم
        else:
            node_colors.append("#8da0cb")   # آبی ملایم

    # گره‌ها
    nx.draw_networkx_nodes(
        g,
        pos,
        node_color=node_colors,
        node_size=node_size,
        edgecolors="black",
        linewidths=0.8,
    )

    # یال‌ها (با کمی خمیدگی برای دیده شدن یال‌های موازی)
    nx.draw_networkx_edges(
        g,
        pos,
        arrows=True,
        arrowstyle="->",
        arrowsize=18,
        width=1.4,
        connectionstyle="arc3,rad=0.12",  # rad بزرگ‌تر برای جداسازی بهتر
    )

    # برچسب نودها
    labels = {}
    for node in g.nodes:
        if node == start_node:
            labels[node] = "Start"
        elif node == end_node:
            labels[node] = "End"
        else:
            task, n = node
            labels[node] = f"{task},{n}"

    nx.draw_networkx_labels(
        g,
        pos,
        labels,
        font_size=9,
        font_weight="bold",
    )

    # برچسب یال‌ها (مثلاً data_size) اگر بخواهی
    if show_edge_labels:
        edge_labels = {}
        for u, v, attr in g.edges(data=True):
            if "data_size" in attr:
                # اگر عدد خیلی بزرگ بود، خلاصه‌اش می‌کنیم
                edge_labels[(u, v)] = f"{attr['data_size']:.0f}"
        nx.draw_networkx_edge_labels(
            g,
            pos,
            edge_labels=edge_labels,
            font_size=7,
            label_pos=0.5,
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7),
        )

    # خطوط افقی lane ها برای وضوح بیشتر
    for task in range(num_tasks):
        y = task * lane_gap
        plt.hlines(
            y,
            xmin=-1.2,
            xmax=num_nodes + 1.5,
            colors="lightgray",
            linestyles="dotted",
            linewidth=0.6,
            zorder=0,
        )

    plt.title("Multi‑Task DAG", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    generator = DAGGenerator(
        num_tasks=3,
        num_nodes=5,
        max_out_degree=2
    )

    dag = generator.generate()

    print("Nodes:")
    for n, attr in dag.nodes(data=True):
        print(n, attr)

    print("\nEdges:")
    for u, v, attr in dag.edges(data=True):
        print(u, "->", v, attr)
    visualize_dag(dag, generator.num_tasks, generator.num_nodes)
