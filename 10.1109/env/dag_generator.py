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
        edge_data_range=(100, 500), # KB
        rng=None,

    ):
        self.num_tasks = num_tasks
        
        self.num_nodes = num_nodes
        self.max_out_degree = max_out_degree
        self.alpha = alpha
        self.beta = beta

        self.cpu_range = cpu_range
        self.data_range = data_range
        self.edge_data_range = edge_data_range
        self.rng = rng if rng is not None else random.Random()

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
                    cpu_cycles=self.rng.uniform(*self.cpu_range),
                    data_size=self.rng.uniform(*self.data_range),
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

                out_degree = self.rng.randint(
                    0,
                    min(self.max_out_degree, len(possible_targets))
                )

                targets = self.rng.sample(
                    possible_targets,
                    out_degree
                )

                for dst in targets:

                    g.add_edge(
                        (task, src),
                        (task, dst),
                        data_size=self.rng.uniform(*self.edge_data_range)
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