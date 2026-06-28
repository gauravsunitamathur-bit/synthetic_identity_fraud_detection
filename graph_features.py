"""
graph_features.py
==================
Builds an identity-linkage graph and derives "entity resolution" style
features for synthetic identity fraud detection.

DESIGN CHOICE: BIPARTITE GRAPH (applications <-> identity elements)
---------------------------------------------------------------------
We build a BIPARTITE graph rather than a direct application-to-application
projection, for three reasons:

1. Interpretability: it's straightforward to explain to a fraud ops
   audience — "this application is connected to this address node, which
   is connected to 6 other applications" — versus an abstract weighted
   projection edge.
2. Efficiency: building an application-application projection graph at
   scale means computing all pairwise shared-element relationships, which
   is O(n^2) in the worst case for popular shared elements (e.g. a single
   address used in 500 applications would create ~125,000 edges in a
   projected graph). The bipartite form keeps the graph sparse: each
   application has only as many edges as it has identity elements (5),
   regardless of how many other applications share those elements.
3. Multi-relational signal preserved natively: in a projected graph you'd
   need a separate edge-weight per element type if you want to know
   *which* element(s) two applications share. In the bipartite graph this
   falls out for free — we can directly query "how many distinct element
   TYPES does this application share with its connected component."

We then derive application-level features by:
  - Computing connected components on the bipartite graph (a fraud "ring"
    and its associated identity elements all fall into one component)
  - For each application node, looking at its direct neighbors (identity
    elements) and 2-hop neighbors (other applications sharing those
    elements) to compute degree / shared-element-type counts
  - Projecting OUT a much smaller application-only view only where useful
    (for the EDA visualization of ring structure), not for the main
    feature computation, to keep this efficient at ~40k applications.

TRAIN/TEST LEAKAGE NOTE
------------------------
Because two applications in the same ring are connected via shared
identity elements, splitting rows into train/test independently can leak
information: a ring's structure (e.g. "this address has 7 applications")
would be partially visible in both train and test if siblings end up on
both sides. We therefore split by CONNECTED COMPONENT, not by row, in
train_model.py. This module just exposes a `component_id` per application
so that split can be done correctly downstream.
"""

import networkx as nx
import pandas as pd
import numpy as np


IDENTITY_ELEMENT_COLUMNS = {
    "ssn": "ssn",
    "address": "address",
    "device_fingerprint": "device_fingerprint",
    "email": "email",
    "phone_number": "phone_number",
}


def build_bipartite_graph(df: pd.DataFrame) -> nx.Graph:
    """
    Build a bipartite graph: application nodes <-> identity-element nodes.
    Identity element nodes are namespaced by type (e.g. "address::ADDR_001")
    so that the same literal string used in two different element types
    (unlikely given our prefixes, but good practice) never collides.
    """
    G = nx.Graph()

    for _, row in df.iterrows():
        app_node = ("app", row["application_id"])
        G.add_node(app_node, node_type="application")

        for elem_type, col in IDENTITY_ELEMENT_COLUMNS.items():
            elem_value = row[col]
            elem_node = ("elem", elem_type, elem_value)
            G.add_node(elem_node, node_type="identity_element", element_type=elem_type)
            G.add_edge(app_node, elem_node, element_type=elem_type)

    return G


def compute_graph_features(df: pd.DataFrame, G: nx.Graph) -> pd.DataFrame:
    """
    Derive per-application graph features from the bipartite identity graph.

    Features:
    - component_id: which connected component this application belongs to
      (used for leakage-safe train/test splitting, not as a model feature
      itself, since raw component IDs are arbitrary labels with no
      meaningful ordinal/categorical structure for a model to learn from)
    - component_size_apps: number of DISTINCT applications in this
      application's connected component (the "ring size")
    - app_degree: number of distinct identity elements this application
      has (should be exactly 5 here — ssn/address/device/email/phone —
      kept for completeness and as a sanity check)
    - max_element_reuse: across this application's 5 identity elements,
      the highest number of OTHER applications also using that same
      element (i.e. "my most-reused identity element is shared with N
      other applications")
    - n_elements_reused: how many of this application's 5 identity
      elements are shared with at least one other application (0-5)
    - n_distinct_element_types_reused: how many DISTINCT element TYPES
      (out of ssn/address/device/email/phone) are shared with at least
      one other application that ALSO shares a different element type
      with this one — i.e. evidence of a multi-element link to the same
      neighbor(s), which is a stronger signal than two applications
      coincidentally sharing just one element (e.g. living at the same
      address is common for families; ALSO sharing a device fingerprint
      with that same neighbor is a much rarer, stronger fraud signal)
    - shares_multgovern_element_with_any_neighbor: boolean flag for the
      above, thresholded at >=2 shared element types with the same
      specific neighbor application
    """
    component_map = {}
    component_size_map = {}

    for component in nx.connected_components(G):
        app_nodes_in_component = [n for n in component if n[0] == "app"]
        comp_id = min(app_nodes_in_component, key=lambda x: x[1])[1] if app_nodes_in_component else None
        size = len(app_nodes_in_component)
        for app_node in app_nodes_in_component:
            component_map[app_node[1]] = comp_id
            component_size_map[app_node[1]] = size

    feature_rows = []
    for _, row in df.iterrows():
        app_id = row["application_id"]
        app_node = ("app", app_id)

        elem_neighbors = list(G.neighbors(app_node))  # the 5 identity-element nodes
        app_degree = len(elem_neighbors)

        # For each identity element, find OTHER applications that also touch it
        reuse_counts = []
        neighbor_app_to_shared_types = {}  # other_app_id -> set of element types shared with THIS app

        for elem_node in elem_neighbors:
            elem_type = elem_node[1]
            other_apps_for_elem = [
                n[1] for n in G.neighbors(elem_node) if n[0] == "app" and n[1] != app_id
            ]
            reuse_counts.append(len(other_apps_for_elem))
            for other_app_id in other_apps_for_elem:
                neighbor_app_to_shared_types.setdefault(other_app_id, set()).add(elem_type)

        max_element_reuse = max(reuse_counts) if reuse_counts else 0
        n_elements_reused = sum(1 for c in reuse_counts if c > 0)

        # multi-element overlap with the SAME neighbor = stronger ring signal
        multi_element_neighbors = [
            other for other, types in neighbor_app_to_shared_types.items() if len(types) >= 2
        ]
        n_distinct_element_types_reused = (
            max((len(types) for types in neighbor_app_to_shared_types.values()), default=0)
        )
        shares_multi_element_with_any_neighbor = len(multi_element_neighbors) > 0

        feature_rows.append({
            "application_id": app_id,
            "component_id": component_map.get(app_id),
            "component_size_apps": component_size_map.get(app_id, 1),
            "app_degree": app_degree,
            "max_element_reuse": max_element_reuse,
            "n_elements_reused": n_elements_reused,
            "n_distinct_element_types_reused_with_top_neighbor": n_distinct_element_types_reused,
            "shares_multi_element_with_any_neighbor": shares_multi_element_with_any_neighbor,
            "n_neighbors_sharing_2plus_elements": len(multi_element_neighbors),
        })

    feat_df = pd.DataFrame(feature_rows)
    return feat_df


def attach_graph_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper: build graph, compute features, merge onto df."""
    G = build_bipartite_graph(df)
    feat_df = compute_graph_features(df, G)
    merged = df.merge(feat_df, on="application_id", how="left")
    return merged, G


if __name__ == "__main__":
    df = pd.read_csv("/home/claude/synthetic_identity_fraud/data/applications_raw.csv")
    merged, G = attach_graph_features(df)
    print("Graph: {} nodes, {} edges".format(G.number_of_nodes(), G.number_of_edges()))
    print("Number of connected components:", nx.number_connected_components(G))
    print(merged[["component_size_apps", "max_element_reuse",
                   "n_elements_reused", "shares_multi_element_with_any_neighbor"]].describe())
    print()
    print("Component size distribution by fraud label:")
    print(merged.groupby("is_fraud")["component_size_apps"].describe()[["mean", "50%", "max"]])
    merged.to_csv("/home/claude/synthetic_identity_fraud/data/applications_with_graph_features.csv", index=False)
    print("Saved merged dataset with graph features.")
