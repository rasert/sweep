from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import to_tree, ClusterNode
#!/usr/bin/python3
# -*- coding: utf-8 -*-

from typing import List


def scipy_tree_to_newick(node: ClusterNode, parentdist: float, leaf_names: List[str]) -> str:
    """
    Constructs a Newick tree from a SciPy hierarchy ClusterNode.

    Args:
        - node (scipy.cluster.hierarchy.ClusterNode): Root node,
          result of scipy.cluster.hierarchy.to_tree from the hierarchical
          clustering linkage matrix.
        - parentdist (float): Distance from the parent node to the `node`.
          leaf_names (list of string): Names of the leaf nodes.

    Returns:
        (string): String in Newick format.
    """
    if node.is_leaf():
        return f"{leaf_names[node.id]}:{parentdist - node.dist}"

    left_newick = scipy_tree_to_newick(node.get_left(), node.dist, leaf_names)
    right_newick = scipy_tree_to_newick(node.get_right(), node.dist, leaf_names)

    return f"({left_newick},{right_newick}):{parentdist - node.dist}"


def generate_tree(
    mat: List[List[float]],
    labels: List[str],
    output_file: str = None,
    metric: str = "euclidean",
    method: str = "complete",
) -> str:
    """
    Generates a Newick format tree from an input matrix.

    Args:
        - mat (List[List[float]]): Input matrix.
        - labels (List[str]): List of labels for the tree leaves.
        - output_file (str, optional): Name of the output file. If provided,
          the tree will be saved to this file.
        - metric (str): Distance metric to be used. Default is 'euclidean'.
        - method (str): Linkage method to be used in hierarchical clustering.
          Default is 'complete'.

    Returns:
        str: String in Newick format representing the tree.
    """
    # Calculates pairwise distances between embeddings
    dist = pdist(mat, metric=metric)

    # Performs hierarchical clustering and creates the linkage matrix
    Z = hierarchy.linkage(dist, method=method)

    # Converts the linkage matrix into a tree of ClusterNodes
    T = to_tree(Z, rd=False)

    # Converts the tree to Newick format
    newick = scipy_tree_to_newick(T, T.dist, labels) + ";"

    # If an output file name is provided, write the Newick string to the file
    if output_file:
        with open(output_file, "w") as f:
            f.write(newick)

    return newick
