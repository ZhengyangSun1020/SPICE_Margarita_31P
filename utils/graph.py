"""
Graph and spatial regularization utilities: neighbour graphs, Laplacian, min-pooling.
"""

import numpy as np
import networkx as nx
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt


def calc_neighbours(img, adj=8, plot=False):
    """Get neighbourhood pairs.
    adj is 4 or 8; returns pairs (i,j) indexing the whole image.
    """
    assert (adj == 4) or (adj == 8)
    radius = 1.001 if adj == 4 else 1.415
    xy = np.indices(img.shape).reshape((2, -1)).T
    tree = KDTree(xy)
    Nb = tree.query_pairs(p=2, r=radius, output_type='ndarray')

    if plot:
        plt.figure(figsize=(6, 6))
        plt.scatter(xy[:, 1], xy[:, 0], s=10, color='black')
        for i, j in Nb:
            p1, p2 = xy[i], xy[j]
            plt.plot([p1[1], p2[1]], [p1[0], p2[0]], 'b-', linewidth=1)
        plt.gca().invert_yaxis()
        plt.title("Neighbour Links (4-adjacency)")
        plt.axis('equal')
        plt.grid(True)
        plt.show()
    return Nb, xy


def calc_W(ref_vec, wmax, Nb):
    """Compute edge confidence weights from reference vector differences."""
    C = (np.diff(ref_vec[Nb], axis=1)**2).flatten()
    np.seterr(divide='ignore')
    W = np.minimum(1 / C, wmax)
    return W


def calc_Bmatrix(ref, wmax, adj, pool_size: int = 2, K: int = 1, ak: list = [1],
                 minpooling_Handler: bool = False, brain_mask: np.ndarray = None,
                 mask_dilate_layers: int = 4, mask_rule: str = 'any'):
    """
    Compute B such that loss = ||B rho||^2.

    Parameters
    ----------
    brain_mask : optional boolean mask (same shape as ref).
        If provided, edge weights outside the dilated mask are reduced.
        mask_rule 'any': zero any edge with at least one endpoint outside mask.
        mask_rule 'both': zero edges only when both endpoints are outside.
    """
    from scipy.ndimage import binary_dilation

    Nb, _ = calc_neighbours(ref, adj)
    ref_vec = ref.flatten()
    n_vox = ref_vec.size
    W = calc_W(ref_vec, wmax, Nb)

    if brain_mask is not None:
        mask = np.asarray(brain_mask, dtype=bool)
        if mask.shape != ref.shape:
            raise ValueError("brain_mask must have the same shape as ref")
        dilated_mask = binary_dilation(mask, iterations=mask_dilate_layers)
        mask_flat = dilated_mask.flatten()

        if mask_rule == 'any':
            out_of_mask = ~(mask_flat[Nb[:, 0]] & mask_flat[Nb[:, 1]])
            if np.any(out_of_mask):
                W = W.copy()
                W[out_of_mask] = wmax / 10.0
        elif mask_rule == 'both':
            both_out = (~mask_flat[Nb[:, 0]]) & (~mask_flat[Nb[:, 1]])
            if np.any(both_out):
                W = W.copy()
                W[both_out] = wmax / 10.0
        else:
            raise ValueError("mask_rule must be 'any' or 'both'")

    if minpooling_Handler:
        dim_x, dim_y = ref.shape
        edge_index = [tuple(pair) for pair in Nb]
        W = directional_min_pool(W, edge_index, dim_x, dim_y, pool_size)

    A = csr_matrix((W, (Nb[:, 0], Nb[:, 1])), shape=(n_vox, n_vox)).toarray()
    A = 0.5 * (A + A.T)

    L = np.diag(np.sum(A, axis=0)) + np.diag(np.sum(A, axis=1)) - 2.0 * A

    D, V = np.linalg.eigh(L)
    B = np.diag(np.sqrt(np.abs(D))) @ V.T

    return B, A, W, Nb


def min_pool_graph(mask, edge_index, num_voxels, pool_size=1):
    """
    Perform min pooling on a graph where edge weights are provided.
    For each edge (v1, v2), its new value is the minimum edge weight in the
    k-hop neighborhood of v1 and v2.
    """
    G = nx.Graph()
    G.add_edges_from(edge_index)

    edge_to_weight = {}
    for idx, (v1, v2) in enumerate(edge_index):
        edge_to_weight[(v1, v2)] = mask[idx]
        edge_to_weight[(v2, v1)] = mask[idx]

    pooled = np.full((num_voxels,), np.inf)

    for v in range(num_voxels):
        nodes = nx.single_source_shortest_path_length(G, v, cutoff=pool_size).keys()
        local_edges = []
        for n1 in nodes:
            for n2 in G[n1]:
                if n2 in nodes and (n1, n2) in edge_to_weight:
                    local_edges.append(edge_to_weight[(n1, n2)])
        if local_edges:
            pooled[v] = min(local_edges)
        else:
            pooled[v] = 0.0

    pooled_edges = np.array([
        min(pooled[v1], pooled[v2])
        for v1, v2 in edge_index
    ])

    return pooled_edges


def sliding_min_pool_1d(arr, window_size):
    """Simple 1D sliding window min pooling."""
    n = len(arr)
    pooled = np.full(n, np.inf)
    half_w = window_size // 2

    for i in range(n):
        start = max(0, i - half_w)
        end = min(n, i + half_w + 1)
        pooled[i] = np.min(arr[start:end])
    return pooled


def classify_edge_direction(v1, v2, dim_y):
    """Classify an edge as vertical, horizontal, main_diag, or anti_diag."""
    x1, y1 = divmod(v1, dim_y)
    x2, y2 = divmod(v2, dim_y)
    dx = x2 - x1
    dy = y2 - y1
    if abs(dx) == 1 and dy == 0:
        return 'vertical'
    elif dx == 0 and abs(dy) == 1:
        return 'horizontal'
    elif abs(dx) == 1 and abs(dy) == 1:
        if dx == dy:
            return 'main_diag'
        else:
            return 'anti_diag'
    else:
        return None


def directional_min_pool(mask, edge_index, dim_x, dim_y, pool_size=1):
    """Apply directional min-pooling to edge weights on a 2D grid graph."""
    G = nx.Graph()
    G.add_edges_from(edge_index)

    edge_to_idx = {}
    for i, (v1, v2) in enumerate(edge_index):
        edge_to_idx[(v1, v2)] = i
        edge_to_idx[(v2, v1)] = i

    edges_by_dir = {'vertical': [], 'horizontal': [], 'main_diag': [], 'anti_diag': []}
    for v1, v2 in edge_index:
        d = classify_edge_direction(v1, v2, dim_y)
        if d is not None:
            edges_by_dir[d].append((v1, v2))

    updated_weights = np.full((len(mask), len(edges_by_dir)), np.inf)

    for d_idx, (direction, edges_dir) in enumerate(edges_by_dir.items()):
        G_sub = nx.Graph()
        G_sub.add_edges_from(edges_dir)

        for comp in nx.connected_components(G_sub):
            sub_nodes = list(comp)

            def proj(node):
                x, y = divmod(node, dim_y)
                if direction == 'vertical':
                    return x
                elif direction == 'horizontal':
                    return y
                elif direction == 'main_diag':
                    return x + y
                elif direction == 'anti_diag':
                    return x - y
                else:
                    return 0

            sub_nodes.sort(key=proj)

            chain_edges = []
            for i in range(len(sub_nodes) - 1):
                e = (sub_nodes[i], sub_nodes[i + 1])
                e_key = (e[0], e[1]) if (e[0], e[1]) in edge_to_idx else (e[1], e[0])
                idx_edge = edge_to_idx[e_key]
                chain_edges.append(idx_edge)

            if not chain_edges:
                continue

            weights_chain = mask[chain_edges]
            pooled_chain = sliding_min_pool_1d(weights_chain, pool_size)

            for idx_edge, val in zip(chain_edges, pooled_chain):
                updated_weights[idx_edge, d_idx] = val

    combined = np.min(updated_weights, axis=1)
    combined[np.isinf(combined)] = mask[np.isinf(combined)]

    return combined
