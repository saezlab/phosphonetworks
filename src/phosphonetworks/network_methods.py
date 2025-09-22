"""Network inference helpers for kinase activity propagation analyses."""

from __future__ import annotations

import os
import pickle
from typing import Dict, Iterable, List

import cvxpy as cp
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

import phosphonetworks as pp

def compute_overlap_with_gt(
    gt_ints: Dict[str, Iterable[str]],
    results: Iterable[Dict[str, object]],
) -> pd.DataFrame:
    """Summarise network overlaps against multiple ground-truth interaction sets."""

    res_df = []
    for gt_type, ints in gt_ints.items():
        print(f"GT type: {gt_type}, number of interactions: {len(ints)}")
        for res in results:
            n_edges_in_pkn = res['pkn'].number_of_edges()
            n_nodes_in_pkn = res['pkn'].number_of_nodes()
            all_pkn_edges = set([ f"{u}-->{v}" for u, v in res['pkn'].edges()])
            max_overlap_with_gt = all_pkn_edges.intersection(set(ints))
            edge_length = res['edge_length']
            for method in ['mean_net', 'pr_net', 'pcst_net']:
                subgraph = res[method]
                if subgraph is None:
                    n_edges_in_subgraph = 0
                    n_nodes_in_subgraph = 0
                    overlap_with_gt = set()
                else:
                    n_edges_in_subgraph = subgraph.number_of_edges()
                    n_nodes_in_subgraph = subgraph.number_of_nodes()
                    subgraph_edges = set([ f"{u}-->{v}" for u, v in subgraph.edges()])
                    overlap_with_gt = subgraph_edges.intersection(set(ints))
                res_df.append({
                    'gt_type': gt_type,
                    'n_gt_ints': len(ints),
                    'n_edges_in_pkn': n_edges_in_pkn,
                    'n_nodes_in_pkn': n_nodes_in_pkn,
                    'max_overlap_with_gt': len(max_overlap_with_gt),
                    'n_edges_in_subgraph': n_edges_in_subgraph,
                    'n_nodes_in_subgraph': n_nodes_in_subgraph,
                    'overlap_with_gt': len(overlap_with_gt),
                    'method': method,
                    'resource': res['resource'],
                    'study': res['study'],
                    'edge_length': edge_length
                })

    res_df = pd.DataFrame(res_df)

    res_df = res_df[~((res_df['gt_type'].str.contains('hek293')) & (res_df['study'] == 'This study (HEK293F TR)'))].copy()
    res_df = res_df[res_df['max_overlap_with_gt'] > 0].copy()
    res_df['norm_overlap_with_gt'] = res_df['overlap_with_gt'] / res_df['max_overlap_with_gt']
    control_studies = ['Chen et al. 2025 (HEK293T)', 'Tuechler et al. 2025 (PDGFRb)']
    res_df['is_control_study'] = res_df['study'].isin(control_studies)
    return res_df

def prepare_input_dict(
    kinase_df: pd.DataFrame,
    edge_lengths: Iterable[int],
    outfile: str = os.path.join(
        pp.config.CACHE_DIR, 'intermediate_files', 'network_input_dicts.pkl'
    )
) -> List[Dict[str, object]]:
    """Materialise terminal sets per resource/study combination for network solvers."""
    resources = kinase_df['resource'].unique()
    studies = kinase_df['study'].unique()
    if os.path.exists(outfile):
        print(f"Input dicts at {outfile} already exist")
        with open(outfile, 'rb') as f:
            input_dicts = pickle.load(f)
        return input_dicts
    input_dicts = []
    kk_pkn = pp.kinsub.get_kk_pkn()
    for resource in resources:
        print(f" Processing resource: {resource}")
        kk_resource = kk_pkn[kk_pkn['resource'] == resource].copy()
        kk_resource_pkn = nx.from_pandas_edgelist(kk_resource, 'source', 'target', edge_attr='weight', create_using=nx.DiGraph)
        for study in studies:
            study_kinases = kinase_df[kinase_df['study'] == study].copy()
            study_kinases = study_kinases[['kinase', 'abs_activity']].set_index('kinase').to_dict()['abs_activity']
            filt_pkn, filt_terminals = pp.utils.prepare_pkn_and_terminals(kk_resource_pkn, study_kinases, verbose = 0)
            for edge_length in edge_lengths:
                input_dicts.append({'pkn': filt_pkn, 'terminals': filt_terminals, 'edge_length': edge_length, 'resource': resource, 'study': study})

    print(f"Total input dicts prepared: {len(input_dicts)}")
    print(f"Saving input dicts to {outfile}")

    with open(outfile, 'wb') as f:
        pickle.dump(input_dicts, f)
    return input_dicts

def mean_selection(
    network: nx.DiGraph,
    terminals: Dict[str, float],
    n_edges: int,
) -> nx.DiGraph:
    """Select top edges by average terminal activity weighted by edge weight."""
    edge_df = nx.to_pandas_edgelist(network)
    edge_df['source_value'] = edge_df['source'].map(terminals)
    edge_df['target_value'] = edge_df['target'].map(terminals)
    edge_df = edge_df.dropna()
    edge_df['rank_value'] = np.nanmean(edge_df[['source_value', 'target_value']], axis=1) * edge_df['weight']
    edge_df = edge_df.sort_values('rank_value', ascending=False)
    filt_edges = edge_df[['source', 'target', 'weight']].head(n_edges)
    subg = nx.from_pandas_edgelist(filt_edges, 'source', 'target', ['weight'], create_using=nx.DiGraph())
    return subg

def pagerank_selection(
    network: nx.DiGraph,
    terminals: Dict[str, float],
    n_edges: int,
    damp_factor: float = 0.85,
) -> nx.DiGraph:
    """Select edges by personalised PageRank importance."""
    pr = nx.pagerank(network, alpha=damp_factor, weight='weight', personalization=terminals)
    edge_df = nx.to_pandas_edgelist(network)
    edge_df['source_value'] = edge_df['source'].map(pr)
    edge_df['target_value'] = edge_df['target'].map(pr)
    edge_df = edge_df.dropna()
    edge_df['rank_value'] = np.nanmean(edge_df[['source_value', 'target_value']], axis=1) 
    edge_df = edge_df.sort_values('rank_value', ascending=False)
    filt_edges = edge_df[['source', 'target', 'weight']].head(n_edges)
    subg = nx.from_pandas_edgelist(filt_edges, 'source', 'target', ['weight'], create_using=nx.DiGraph())
    return subg

def rpcst_selection(
    network: nx.DiGraph,
    terminals: Dict[str, float],
    root: str = 'P00533',
    n_edges: int = 10,
    verbose: int = 0,
    mip: float = 0.01,
) -> nx.DiGraph:
    """Solve a rooted PCST formulation using cvxpy and a GUROBI backend."""

    # add a node called perturbation that is connected to all nodes
    ph_net = network.copy()
    edges_to_add = [('perturbation', root, {'weight': 1})]
    ph_net.add_edges_from(edges_to_add)

    m = 1000
    edge_weights = np.array([ph_net[i][j]['weight'] for i, j in ph_net.edges()])
    use_weights = not np.all(edge_weights == 1)

    nodes = list(ph_net.nodes())
    incidence_matrix = nx.incidence_matrix(ph_net, oriented=True).tocsr()

    # define variables and indices
    node_vars = cp.Variable(len(nodes), boolean=True)
    edge_vars = cp.Variable(len(ph_net.edges()), boolean=True)
    node_indices = {i: j for j, i in enumerate(nodes)}
    edge_indices = {i + '-->' + j: k for k, (i, j) in enumerate(ph_net.edges())}
    measured_nodes_indices = [node_indices[i] for i in terminals.keys()]

    # C1: Edge is only selected if both target and source are selected (source and target >= 2)
    source_indices = [node_indices[i.split('-->')[0]] for i in edge_indices.keys()]
    target_indices = [node_indices[i.split('-->')[1]] for i in edge_indices.keys()]
    constraints = [node_vars[source_indices] + node_vars[target_indices] >= 2 * edge_vars]

    # C2: If node is selected, but is not 'perturbation', it should have at least one incoming edge
    receiver_node_indices = [node_indices[i] for i in nodes if i != 'perturbation']
    incidence_matrix_subset = incidence_matrix[receiver_node_indices, :]
    incidence_matrix_subset.data[incidence_matrix_subset.data < 0] = 0
    constraints += [node_vars[receiver_node_indices] <= edge_vars @ incidence_matrix_subset.T] 

    # C3: All nodes included that are not p_nodes should have at least one outgoing edge if selected
    sender_node_indices = [node_indices[i] for i in nodes if i not in terminals.keys()]
    incidence_matrix_subset = incidence_matrix[sender_node_indices, :] * -1
    incidence_matrix_subset.data[incidence_matrix_subset.data < 0] = 0
    constraints += [node_vars[sender_node_indices] <= edge_vars @ incidence_matrix_subset.T]

    # C3: All nodes included that are not p_nodes should have at least one outgoing edge if selected
    sender_node_indices = [node_indices[i] for i in nodes if i not in terminals.keys()]
    incidence_matrix_subset = incidence_matrix[sender_node_indices, :] * -1
    incidence_matrix_subset.data[incidence_matrix_subset.data < 0] = 0
    constraints += [node_vars[sender_node_indices] <= edge_vars @ incidence_matrix_subset.T]

    # C4: Loop breaking constraint, if edge is selected, source should be at least one further away from perturbation than target
    distance_vars = cp.Variable(len(nodes), integer=True)
    constraints += [node_vars <= distance_vars]
    all_source_indices = incidence_matrix * -1
    all_source_indices.data[all_source_indices.data < 0] = 0
    all_target_indices = incidence_matrix
    all_target_indices.data[all_target_indices.data < 0] = 0
    source_distance_vars = all_source_indices.T @ distance_vars
    target_distance_vars = all_target_indices.T @ distance_vars
    constraints += [target_distance_vars >= source_distance_vars + 1 - m + (m * edge_vars)]
    constraints += [distance_vars <= m]
    constraints += [distance_vars[node_indices['perturbation']] == 1]

    # select exactly n_edges
    constraints += [cp.sum(edge_vars) == n_edges + 1]

    # objective function
    node_weights = np.array([terminals[i] for i in terminals.keys()])
    measured_nodes_error = cp.sum(cp.abs((node_vars[measured_nodes_indices] - np.ones(len(measured_nodes_indices))) @ node_weights))
    if use_weights:
        edge_weights = 1 - edge_weights
        objective = cp.Minimize(measured_nodes_error + (cp.sum(edge_vars @ edge_weights)))
    else:
        objective = cp.Minimize(measured_nodes_error)

    # problem and objective
    problem = cp.Problem(objective, constraints)
    problem.solve(
        solver='GUROBI', verbose=verbose, 
        time_limit=60, 
        #threads=1,
        mipGap=mip
    )

    # create a subgraph with selected edges
    selected_edges = [i for i, v in zip(ph_net.edges(), edge_vars.value) if v > 0.5]
    subgraph = nx.edge_subgraph(ph_net, selected_edges).copy()
    subgraph.remove_node('perturbation')
    return subgraph

def solve_networks(input_dict_list: str, output_path: str) -> List[Dict[str, object]]:
    """Apply multiple subnet selection strategies across cached inputs."""

    if os.path.exists(output_path):
        print(f"Output path {output_path} exists, loading and returning")
        with open(output_path, 'rb') as f:
            return pickle.load(f)

    with open(input_dict_list, 'rb') as f:
        data = pickle.load(f)
    outdata = []

    for int_data in tqdm(data):
        int_element = int_data.copy()
        mean_net = mean_selection(int_data['pkn'], int_data['terminals'], int_data['edge_length'])
        pr_net = pagerank_selection(int_data['pkn'], int_data['terminals'], int_data['edge_length'])
        pcst_net = rpcst_selection(int_data['pkn'], int_data['terminals'], n_edges = int_data['edge_length'], mip = 0.05)
        # save networks as new elements in int_data
        int_element['mean_net'] = mean_net
        int_element['pr_net'] = pr_net 
        int_element['pcst_net'] = pcst_net
        outdata.append(int_element)
    # save to pickle
    with open(output_path, 'wb') as f:
        pickle.dump(outdata, f)
    print(f"Saved output to {output_path}")
    return outdata
