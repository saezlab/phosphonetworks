"""Kinase activity scoring and benchmarking utilities."""

from __future__ import annotations

import os
import pickle
from typing import Iterable, Tuple

import decoupler as dc
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

import phosphonetworks as pp

def kinase_activity_analysis(
    site_data: pd.DataFrame,
    kinsub_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Perform kinase activity inference from phosphosite data using ULM model.

    Args:
        site_data (pd.DataFrame): Matrix of phosphosite measurements (sites × conditions).
        kinsub_df (pd.DataFrame): Kinase-substrate network with 'source' and 'target' columns.

    Returns:
        pd.DataFrame: Kinase activity results annotated with target counts and site coverage.
    """
    all_sites = site_data.columns.tolist()

    # if the column 'score' and not 'weight' is present, rename to 'weight'
    if 'score' in kinsub_df.columns and 'weight' not in kinsub_df.columns:
        kinsub_df = kinsub_df.rename(columns={'score': 'weight'})

    # Filter kinase-substrate pairs to only those in the data
    kinsub_df = kinsub_df[kinsub_df['target'].isin(all_sites)]

    # Count number of targets per kinase
    targets_per_kinase = (
        kinsub_df
        .groupby('source')['target']
        .nunique()
        .reset_index()
    )

    # Compute proportion of phosphosites covered
    proportion_sites_covered = len(kinsub_df['target'].unique()) / len(all_sites)

    # Run ULM model via dense_run
    kin_result = dc.dense_run(
        dc.run_zscore,
        net=kinsub_df,
        mat=site_data,
        verbose=False
    )

    # Reshape results to long format
    dc_df = (
        pd.DataFrame(kin_result[0].T)
        .reset_index()
        .melt(id_vars='index', var_name='condition', value_name='activity')
        .rename(columns={'index': 'kinase'})
    )

    # Merge with kinase target counts and annotate with coverage
    dc_df = (
        dc_df
        .merge(targets_per_kinase, left_on='kinase', right_on='source', how='left')
        .rename(columns={'target': 'total_targets'})
    )
    dc_df['proportion_sites_covered'] = proportion_sites_covered

    return dc_df

def multi_resource_kinase_activity_analysis(
    site_df: pd.DataFrame,
    kinsub_df: pd.DataFrame,
    id_col: str = 'id',
    value_col: str = 'logFC',
    condition_col: str = 'comparison'
) -> pd.DataFrame:
    """Run kinase activity inference for each resource separately."""

    dc_input_df = site_df[[id_col, value_col, condition_col]].dropna().copy()
    wide_df = dc_input_df.pivot(columns=id_col, index=condition_col, values=value_col)

    results = []
    for resource in kinsub_df['resource'].unique():
        print('Running analysis for resource:', resource)
        res_df = kinsub_df[kinsub_df['resource'] == resource]
        dc_res = kinase_activity_analysis(site_data=wide_df, kinsub_df=res_df)
        dc_res['resource'] = resource
        results.append(dc_res)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def hijazi_kinase_activity(
    hijazi_data: pd.DataFrame,
    kinsub_df: pd.DataFrame,
    cache_filename: str = 'intermediate_files/hijazi_kinase_activity.pkl'
) -> pd.DataFrame:
    """Score Hijazi et al. drug responses for every kinase resource, caching results."""

    cache_path = os.path.join(pp.config.CACHE_DIR, cache_filename)
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as handle:
            return pickle.load(handle)

    kinase_activities: list[pd.DataFrame] = []

    for cell in hijazi_data['cell'].unique():
        cell_data = hijazi_data[hijazi_data['cell'] == cell].copy()
        cell_sites = set(cell_data[cell_data['fold'] != 0]['site_id'])
        cell_data = cell_data[cell_data['site_id'].isin(cell_sites)].copy()
        cell_to_dc = cell_data.pivot_table(index='drug', columns='site_id', values='fold')

        for resource in tqdm(kinsub_df['resource'].unique()):
            resource_kinsub = kinsub_df[kinsub_df['resource'] == resource].copy()
            dc_df = kinase_activity_analysis(
                site_data=cell_to_dc,
                kinsub_df=resource_kinsub,
            )

            dc_df['resource'] = resource
            dc_df['cell'] = cell
            kinase_activities.append(dc_df)

    kinase_activities_df = pd.concat(kinase_activities, ignore_index=True)

    with open(cache_path, "wb") as handle:
        pickle.dump(kinase_activities_df, handle)

    return kinase_activities_df

def hijazi_roc_analysis(
    hijazi_kin_df: pd.DataFrame,
    hijazi_drug_to_kinase: pd.DataFrame,
    n_iterations: int = 100,
    only_kinobeads_kinases: bool = True,
    inhibition_threshold: float = 0.5,
    only_common_kinases: bool = True,
    kinase_superfamily: str | None = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run ROC benchmarking against inhibition profiles for each resource."""

    def log(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    merged_df = hijazi_kin_df.merge(
        hijazi_drug_to_kinase,
        left_on=['kinase', 'condition'],
        right_on=['kin_uniprot', 'drug'],
        how='left'
    )
    log('Merged DataFrame shape:', merged_df.shape)

    merged_df = merged_df.dropna(subset=['activity'])
    log('After dropping NaN activities, DataFrame shape:', merged_df.shape)

    if only_kinobeads_kinases:
        merged_df = merged_df.dropna(subset=['inhibition'])
        log('After dropping NaN inhibition, DataFrame shape:', merged_df.shape)

    if only_common_kinases:
        total_resources = merged_df['resource'].nunique()
        kin_per_resource = (
            merged_df[['kinase', 'resource']]
            .drop_duplicates()
            .groupby('kinase')
            .count()
        )
        kinases_to_keep = kin_per_resource[
            kin_per_resource['resource'] == total_resources
        ].index

        merged_df = merged_df[merged_df['kinase'].isin(kinases_to_keep)]

        log('After filtering for common kinases, DataFrame shape:', merged_df.shape)
        log('Number of common kinases:', len(kinases_to_keep))

    if kinase_superfamily:
        kin_info = pp.kinsub.get_kinase_info_df()
        int_kinases = kin_info[kin_info['superfamily'] == kinase_superfamily]['UniprotID'].unique()
        merged_df = merged_df[merged_df['kinase'].isin(int_kinases)].copy()
        log('After filtering for superfamily', kinase_superfamily, 'DataFrame shape:', merged_df.shape)

    # Define true positive labels
    merged_df['tp'] = 0
    merged_df['tp'] = np.where(
        merged_df['inhibition'] <= inhibition_threshold,
        1,
        merged_df['tp']
    )

    roc_summary_list = []
    roc_data_list = []

    for int_resource in tqdm(merged_df['resource'].unique()):
        resource_df = merged_df[merged_df['resource'] == int_resource].copy()
        tp_df = resource_df[resource_df['tp'] == 1].copy()
        tn_df = resource_df[resource_df['tp'] == 0].copy()

        np.random.seed(42)

        for i in range(n_iterations):
            tn_df_sampled = tn_df.sample(n=len(tp_df), replace=False)
            toroc_df = pd.concat([tp_df, tn_df_sampled], ignore_index=True)

            try:
                fpr, tpr, thresholds = roc_curve(
                    toroc_df['tp'],
                    toroc_df['activity'] * -1
                )
            except Exception:
                na_elements = toroc_df[toroc_df['activity'].isna()]
                log('NaN activities encountered in ROC computation:')
                log(na_elements)
                continue

            auroc_score = roc_auc_score(
                toroc_df['tp'],
                toroc_df['activity'] * -1
            )
            n_elements = len(toroc_df) / 2

            summary_results = (
                i,
                int_resource,
                n_elements,
                round(auroc_score, 4)
            )
            roc_summary_list.append(summary_results)

            roc_data = pd.DataFrame({
                'tpr': tpr,
                'fpr': fpr,
                'thresholds': thresholds,
                'iteration': i,
                'resource': int_resource,
                'n_elements': n_elements,
                'auroc_score': auroc_score
            })
            roc_data_list.append(roc_data)

    roc_summary_df = pd.DataFrame(
        roc_summary_list,
        columns=['iteration', 'resource', 'n_elements', 'auroc_score']
    )
    roc_data_df = pd.concat(roc_data_list, ignore_index=True)

    return roc_summary_df, roc_data_df


def hijazi_roc_wrapper(
    hijazi_kin_act: pd.DataFrame,
    hijazi_drug_to_target: pd.DataFrame,
    thresholds: Iterable[float] = np.arange(0.1, 0.9, 0.1),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute ROC summaries across multiple inhibition thresholds."""
    cache_file = os.path.join(pp.config.CACHE_DIR, 'intermediate_files/hijazi_roc.pickle')
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            roc_summary, roc_data = pickle.load(f)
    else:
        roc_list = []
        roc_data_list = []
        for int_threshold in thresholds:
            roc_summary, roc_data = hijazi_roc_analysis(
                hijazi_kin_act, hijazi_drug_to_target, 
                inhibition_threshold=int_threshold, verbose=False
            )
            roc_summary['threshold'] = int_threshold
            roc_data['threshold'] = int_threshold
            roc_list.append(roc_summary)
            roc_data_list.append(roc_data)

        roc_summary = pd.concat(roc_list, ignore_index=True)
        roc_data = pd.concat(roc_data_list, ignore_index=True)
        with open(cache_file, 'wb') as f:
            pickle.dump((roc_summary, roc_data), f)
    return roc_summary, roc_data

def egf_kinase_activity_analysis(
    kinsub: pd.DataFrame,
    data_df: pd.DataFrame,
    permutations: int = 0,
    cache_file: str | None = os.path.join(
        pp.config.CACHE_DIR, 'intermediate_files', 'egf_kinase_activity.pkl'
    ),
) -> pd.DataFrame:
    """Compute kinase activities per resource/study, with optional random permutations."""

    if cache_file is not None and os.path.exists(cache_file):
        print("Loading cached results from", cache_file)
        with open(cache_file, 'rb') as handle:
            return pickle.load(handle)

    todc = data_df.copy()
    results: list[pd.DataFrame] = []
    for resource in kinsub['resource'].unique():
        print(resource)
        res_df = kinsub[kinsub['resource'] == resource]
        for study in todc['study'].unique():
            print("  ", study)
            study_df = todc[todc['study'] == study]
            wide_study = study_df[['logFC', 'comparison', 'id']].pivot(
                index='comparison', columns='id', values='logFC'
            )
            dc_res = kinase_activity_analysis(site_data=wide_study, kinsub_df=res_df)
            dc_res['resource'] = resource
            dc_res['study'] = study
            dc_res['random'] = False
            dc_res['iteration'] = 0
            results.append(dc_res)
            if permutations > 0:
                for iteration in range(permutations):
                    shuffled_kinsub = res_df.copy()
                    shuffled_kinsub['target'] = np.random.permutation(
                        shuffled_kinsub['target'].values
                    )
                    shuffled_kinsub = shuffled_kinsub.drop_duplicates(
                        subset=['source', 'target']
                    )
                    dc_res = kinase_activity_analysis(
                        site_data=wide_study,
                        kinsub_df=shuffled_kinsub,
                    )
                    dc_res['resource'] = resource
                    dc_res['study'] = study
                    dc_res['random'] = True
                    dc_res['iteration'] = iteration + 1
                    results.append(dc_res)

    final_df = pd.concat(results, ignore_index=True)
    if cache_file is not None:
        print("Saving results to cache file", cache_file)
        with open(cache_file, 'wb') as handle:
            pickle.dump(final_df, handle)
    return final_df
