"""Data loading and preprocessing utilities for phosphoproteomics datasets."""

from __future__ import annotations

import os
import re
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from phosphonetworks import config, netgt, utils

def get_hijazi() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load Hijazi et al. drug-response data and curate inhibition metadata."""

    uniprot_to_symbol, symbol_to_uniprot = utils.get_uniprot_symbol_mapping()
    cache_dir = config.CACHE_DIR

    hl_60_file = os.path.join(cache_dir, "phosphodata/hijazi/HL60.xlsm")
    mcf7_file = os.path.join(cache_dir, "phosphodata/hijazi/MCF7.xlsm")
    drug_specs_file = os.path.join(cache_dir, "phosphodata/hijazi/kinaseInhibitionSpecificity.csv")

    hijazi_hl60 = pd.read_excel(hl_60_file, sheet_name='fold')
    hijazi_mcf7 = pd.read_excel(mcf7_file, sheet_name='fold')
    hl60_long = hijazi_hl60.melt(id_vars=['sh.index.sites', 'FDR'], var_name = 'treat', value_name='fold')
    hl60_long['cell'] = 'HL60'
    mcf7_long = hijazi_mcf7.melt(id_vars=['sh.index.sites', 'FDR'], var_name = 'treat', value_name='fold')
    mcf7_long['cell'] = 'MCF7'
    hijazi_long = pd.concat([hl60_long, mcf7_long])
    hijazi_long = hijazi_long[hijazi_long['FDR'] <= 0.05].copy()
    hijazi_long['site'] = hijazi_long['sh.index.sites'].str.extract(r'\((.*?)\)')
    hijazi_long['symbol'] = hijazi_long['sh.index.sites'].str.extract(r'(.*?)\(')
    hijazi_long['uniprot'] = hijazi_long['symbol'].map(symbol_to_uniprot)
    hijazi_long['drug'] = hijazi_long['treat'].str.replace(r'\.fold', '', regex=True)
    hijazi_long['drug'] = hijazi_long['drug'].str.replace(r'^.*\.', '', regex=True)

    drug_specs = pd.read_csv(drug_specs_file)
    drug_long = drug_specs.melt(id_vars='kinase', var_name='drug', value_name='inhibition')

    drug_to_drug = []
    for drug in hijazi_long['drug'].unique():
        for value in drug_long['drug'].unique():
            if str(drug.lower()) in str(value.lower()):
                drug_to_drug.append((drug, value))
    dd_df = pd.DataFrame(drug_to_drug, columns=['dname', 'drug'])
    drug_long = drug_long.merge(dd_df, on='drug', how='left')
    
    drug_long['kinase_dec'] = drug_long['kinase'].str.split('.')
    drug_long['kinase_dec'] = drug_long['kinase_dec'].map(lambda x: [x[0], x[0][:-1] + x[1]] if len(x) > 1 else x)
    drug_long['drug'] = drug_long['drug'].str.split('.').str[1]
    drug_long = drug_long.explode('kinase_dec')
    ammendments = {
        'ICK': 'CLK1',
        'PAK7': 'PAK5'
    }
    drug_long['kinase_dec'] = drug_long['kinase_dec'].map(lambda x: ammendments[x] if x in ammendments else x)
    drug_long['uniprot'] = drug_long['kinase_dec'].map(symbol_to_uniprot)
    drug_long = drug_long.dropna()
    drug_to_kinase = drug_long.copy()

    drug_to_kinase['keep'] = drug_to_kinase.groupby(['drug', 'uniprot'])['inhibition'].transform(lambda x: x == x.min())
    drug_to_kinase = drug_to_kinase[drug_to_kinase['keep']].copy()
    drug_to_kinase['top_specific'] = drug_to_kinase.groupby('drug')['inhibition'].transform(lambda x: x == x.min())  
    drug_to_kinase = drug_to_kinase[['drug', 'uniprot', 'inhibition', 'top_specific']].drop_duplicates().rename(columns={'uniprot': 'kin_uniprot'})

    hijazi_long['site_id'] = hijazi_long['uniprot'] + '_' + hijazi_long['site']
    hijazi_long = hijazi_long[['drug', 'site_id', 'fold','cell']]

    return hijazi_long, drug_to_kinase

def get_egf_site_data() -> pd.DataFrame:
    """Return the preprocessed EGF stimulation dataset."""

    egf_path = os.path.join(
        config.CACHE_DIR, "intermediate_files/preprocessed_egf_data.csv.gz"
    )
    return pd.read_csv(egf_path, compression='gzip', low_memory=False)

def get_chen_interferon_data() -> pd.DataFrame:
    """Load Chen et al. interferon phosphoproteomics timepoints."""

    chen_path = os.path.join(config.CACHE_DIR, "phosphodata/chen_interferon/mmc10.xlsx")
    chen_df = pd.read_excel(chen_path, sheet_name=1)
    uniprot_to_symbol, symbol_to_uniprot = utils.get_uniprot_symbol_mapping()
    chen_df['uniprot'] = chen_df['GeneName'].map(symbol_to_uniprot)
    chen_df['site'] = chen_df['Phosphosite'].str.replace('.*_', '', regex=True)
    chen_df['site_id'] = chen_df['uniprot'] + '_' + chen_df['site']

    # Define the timepoints and initialize a list to collect dataframes
    timepoints = ['4h', '10min']
    data_frames = []

    # Loop through each timepoint and process
    for tp in timepoints:
        cols = [
            'site_id',
            f'logFC_Phospho.{tp}',
            f'P.Value_Phospho.{tp}',
            f'FDR_Phospho.{tp}',
            f'Z.score_Phospho.{tp}',
            f'AveExpr_Phospho.{tp}'
        ]
        
        df = chen_df[cols].copy()
        df.rename(columns={
            f'logFC_Phospho.{tp}': 'logFC',
            f'P.Value_Phospho.{tp}': 'p_value',
            f'FDR_Phospho.{tp}': 'fdr',
            f'Z.score_Phospho.{tp}': 'z_score',
            f'AveExpr_Phospho.{tp}': 'ave_expr'
        }, inplace=True)
        df['comparison'] = tp
        data_frames.append(df)

    # Concatenate and annotate the final dataframe
    data_df = pd.concat(data_frames, ignore_index=True)

    # group by site_id and comparison and keep the row with the lowest p_value
    data_df = data_df.sort_values('p_value').drop_duplicates(subset=['site_id', 'comparison'], keep='first')
    data_df['study'] = 'Chen et al. 2025 (HEK293T)'

    data_df = data_df.dropna(subset=['site_id'])

    # rename to match the expected format
    data_df = data_df[[
        'site_id', 'logFC', 'ave_expr', 'z_score', 
        'p_value', 'fdr', 'study', 'comparison'
    ]].rename(columns={
        'site_id': 'id',
        'ave_expr': 'AveExpr',
        'z_score': 't',
        'p_value': 'P.Value',
        'fdr': 'adj.P.Val'
    })

    # filter sites without logFC
    data_df = data_df[data_df['logFC'].notna()]

    return data_df

def get_tuechler_tgfb_data() -> pd.DataFrame:
    """Load Tuechler et al. TGFβ phosphoproteomic contrasts."""
    f_path = os.path.join(config.CACHE_DIR, "phosphodata/tuechler_tgfb/diff_phospho.csv")
    uniprot_to_symbol, symbol_to_uniprot = utils.get_uniprot_symbol_mapping()
    df = pd.read_csv(f_path)
    int_data = df[df['comparison'] == 'groupTGF_0.08h-groupctrl_0.08h'].copy()
    # from feature_id, extract gene and position. Gene is first after splitting by '_', and position is the last element
    int_data['gene'] = int_data['feature_id'].apply(lambda x: x.split('_')[0])
    int_data['position'] = int_data['feature_id'].apply(lambda x: x.split('_')[-1])
    int_data['uniprot'] = int_data['gene'].apply(lambda x: symbol_to_uniprot.get(x, x))

    # divide position by ';' and explode the dataframe
    int_data['position'] = int_data['position'].str.split(';')
    int_data = int_data.explode('position')
    int_data['id'] = int_data['uniprot'] + '_' + int_data['position']

    # per site id, keep only the row with lowest p value
    int_data = int_data[['id', 'logFC', 'AveExpr', 't', 'P.Value', 'adj.P.Val', 'B']].copy()
    int_data = int_data.sort_values('P.Value')
    int_data = int_data.drop_duplicates(subset='id', keep='first')
    int_data['study'] = 'Tuechler et al. 2025 (PDGFRb)'
    int_data['comparison'] = '5min'

    return int_data

def get_all_studies() -> pd.DataFrame:
    """Combine all curated studies into a single annotated DataFrame."""

    egf_data = get_egf_site_data()
    tgf_data = get_tuechler_tgfb_data()
    insulin_data = get_chen_interferon_data()

    # concat all dataframes
    data_df = pd.concat([egf_data, tgf_data, insulin_data], ignore_index=True)
    study_order = config.STUDY_COLORS.keys()
    study_order = list(reversed(study_order))
    data_df['study'] = pd.Categorical(data_df['study'], categories=study_order, ordered=True)
    data_df['stimuli'] = 'EGF'
    data_df.loc[data_df['study'] == 'Tuechler et al. 2025 (PDGFRb)', 'stimuli'] = 'TGFb'
    data_df.loc[data_df['study'] == 'Chen et al. 2025 (HEK293T)', 'stimuli'] = 'IFN'

    data_df['log_pval'] = -np.log10(data_df['P.Value'])
    data_df['symbol_id'] = utils.flex_site_mapping(data_df['id'])
    data_df['is_egf'] = data_df['id'].isin(netgt.get_canonical_egf_sites())

    return data_df

def filter_studies(data_df: pd.DataFrame) -> pd.DataFrame:
    """Select the most responsive comparison per study (by |logFC|)."""
    filtered_data = []
    for study, study_df in data_df.groupby('study', observed = True):
        if len(study_df['comparison'].unique()) > 1:
            avg_logfc = study_df.groupby('comparison')['logFC'].apply(lambda x: np.mean(np.abs(x)))
            best_comparison = avg_logfc.idxmax()
            filtered_data.append(study_df[study_df['comparison'] == best_comparison])
            print(f'Keeping {best_comparison} for study: {study}')
        else:
            filtered_data.append(study_df)
        
    return pd.concat(filtered_data, ignore_index=True)

def prepare_timecourse_df(data_df: pd.DataFrame, int_study: str) -> pd.DataFrame:
    """Return a wide timecourse matrix for the requested study."""
    study_df = data_df[data_df['study'] == int_study].copy()
    sig_sites = study_df[(study_df['adj.P.Val'] <= 0.05) & (study_df['logFC'].abs() >= 1)]['id'].unique()
    study_df = study_df[study_df['id'].isin(sig_sites)]
    study_df['timepoint'] = study_df['comparison'].str.replace('min', '', regex = True).astype(int)
    study_df = study_df[['id', 'logFC', 'timepoint']].pivot(index='id', columns='timepoint', values='logFC')
    study_df = study_df.dropna(axis=0)
    return study_df

def cluster_auto(
    df: pd.DataFrame,
    k_range: Iterable[int] = range(2, 13),
    random_state: int = 42,
) -> Tuple[List[Tuple[int, float]], dict[str, int]]:
    """Run KMeans across ``k_range`` and return scores plus hard assignments."""
    X = df.dropna(axis=0).to_numpy()
    kept_index = df.dropna(axis=0).index
    best_score, best_labels = -1, None
    all_scores = []
    for k in tqdm(k_range):
        labels = KMeans(n_clusters=k, n_init="auto", random_state=random_state).fit_predict(X)
        score = silhouette_score(X, labels)
        all_scores.append((k, score))
        if score > best_score:
            best_score, best_labels = score, labels
    print(f"Best k: {len(set(best_labels))} (silhouette score: {best_score:.3f})")
    out_labels = dict(zip(kept_index, map(int, best_labels)))
    return all_scores, out_labels

def get_timecourse_clusters(data_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster the EGF timecourse studies and return silhouettes plus assignments."""
    cluster_results = []
    sil_results = []
    for s in ['This study (HEK293T)', 'This study (HEK293F)', 'This study (HEK293F TR)']:
        timecourse_df = prepare_timecourse_df(data_df, s)
        print(timecourse_df.shape)
        sil_scores, clusters = cluster_auto(timecourse_df)
        sil_df = pd.DataFrame(sil_scores, columns=['k', 'silhouette_score'])
        sil_df['study'] = s
        sil_results.append(sil_df)
        timecourse_df['cluster'] = timecourse_df.index.map(clusters)
        # pivot longer and add study and cluster
        timecourse_df = timecourse_df.reset_index().melt(id_vars=['id', 'cluster'], var_name='timepoint', value_name='logFC')
        timecourse_df['study'] = s
        cluster_results.append(timecourse_df)
    sil_df = pd.concat(sil_results, ignore_index=True)
    cluster_df = pd.concat(cluster_results, ignore_index=True)
    return sil_df, cluster_df

def get_egf_timecourse_df() -> pd.DataFrame:
    """Return the EGF TR study in wide format (logFC per timepoint)."""

    all_data = get_all_studies()
    filt_data = all_data[all_data['study'] == 'This study (HEK293F TR)'].copy()
    filt_data = filt_data[['id', 'logFC', 'comparison']].pivot(index='id', columns='comparison', values='logFC')
    filt_data = filt_data.dropna(axis=0)
    return filt_data

def get_lun_gt_interactions(threshold: float = 0.13) -> List[str]:
    """Delegate to ``netgt.get_lun_gt_interactions`` for backwards compatibility."""

    return netgt.get_lun_gt_interactions(threshold=threshold)

def get_canonical_egf_interactions() -> List[str]:
    """Delegate to ``netgt.get_canonical_egf_interactions`` for backwards compatibility."""

    return netgt.get_canonical_egf_interactions()
