"""Ground-truth graph utilities for EGFR-centric benchmarking data."""

from __future__ import annotations

import os
import pickle
import re
from typing import Dict, Iterable, List

import networkx as nx
import numpy as np
import pandas as pd
from Bio import SeqIO

import phosphonetworks as pp

def get_signor_egf_df(mode: str = 'protein') -> pd.DataFrame:
    """Return SIGNOR EGFR interactions at protein or site resolution."""

    signor_dir = os.path.join(pp.config.CACHE_DIR, 'signor')
    signor_egfr = pd.read_csv(os.path.join(signor_dir, 'SIGNOR_egfr.tsv'), sep='\t')
    signor_pfs = pd.read_csv(os.path.join(signor_dir, 'SIGNOR_PF.csv'), sep=';')
    signor_cps = pd.read_csv(os.path.join(signor_dir, 'SIGNOR_complexes.csv'), sep=';')
    signor_phs = pd.read_csv(os.path.join(signor_dir, 'SIGNOR-PH.csv'), sep=';')

    # prepare mapping between SIGNOR IDs and entity names
    signor_pfs_dict = dict(zip(signor_pfs['SIGNOR ID'], signor_pfs['LIST OF ENTITIES']))
    signor_cps_dict = dict(zip(signor_cps['SIGNOR ID'], signor_cps['LIST OF ENTITIES']))
    signor_phs_dict = dict(zip(signor_phs['SIGNOR ID'], signor_phs['PHENOTYPE NAME']))
    signor_dict = {**signor_pfs_dict, **signor_cps_dict, **signor_phs_dict}

    # create large Df
    large_egf = signor_egfr.copy()
    large_egf['IDA_expl'] = [signor_dict[x] if x in signor_dict.keys() else x for x in large_egf['IDA']]
    large_egf['IDB_expl'] = [signor_dict[x] if x in signor_dict.keys() else x for x in large_egf['IDB']]
    large_egf['IDA_expl'] = large_egf['IDA_expl'].str.split(',').apply(lambda x: [y.strip() for y in x])
    large_egf['IDB_expl'] = large_egf['IDB_expl'].str.split(',').apply(lambda x: [y.strip() for y in x])
    large_egf = large_egf.explode('IDA_expl')
    large_egf = large_egf.explode('IDB_expl')

    if mode == 'protein':
        outdf = large_egf[['IDA_expl', 'IDB_expl', 'EFFECT']].rename(columns={'IDA_expl': 'source', 'IDB_expl': 'target', 'EFFECT': 'effect'})
        outdf['source_symbol'] = pp.utils.flex_site_mapping(outdf['source'])
        outdf['target_symbol'] = pp.utils.flex_site_mapping(outdf['target'])
        return outdf

    if mode == 'site':

        egf_net = large_egf.copy()
        egf_site_level = egf_net.dropna(subset = ['RESIDUE'])
        egf_site_level = egf_site_level[['IDA_expl', 'IDB_expl', 'RESIDUE', 'EFFECT']]
        replacement = {'Ser': 'S', 'Thr': 'T', 'Tyr': 'Y'}
        egf_site_level['site_id'] = egf_site_level['RESIDUE'].replace(replacement, regex=True)

        # drop rows with None site_id
        egf_site_level = egf_site_level.dropna(subset = ['site_id'])
        egf_site_level['source'] = egf_site_level['IDA_expl']
        egf_site_level['target'] = egf_site_level['IDB_expl'] + '_' + egf_site_level['site_id']
        egf_site_level= egf_site_level[['source', 'target', 'EFFECT']].rename(columns={'EFFECT': 'effect'})


        integrator_df = pd.DataFrame({
            'source': egf_site_level['target'],
            'target': egf_site_level['target'].str.replace(r'_.*', '', regex=True),
            'effect': 'integrator'
        })
        egf_site_level = pd.concat([egf_site_level, integrator_df], ignore_index=True)

        # add the source_symbol and target_symbol columns
        egf_site_level['source_symbol'] = pp.utils.flex_site_mapping(egf_site_level['source'])
        egf_site_level['target_symbol'] = pp.utils.flex_site_mapping(egf_site_level['target'])

        return egf_site_level
    
def get_canonical_egf_sites() -> List[str]:
    """Return curated EGFR pathway sites present in SIGNOR."""
    site_df = get_signor_egf_df(mode='site')
    source_sites= set([i for i in site_df['source'] if '_' in i])
    target_sites= set([i for i in site_df['target'] if '_' in i])
    out_sites = list(source_sites.union(target_sites))
    return out_sites

def get_canonical_egf_proteins() -> List[str]:
    """Return the set of EGFR pathway proteins captured in SIGNOR."""
    protein_df = get_signor_egf_df(mode='protein')
    source_proteins = set(protein_df['source'])
    target_proteins = set(protein_df['target'])
    out_proteins = list(source_proteins.union(target_proteins))
    return out_proteins

def get_canonical_egf_network(mode: str = 'protein', symbol: bool = True) -> nx.DiGraph:
    """Build a directed EGFR network with optional symbol remapping."""
    if mode == 'protein':
        df = get_signor_egf_df(mode='protein')
    elif mode == 'site':
        df = get_signor_egf_df(mode='site')
    else:
        raise ValueError("Mode must be either 'protein' or 'site'")
    
    # in df, replace CHEBI:16618 with 'PIP3'
    df['source'] = df['source'].replace('CHEBI:16618', 'PIP3')
    df['target'] = df['target'].replace('CHEBI:16618', 'PIP3')
    df['source_symbol'] = df['source_symbol'].replace('CHEBI:16618', 'PIP3')
    df['target_symbol'] = df['target_symbol'].replace('CHEBI:16618', 'PIP3')
    if symbol:
        graph = nx.from_pandas_edgelist(
            df,
            source='source_symbol',
            target='target_symbol',
            edge_attr='effect',
            create_using=nx.DiGraph,
        )
    else:
        graph = nx.from_pandas_edgelist(
            df,
            source='source',
            target='target',
            edge_attr='effect',
            create_using=nx.DiGraph,
        )

    return graph

def get_ochoa_functional_scores() -> Dict[str, float]:
    """Return Ochoa et al. functional scores keyed by UniProt site IDs."""

    int_f = os.path.join(pp.config.CACHE_DIR, 'functional_scores', '41587_2019_344_MOESM5_ESM.xlsx')
    int_df = pd.read_excel(int_f, sheet_name='functional_score')

    fasta_file = os.path.join(pp.config.CACHE_DIR, 'uniprot', 'UP000005640_9606.fasta')
    seq_dict = SeqIO.to_dict(SeqIO.parse(fasta_file, 'fasta'))
    seq_dict = {re.search(r'\|(.+?)\|', k).group(1): v for k, v in seq_dict.items()}

    # iterate over rows of int_df, and retrieve the position in the sequence
    residues = []
    for i, row in int_df.iterrows():
        prot = row['uniprot']
        pos = row['position']
        if prot in seq_dict and pos <= len(seq_dict[prot]):
            residues.append(seq_dict[prot][pos - 1])
        else:
            residues.append(None)
    int_df['residue'] = residues

    # discard all residues that are not S,T,Y and remove None
    int_df = int_df[int_df['residue'].isin(['S', 'T', 'Y'])]
    int_df = int_df.dropna(subset=['residue'])
    int_df['site_id'] = int_df['uniprot'] + '_' +  int_df['residue'] + int_df['position'].astype(str)
    out_dict = int_df.set_index('site_id')['functional_score'].to_dict()

    return out_dict

def get_lun_diffdata() -> pd.DataFrame:
    """Process Lun et al. kinase overexpression screen into tidy differences."""

    # Step 1 — Define paths and load raw data (metadata, antibodies, assay data)
    base_dir = os.path.join(pp.config.CACHE_DIR, 'egf_kinase_overexpression')
    meta = pd.read_excel(base_dir + '/1-s2.0-S1097276519303132-mmc2.xlsx')
    ab_info = pd.read_excel(base_dir + '/ab_manually_annotated.xlsx')
    data = pd.read_excel(base_dir + '/1-s2.0-S1097276519303132-mmc5.xlsx')
    fasta_file = os.path.join(pp.config.CACHE_DIR, 'uniprot', 'UP000005640_9606.fasta')
    proteome = SeqIO.to_dict(SeqIO.parse(fasta_file, 'fasta'))
    proteome = {re.search(r'\|(.+?)\|', k).group(1): v for k, v in proteome.items()}

    # Step 2 — Normalize antibody annotation fields and expand multi-site entries
    ab_info['Antigen'] = ab_info['Antigen'].str.strip()
    ab_info['UniProt Entry'] = ab_info['UniProt Entry'].str.strip()
    ab_info = ab_info.rename(columns={'Phosphosite (Short Name)': 'site_id'})
    ab_info['site_id'] = ab_info['site_id'].str.split('/')
    ab_info = ab_info.explode('site_id')

    # Step 3 — Validate phosphosite positions against the proteome
    for i, row in ab_info.iterrows():
        site_id = row['site_id']
        if pd.isna(site_id):
            continue
        uniprot = row['UniProt Entry']
        residue = site_id[0]
        position = int(site_id[1:])
        try:
            sequence = proteome[uniprot].seq
            int_residue = sequence[position-1]
            if int_residue != residue:
                ab_info.loc[i, 'site_id'] = np.nan
        except (KeyError, IndexError):
            ab_info.loc[i, 'site_id'] = np.nan

    # Step 4 — Append validated site to UniProt to disambiguate targets
    newuniprot = []
    for i, row in ab_info.iterrows():
        if not pd.isna(row['site_id']):
            newuniprot.append(row['UniProt Entry'] + '_' + row['site_id'])
        else:
            newuniprot.append(row['UniProt Entry'])
    ab_info['UniProt Entry'] = newuniprot
    ab_info = ab_info[['Antigen', 'UniProt Entry']].drop_duplicates().dropna()

    # Step 5 — Map markers in assay data to UniProt and clean columns
    marker_to_uniprot = meta[['marker', 'Entry']].dropna().set_index('marker').to_dict()['Entry']
    data['kinase_uniprot'] = data['marker'].map(marker_to_uniprot)
    data['kinase_uniprot'] = data['kinase_uniprot'].fillna(data['marker'])
    data = data.drop(columns=['Unnamed: 0', 'library_t', 'marker'])

    # Step 6 — Threshold signals vs. controls (per readout column)
    pdata = data.copy()
    control_re = re.compile(r'FLAG-GFP|untransfected')
    pdata['group'] = np.where(pdata['kinase_uniprot'].str.contains(control_re), 'control', 'case')

    for col in pdata.columns:
        if col in ['kinase_uniprot', 'group']:
            continue
        max_control = pdata[pdata['group'] == 'control'][col].max()
        pdata[col] = np.where(pdata[col] >= max_control, pdata[col], 0)

    pdata.columns = pdata.columns.str.strip()
    pdata = pdata[pdata['group'] == 'case']
    pdata

    # Step 7 — Compute timepoint differences per antibody (10 min – 0 min)
    ab_list = list(ab_info['Antigen'].unique())
    diffdata = []
    for key in ab_list:
        if key + '_0' in pdata.columns and key + '_10' in pdata.columns:
            diffdata.append(pdata[key + '_10'].values - pdata[key + '_0'].values)
        else:
            print(key)
    diffdata = pd.DataFrame(diffdata).T
    diffdata.columns = ab_list

    # Step 8 — Tidy long-form output and attach UniProt target annotations
    diffdata['kinase_uniprot'] = pdata['kinase_uniprot'].values
    diffdata = diffdata.melt(id_vars='kinase_uniprot', var_name='target_ab', value_name='diff')
    diffdata = diffdata.merge(ab_info, left_on='target_ab', right_on='Antigen', how='left')
    diffdata = diffdata.rename(columns={'UniProt Entry': 'target_site', 'kinase_uniprot': 'ko_uniprot'})
    diffdata = diffdata[['ko_uniprot', 'target_site', 'target_ab', 'diff']]
    diffdata['target_uniprot'] = diffdata['target_site'].str.replace(r'_.*', '', regex=True)

    diffdata['ko_symbol'] = pp.utils.flex_site_mapping(diffdata['ko_uniprot'])
    diffdata['target_symbol'] = pp.utils.flex_site_mapping(diffdata['target_site'])

    return diffdata

def get_lun_gt_interactions(threshold: float = 0.13) -> List[str]:
    """Return Lun et al. kinase → kinase hits above ``threshold``."""
    lun_gt_ints = (
        get_lun_diffdata()
        .query(f"diff >= {threshold}")
        .assign(target_uniprot=lambda df: df['target_site'].str.replace('_.*', '', regex=True))
        .query("ko_uniprot != target_uniprot")
        .loc[:, ['ko_uniprot', 'target_uniprot']]
        .drop_duplicates()
        .rename(columns={'ko_uniprot': 'source', 'target_uniprot': 'target'})
    )
    # restrict to kinases in kinsub
    kinases = pp.kinsub.get_kinase_uniprot_list()
    lun_gt_ints = lun_gt_ints[lun_gt_ints['source'].isin(kinases) & lun_gt_ints['target'].isin(kinases)].copy()
    lun_gt_ints = list(set([ x['source'] + '-->' + x['target'] for _, x in lun_gt_ints.iterrows()]))
    return lun_gt_ints

def get_canonical_egf_interactions() -> List[str]:
    """Return kinase → kinase strings drawn from the curated SIGNOR set."""
    canonical_egf_ints = (
        get_signor_egf_df()
        .loc[:, ['source', 'target']]
        .drop_duplicates()
    )
    # restrict to kinases in kinsub
    kinases = pp.kinsub.get_kinase_uniprot_list()
    canonical_egf_ints = canonical_egf_ints[canonical_egf_ints['source'].isin(kinases) & canonical_egf_ints['target'].isin(kinases)].copy()
    canonical_egf_ints = list(set([ x['source'] + '-->' + x['target'] for _, x in canonical_egf_ints.iterrows()]))
    return canonical_egf_ints

def get_wide_hek293f_tr() -> pd.DataFrame:
    """Return the HEK293F TR study as a wide logFC matrix."""
    all_data = pp.phosphodata.get_all_studies()
    filt_data = all_data[all_data['study'] == 'This study (HEK293F TR)'].copy()
    filt_data['id_prot'] = filt_data['id'].str.replace('_.*', '', regex=True)
    filt_data = filt_data[filt_data['id_prot'].isin(pp.kinsub.get_kinase_uniprot_list())]
    filt_data = filt_data[['id', 'logFC', 'comparison']].pivot(index='id', columns='comparison', values='logFC')
    filt_data = filt_data.dropna(axis=0)
    return filt_data

def corr_func_sites(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Correlate functional sites with high Ochoa scores."""
    func_score = get_ochoa_functional_scores()
    filt_func_score = {k: v for k, v in func_score.items() if v >= threshold}
    func_sites = list(filt_func_score.keys())
    func_data = df.loc[df.index.isin(func_sites)].copy()
    corr_data = pp.utils.corr_from_df(func_data.T, min_pairs = 5, method = 'pearson')
    corr_data = corr_data.rename(columns={'study_a': 'site_a', 'study_b': 'site_b'})
    corr_data['prot_a'] = corr_data['site_a'].str.replace('_.*', '', regex=True)
    corr_data['prot_b'] = corr_data['site_b'].str.replace('_.*', '', regex=True)
    corr_data = corr_data[(corr_data['prot_a'] != corr_data['prot_b'])].copy()
    return corr_data

def get_hek293tr_interactions(
    funcscore_threshold: float = 0.5,
    pearson_cutoff: float = 0.9,
) -> List[str]:
    """Derive putative kinase interactions from HEK293F TR correlations."""
    wide_hek = get_wide_hek293f_tr()
    filt_corr = corr_func_sites(wide_hek, threshold=funcscore_threshold)
    out_ints = (
        filt_corr[['prot_a', 'prot_b', 'pearson_r']]
        .drop_duplicates(subset=['prot_a', 'prot_b'])
        .query(f'pearson_r >= {pearson_cutoff}')
    )
    true_ints_a = list(set([ x['prot_a'] + '-->' + x['prot_b'] for _, x in out_ints.iterrows()]))
    true_ints_b = list(set([ x['prot_b'] + '-->' + x['prot_a'] for _, x in out_ints.iterrows()]))
    true_ints = list(set(true_ints_a + true_ints_b))
    return true_ints

def prepare_all_gt_ints(
    cache_file: str = os.path.join(
        pp.config.CACHE_DIR, 'intermediate_files', 'gt_interactions.pkl'
    )
) -> Dict[str, List[str]]:
    """Assemble and optionally cache all ground-truth interaction collections."""
    # if exists, return cached
    if os.path.exists(cache_file):
        print(f"GT ints at {cache_file} already exist")
        with open(cache_file, 'rb') as f:
            gt_int_dict = pickle.load(f)
        return gt_int_dict
    # else compute and save
    print("Preparing GT interactions")
    gt_int_dict = {
        'signor': get_canonical_egf_interactions(),
        'lun_lenient': get_lun_gt_interactions(threshold=0.08),
        'lun_moderate': get_lun_gt_interactions(threshold=0.13),
        'lun_strict': get_lun_gt_interactions(threshold=0.18),
        'hek293_lenient': get_hek293tr_interactions(pearson_cutoff=0.7),
        'hek293_moderate': get_hek293tr_interactions(pearson_cutoff=0.8),
        'hek293_strict': get_hek293tr_interactions(pearson_cutoff=0.9)
    }
    # print length of each
    for k, v in gt_int_dict.items():
        print(f"{k}: {len(v)} interactions")

    print(f"Saving GT ints to {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(gt_int_dict, f)
    return gt_int_dict
