"""Kinase–substrate resource loaders and helper utilities."""

from __future__ import annotations

import hashlib
import os
import pickle
import subprocess
import tempfile
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import phosphonetworks as pp


def get_kinase_info_df(sequences: bool = False) -> pd.DataFrame:
    """Return the Coral kinase annotation table, optionally with sequence strings."""

    coral_label_file = os.path.join(
        pp.config.DATA_DIR, 
        "kinsub/coral_kinmaplabels.txt"
    )
    dual_spec_kins = [
        'CLK1','CLK2','CLK3','CLK4','DYRK1A','DYRK1B','DYRK2',
        'DYRK3','DYRK4','MAP2K1','MAP2K2','MAP2K3','MAP2K4',
        'MAP2K5','MAP2K6','MAP2K7','TESK1','TESK2','SgK496','TTK'
    ]
    # superfamily is Tyr if TK, Dual if in dual_spec_kins, else SerThr
    kinase_info = pd.read_csv(coral_label_file, sep = '\t')

    # superfamily is by default SerThr
    kinase_info['superfamily'] = 'SerThr'
    kinase_info.loc[kinase_info['Group'] == 'TK', 'superfamily'] = 'Tyr'
    ds_index = kinase_info['ID'].isin(dual_spec_kins)
    kinase_info.loc[ds_index, 'superfamily'] = 'Dual'
    kinase_info = kinase_info.drop_duplicates(subset='UniprotID')

    if sequences:
        sequences_file = os.path.join(
            pp.config.DATA_DIR, "uniprot/UP000005640_9606.fasta"
        )
        # read the sequences from the fasta file
        sequences = {}
        for record in SeqIO.parse(sequences_file, "fasta"):
            sequences[record.id.split('|')[1]] = str(record.seq)
        kinase_info['sequence'] = kinase_info['UniprotID'].apply(
            lambda x: sequences.get(x, '')
        )

    return kinase_info

def get_kinase_uniprot_list() -> List[str]:
    """Return UniProt accessions for all kinases in the Coral mapping."""

    kinase_info = get_kinase_info_df()
    return kinase_info['UniprotID'].tolist()

def get_omnipath_kinsub() -> pd.DataFrame:
    """Load and filter OmniPath enzyme–substrate annotations."""

    omnipath_file = os.path.join(
        pp.config.DATA_DIR, "kinsub/omnipath_enzsub.csv.gz"
    )

    kinase_info = get_kinase_info_df()

    omnipath_pkn = pd.read_csv(omnipath_file)
    omnipath_pkn = omnipath_pkn[omnipath_pkn['n_references'] >= 1]
    omnipath_pkn['id'] = omnipath_pkn['source'] + '-->' + omnipath_pkn['target']
    omnipath_pkn = omnipath_pkn.drop_duplicates(subset='id')
    omnipath_pkn = omnipath_pkn[
        omnipath_pkn['source'].isin(kinase_info['UniprotID'])
        ]
    omnipath_pkn = omnipath_pkn.merge(kinase_info[['UniprotID', 'superfamily']], left_on='source', right_on='UniprotID', how = 'left')
    mask = (
        (omnipath_pkn['superfamily'] == 'Tyr') & (omnipath_pkn['residue_type'] != 'Y') |
        (omnipath_pkn['superfamily'] == 'SerThr') & (~omnipath_pkn['residue_type'].isin(['S', 'T'])) 
    )
    omnipath_pkn = omnipath_pkn[~mask].copy()
    omnipath_pkn['target_protein'] = omnipath_pkn['target'].apply(lambda x: x.split('_')[0])
    omnipath_pkn = omnipath_pkn[['source', 'target', 'n_references']].rename(columns={'n_references': 'score'})
    omnipath_pkn['score'] = omnipath_pkn['score'].astype(float)
    omnipath_pkn['score'] = omnipath_pkn['score'] / omnipath_pkn['score'].max()

    # group by source and target and keep only max score
    omnipath_pkn = omnipath_pkn.groupby(['source', 'target'], as_index=False).agg({'score': 'max'})
    
    return omnipath_pkn

def get_st_kinase_library() -> pd.DataFrame:
    """Return the serine/threonine kinase library in long form."""

    data_f = os.path.join(pp.config.DATA_DIR, "kinsub/st_kinase_library_data.csv.gz")
    mapping_f = os.path.join(pp.config.DATA_DIR, "kinsub/st_kinase_library_mapping.csv") 

    data = pd.read_csv(data_f)
    mapping = pd.read_csv(mapping_f)
    kinase_cols = [i for i in data.columns if '_percentile' in i and 'median_' not in i]
    int_columns = ['Uniprot Primary Accession', 'Phosphosite', 'median_percentile', 'promiscuity_index'] + kinase_cols
    filt_data = data[int_columns]
    melted_data = filt_data.melt(id_vars=['Uniprot Primary Accession', 'Phosphosite', 'median_percentile', 'promiscuity_index'], var_name='kinase', value_name='percentile')
    melted_data['kinase'] = melted_data['kinase'].str.replace('_percentile', '')
    melted_data = melted_data.merge(
        mapping[['Matrix_name', 'Uniprot id']],
        left_on='kinase',
        right_on='Matrix_name',
        how='left'
    )
    melted_data['target'] = melted_data['Uniprot Primary Accession'] + '_' + melted_data['Phosphosite']
    melted_data['source'] = melted_data['Uniprot id']
    melted_data = melted_data[['source', 'target', 'percentile']].rename(columns={'percentile': 'score'})
    st_long = melted_data.copy().drop_duplicates()

    return st_long

def get_tyr_kinase_library() -> pd.DataFrame:
    """Return the tyrosine kinase library in long form."""

    data_f = os.path.join(pp.config.DATA_DIR, "kinsub/tyr_kinase_library_data.csv.gz")
    mapping_f = os.path.join(pp.config.DATA_DIR, "kinsub/tyr_kinase_library_mapping.csv")

    data = pd.read_csv(data_f)
    mapping = pd.read_csv(mapping_f)

    kinase_cols = [i for i in data.columns if '_percentile' in i and 'median_' not in i]
    int_columns = ['Uniprot Primary Accession', 'Phosphosite', 'median_percentile', 'promiscuity_index'] + kinase_cols
    kinase_cols = [i for i in data.columns if '_percentile' in i and 'median_' not in i]
    int_columns = ['Uniprot Primary Accession', 'Phosphosite', 'median_percentile', 'promiscuity_index'] + kinase_cols
    filt_data = data[int_columns]
    melted_data = filt_data.melt(id_vars=['Uniprot Primary Accession', 'Phosphosite', 'median_percentile', 'promiscuity_index'], var_name='kinase', value_name='percentile')
    melted_data['kinase'] = melted_data['kinase'].str.replace('_percentile', '')

    melted_data = melted_data.merge(
        mapping[['MATRIX_NAME', 'UNIPROT_ID']],
        left_on='kinase',
        right_on='MATRIX_NAME',
        how='left'
    )
    melted_data['target'] = melted_data['Uniprot Primary Accession'] + '_' + melted_data['Phosphosite']
    melted_data['source'] = melted_data['UNIPROT_ID']
    melted_data = melted_data[['source', 'target', 'percentile']].rename(columns={'percentile': 'score'})
    melted_data = melted_data.dropna(subset=['source', 'target', 'score']).copy()
    tyr_long = melted_data.copy().drop_duplicates()

    return tyr_long

def get_kinase_library() -> pd.DataFrame:
    """Combine serine/threonine and tyrosine kinase libraries."""

    st_long = get_st_kinase_library()
    tyr_long = get_tyr_kinase_library()

    kinase_library = pd.concat([st_long, tyr_long], ignore_index=True)
    kinase_library['score'] = kinase_library['score'].astype(float)

    kinase_library = kinase_library.drop_duplicates(subset=['source', 'target'])
    kinase_library['score'] = kinase_library['score'] / 100.0

    # group by source and target and keep only max score
    kinase_library = kinase_library.groupby(['source', 'target'], as_index=False).agg({'score': 'max'})

    return kinase_library


def get_phosformer() -> pd.DataFrame:
    """Load the Phosformer kinase–substrate predictions."""

    data_f = os.path.join(pp.config.DATA_DIR, "kinsub/phosformer.csv.gz")
    phosformer = pd.read_csv(data_f)

    # group by source and target and keep only max score
    phosformer = phosformer.groupby(['source', 'target'], as_index=False).agg({'score': 'max'})

    return phosformer


def filter_by_score(
    input_df: pd.DataFrame,
    threshold: float,
    rescale_score: bool = True,
) -> pd.DataFrame:
    """Filter interactions by score and optionally rescale to [0, 1]."""

    filt_df = input_df[input_df['score'] >= threshold].copy()
    if rescale_score:
        filt_df['score'] = filt_df['score'] / filt_df['score'].max()
    return filt_df

def get_combined_kinsub(
    cutoffs: Optional[Dict[str, Tuple[float, float]]] = None
) -> pd.DataFrame:
    """Assemble kinase resources (OmniPath, kinase libraries, Phosformer)."""

    cache_dir = pp.config.DATA_DIR
    kinase_info = get_kinase_info_df()

    if cutoffs is None:
        cutoffs = {
            'Strict': (0.99, 0.8),
            'Moderate': (0.95, 0.65),
            'Lenient': (0.90, 0.5)
        }

    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Create a hash of the cutoffs dictionary
    cutoff_str = str(sorted(cutoffs.items()))
    cutoff_hash = hashlib.md5(cutoff_str.encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"intermediate_files/combined_kinsub_{cutoff_hash}.pkl")

    # Check for cached file
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # Load data and filter
    kin_info = get_kinase_info_df()
    omnipath = get_omnipath_kinsub()
    kinlib = get_kinase_library()
    phosformer = get_phosformer()

    cutoff_dfs = []
    for cutoff, (kinlib_cutoff, phosformer_cutoff) in cutoffs.items():
        kinlib_filt = filter_by_score(kinlib, kinlib_cutoff)
        kinlib_filt['resource'] = 'kinlib-' + cutoff
        cutoff_dfs.append(kinlib_filt)

        phosformer_filt = filter_by_score(phosformer, phosformer_cutoff)
        phosformer_filt['resource'] = 'phosformer-' + cutoff
        cutoff_dfs.append(phosformer_filt)

        combined = pd.concat([omnipath, kinlib_filt, phosformer_filt], ignore_index=True)
        # incombined, group by source and target and keep only max score
        combined = combined.groupby(['source', 'target'], as_index=False).agg({'score': 'max'})

        combined['resource'] = 'combined-' + cutoff
        cutoff_dfs.append(combined)

    out_df = pd.concat(cutoff_dfs, ignore_index=True)
    omnipath['resource'] = 'literature'
    out_df = pd.concat([out_df, omnipath], ignore_index=True)

    out_df = out_df[out_df['source'].isin(kin_info['UniprotID'])].copy()

    # keep only source within kinase_info
    out_df = out_df[out_df['source'].isin(kinase_info['UniprotID'])].copy()

    # Cache the result
    with open(cache_path, "wb") as f:
        pickle.dump(out_df, f)

    return out_df


def get_one_cutoff_combined_kinsub(
    cutoff: str = 'Moderate', **kwargs
) -> pd.DataFrame:
    """Retrieve combined resources for a single cutoff label plus literature."""

    kinsub = get_combined_kinsub(**kwargs)
    kinsub_selected = kinsub[kinsub['resource'].str.contains('literature|' + cutoff, case=False)].copy()
    kinsub_selected['resource'] = kinsub_selected['resource'].str.replace('-.*', '', regex=True)
    return kinsub_selected

def write_fasta(sequences_dict: Dict[str, str], fasta_path: str) -> None:
    """Write a temporary FASTA file from a mapping of ID to sequence."""

    records = [SeqRecord(Seq(seq), id=seq_id, description="") for seq_id, seq in sequences_dict.items()]
    with open(fasta_path, "w") as handle:
        SeqIO.write(records, handle, "fasta")

def parse_mmseqs_tsv(tsv_path: str, sequences_dict: Dict[str, str]) -> pd.DataFrame:
    """Convert MMseqs pairwise identity output into a symmetric matrix."""

    df = pd.read_csv(tsv_path, sep='\t', header=None, names=['query', 'target', 'pident'])

    # Pivot to full matrix
    ids = list(sequences_dict.keys())
    matrix = pd.DataFrame(0.0, index=ids, columns=ids)

    for _, row in df.iterrows():
        q, t, pid = row['query'], row['target'], row['pident']
        matrix.at[q, t] = pid
        matrix.at[t, q] = pid  # ensure symmetry

    # Fill diagonal
    for i in ids:
        matrix.at[i, i] = 100.0

    return matrix


def run_mmseqs_pipeline() -> pd.DataFrame:
    """Compute sequence identity matrix for kinases, caching results."""

    # check if cache exists
    kin_info_df = get_kinase_info_df(sequences=True)
    cache_dir = pp.config.DATA_DIR
    cache_file = os.path.join(cache_dir, "intermediate_files/identity_seq_matrix.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    with tempfile.TemporaryDirectory() as tmpdir:

        fasta_path = os.path.join(tmpdir, "input.fasta")
        db_path = os.path.join(tmpdir, "seqdb")
        result_path = os.path.join(tmpdir, "result")
        tmp_mmseqs = os.path.join(tmpdir, "tmp")
        tsv_path = os.path.join(tmpdir, "matches.tsv")


        kin_info_df = kin_info_df[kin_info_df['sequence'] != ''].copy()
        kin_sequences = kin_info_df.set_index('UniprotID')['sequence'].to_dict()

        write_fasta(kin_sequences, fasta_path)

        # Step 1: create DB
        subprocess.run(["mmseqs", "createdb", fasta_path, db_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Step 2: run search (self vs self)
        subprocess.run(["mmseqs", "search", db_path, db_path, result_path, tmp_mmseqs, "-a"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Step 3: convert to tsv
        subprocess.run(["mmseqs", "convertalis", db_path, db_path, result_path, tsv_path, "--format-output", "query,target,pident"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        result = parse_mmseqs_tsv(tsv_path, kin_sequences)

        # Cache the result
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)

        return result

def compute_kinase_target_similarity(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Simpson overlap between kinase target sets."""

    binary_matrix = pd.crosstab(df['source'], df['target']).astype(bool).astype(int)

    # Step 2: Compute intersection matrix (|A ∩ B|)
    intersection = binary_matrix.values @ binary_matrix.values.T

    # Step 3: Set sizes (|A|, |B|)
    set_sizes = binary_matrix.sum(axis=1).values
    min_set = np.minimum.outer(set_sizes, set_sizes)

    # Step 4: Compute metrics
    simpson = np.divide(intersection, min_set, out=np.zeros_like(intersection, dtype=float), where=min_set!=0)

    # Step 5: Convert to DataFrames with appropriate labels
    index = binary_matrix.index
    simpson_df = pd.DataFrame(simpson, index=index, columns=index)

    return simpson_df


def combined_kinsub_target_similarity(
    kinsub_df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Compute target similarity for each resource with caching."""

    cache_dir = pp.config.DATA_DIR
    cache_file = os.path.join(cache_dir, "intermediate_files/combined_kinsub_target_similarity.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    
    result = {}
    for int_resource in kinsub_df['resource'].unique():
        print(f"Computing similarity for resource: {int_resource}")
        resource_df = kinsub_df[kinsub_df['resource'] == int_resource].copy()
        simpson_df = compute_kinase_target_similarity(resource_df)
        result[int_resource] = simpson_df

    # Cache the result
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)

    return result


def get_target_similarity_df(kinsub_df: pd.DataFrame) -> pd.DataFrame:
    """Combine sequence and target overlap similarities into a long table."""

    seq_similarity = pp.utils.pairwise_matrix_to_pairwise_df(run_mmseqs_pipeline())

    seq_similarity['id'] = seq_similarity.apply(
        lambda row: '-'.join(sorted([row['kinase_a'], row['kinase_b']])), axis=1
    )

    target_similarity = pp.kinsub.combined_kinsub_target_similarity(kinsub_df)
    target_similarity_df = {k: pp.utils.pairwise_matrix_to_pairwise_df(v) for k, v in target_similarity.items()}
    # annotate with key and concat dfs
    target_similarity_df = pd.concat(
        [df.assign(resource=key) for key, df in target_similarity_df.items()],
        ignore_index=True
    )

    # id is kinase_a and kinase_b concatenated in alphabetical order
    target_similarity_df['id'] = target_similarity_df.apply(
        lambda row: '-'.join(sorted([row['kinase_a'], row['kinase_b']])), axis=1
    )

    outdf = target_similarity_df.merge(
        seq_similarity[['id', 'score']],
        on='id', how='left', suffixes=('', '_seq')
    ).drop(columns=['id'])
    
    return outdf


def get_kk_pkn(n_top: int = 5) -> pd.DataFrame:
    """Derive a kinase–kinase PKN using literature plus top ``n_top`` edges per source."""

    all_kinases = get_kinase_uniprot_list()
    kinsub = get_one_cutoff_combined_kinsub(cutoff='Moderate')

    kinsub_filt = (
        kinsub
        .assign(target_uniprot=lambda df: df['target'].str.replace('_.*', '', regex=True))
        .query('target_uniprot in @all_kinases')
        .query('source in @all_kinases')
    )
    assert kinsub_filt['source'].isin(all_kinases).all()
    assert kinsub_filt['target_uniprot'].isin(all_kinases).all()

    kinsub_lit = kinsub_filt[kinsub_filt['resource'] == 'literature'].copy()
    non_lit = kinsub_filt[kinsub_filt['resource'] != 'literature'].copy()
    # for non_lit, sort by score and keep n_top per source
    non_lit = (
        non_lit
        .sort_values(['source', 'score'], ascending=[True, False])
        .groupby(['resource', 'source'])
        .head(n_top)
    )
    kinsub_final = pd.concat([kinsub_lit, non_lit], ignore_index=True)
    kinsub_final = kinsub_final[['source', 'target_uniprot', 'resource', 'score']].rename(columns={'target_uniprot': 'target', 'score': 'weight'})
    return kinsub_final
