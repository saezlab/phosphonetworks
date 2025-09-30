"""Plotting utilities for phosphonetworks analyses and figures."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats as stats
from IPython.display import Image
from matplotlib import cm
from matplotlib.colors import Normalize
from mizani.formatters import scientific_format
from plotnine import *  
from sklearn.preprocessing import StandardScaler

import phosphonetworks as pp

def plot_top_kinases(kin_act_df: pd.DataFrame) -> ggplot:
    """Bar plot highlighting the top kinase activities per resource."""
    kin_act_df['abs_activity'] = kin_act_df['activity'].abs()
    kin_act_df['kin_symbol'] = pp.utils.flex_site_mapping(kin_act_df['kinase'])
    top_kinases = kin_act_df.groupby('resource').apply(lambda df: df.nlargest(10, 'abs_activity')).reset_index(drop=True)
    top_kinases['res_kin'] = top_kinases['resource'] + '_' + top_kinases['kin_symbol']
    top_kinases['res_kin'] = pd.Categorical(top_kinases['res_kin'], categories=top_kinases.sort_values('abs_activity', ascending=True)['res_kin'].unique(), ordered=True)
    p = (
        ggplot(top_kinases, aes(x = 'res_kin', y = 'activity', fill = 'activity')) +
        coord_flip() + 
        facet_wrap('~resource', scales = 'free') +
        geom_bar(stat = 'identity', position = 'dodge') +
        theme(axis_text_x = element_text(angle = 45, hjust = 1)) +
        scale_x_discrete(labels = lambda x: [re.sub('.*_', '', i) for i in x]) +
        xlab('Kinase') +
        ylab('Kinase Activity') +
        # scale fill gradient2 
        scale_fill_gradient2(low='blue', mid='white', high='red', midpoint=0, name = 'Activity')
    )
    return p

def plot_net_results_heatmap(res_df: pd.DataFrame) -> Tuple[pd.DataFrame, ggplot]:
    """Visualize network overlaps against ground-truth sets as faceted heatmap."""

    gt_types  = res_df['gt_type'].unique()
    studies   = res_df['study'].unique()
    methods   = res_df['method'].unique()
    resources = res_df['resource'].unique()

    # Cartesian product of all unique values
    all_combos = pd.MultiIndex.from_product(
        [gt_types, studies, methods, resources],
        names=['gt_type', 'study', 'method', 'resource']
    )

    # group by gt_type, study, method, resource and compute mean and std of overlap_with_gt
    toplot = res_df.groupby(['gt_type', 'study', 'method', 'resource']).agg(
        max_overlap=('overlap_with_gt', 'max')
    )

    toplot = toplot.reindex(all_combos, fill_value=0).reset_index()

    # reorder studies using config
    toplot['study'] = pd.Categorical(toplot['study'], categories=list(pp.config.STUDY_COLORS.keys()), ordered=True)
    toplot['resource'] = pd.Categorical(toplot['resource'], categories=list(pp.config.KINSUB_COLORS.keys()), ordered=True)
    method_dict = {
        'mean_net': 'Source/Target\nmean activity',
        'pr_net': 'Source/Target\nPPR score',
        'pcst_net': 'Rooted\nPCST'
    }
    toplot['method'] = toplot['method'].map(method_dict)
    # set order as keys
    toplot['method'] = pd.Categorical(toplot['method'], categories=list(method_dict.values()), ordered=True)
    toplot['gt_type'] = toplot['gt_type'].map(pp.config.GT_LABELS)
    toplot['resource'] = toplot['resource'].map(pp.config.KINSUB_LABELS)
    # set order as label values
    toplot['gt_type'] = pd.Categorical(toplot['gt_type'], categories=list(pp.config.GT_LABELS.values()), ordered=True)
    toplot['resource'] = pd.Categorical(toplot['resource'], categories=list(pp.config.KINSUB_LABELS.values()), ordered=True)


    p = (
        ggplot(toplot, aes(x = 'method', y = 'study', fill = 'max_overlap', label = 'max_overlap')) +
        geom_tile(color = 'black') +
        facet_grid('gt_type~resource', scales='free') +
        geom_text(color = 'black', size = 8) +
        scale_fill_continuous(cmap_name='Reds', name = 'Maximun overlap\n with ground truth\n(pathways up to 250 edges)') +
        theme_bw() +
        theme(axis_text_x = element_text(angle = 45, hjust = 1)) +
        xlab('Method') + ylab('Study') +
        theme(
            legend_position = 'bottom',
            figure_size=(8, 10)
        )
    )
    return toplot, p

def plot_n_top_cutoffs(kinsub: pd.DataFrame) -> Tuple[pd.DataFrame, ggplot]:
    """Plot how network density changes with the top-N cutoff per resource."""

    all_kinases = pp.kinsub.get_kinase_uniprot_list()
    kinsub_filt = (
        kinsub
        .assign(target_uniprot=lambda df: df['target'].str.replace('_.*', '', regex=True))
        .query('target_uniprot in @all_kinases')
        .query('source in @all_kinases')
    )
    res = []
    int_top_n = [250, 100, 50, 10, 5, 3]
    for resource in kinsub_filt['resource'].unique():
        pkn = kinsub_filt[kinsub_filt['resource']==resource][['source', 'target_uniprot', 'score']]
        pkn_g = nx.from_pandas_edgelist(pkn, 'source', 'target_uniprot')
        res.append({
            'resource':resource, 'density': nx.density(pkn_g), 'n_top': 'all'
        })

        if resource != 'literature':
            for top_n in int_top_n:
                # keep only the top N per source based on score
                top_n_df = pkn.sort_values('score', ascending=False).groupby('source').head(top_n)
                pkn_topn = nx.from_pandas_edgelist(top_n_df, 'source', 'target_uniprot')
                res.append({
                    'resource':resource, 'density': nx.density(pkn_topn), 'n_top': str(top_n)
                })
    toplot = pd.DataFrame(res)
    print(toplot)
    # n top in order: All and then decreasing from 10 to 1
    toplot['n_top'] = pd.Categorical(toplot['n_top'], categories=['all'] + [str(i) for i in int_top_n], ordered=True)

    p = (
        ggplot(data = toplot, mapping= aes(x = 'n_top', y = 'density', color = 'resource', group = 'resource')) +
        geom_point(size = 4) + geom_line() +
        scale_color_manual(values=pp.config.KINSUB_COLORS, labels = pp.config.KINSUB_LABELS) + 
        theme_bw() +
        # add horizontal line for literature density value
        geom_hline(yintercept=toplot[toplot['resource']=='literature']['density'].values[0], linetype='dashed', color='grey') +
        labs(x = 'Number of Top Interactions per Kinase', y = 'Network Density')
    )
    return toplot, p

def plot_gt_counts(kk_df: pd.DataFrame, all_ints: Dict[str, List[str]]) -> Tuple[pd.DataFrame, ggplot]:
    """Compare ground-truth coverage counts across resources."""

    toplot = []
    for resource in kk_df['resource'].unique():
        res_df = kk_df[kk_df['resource']==resource]
        res_ints = set(res_df['int_id'])
        for gt_name, gt_ints in all_ints.items():
            n_overlap = len(res_ints.intersection(gt_ints))
            toplot.append({'Resource': resource, 'GroundTruthSet': gt_name, 'n_overlapping_interactions': n_overlap, 'total_gt_interactions': len(gt_ints)})
    toplot = pd.DataFrame(toplot)

    toplot_p1 = toplot[['GroundTruthSet', 'total_gt_interactions']].drop_duplicates().copy()
    p1 = (
        ggplot(toplot_p1, aes(x='GroundTruthSet', y='total_gt_interactions')) +
        geom_bar(stat='identity', fill='lightgrey', color='black') +
        scale_x_discrete(labels = pp.config.GT_LABELS) +
        theme_bw() +
        geom_text(aes(label='total_gt_interactions'), va='bottom') +
        labs(x='Ground Truth Set', y='Total Interactions') +
        theme(figure_size=(7, 3)) +
        theme(legend_position='none') +
        # expand y axis 10%
        expand_limits(y=toplot_p1['total_gt_interactions'].max()*1.2)
    )
    toplot['Resource'] = pd.Categorical(toplot['Resource'], categories=pp.config.KINSUB_LABELS.keys(), ordered=True)
    p2 = (
        ggplot(toplot, aes(x='GroundTruthSet', y='n_overlapping_interactions', fill='Resource', labels = 'n_overlapping_interactions')) +
        geom_bar(stat='identity', position=position_dodge(width=0.9), color='black') +
        geom_text(aes(label='n_overlapping_interactions'),va = 'bottom',position=position_dodge(width=0.9)) +
        scale_x_discrete(labels=pp.config.GT_LABELS) +
        theme_bw() +
        scale_fill_manual(values = pp.config.KINSUB_COLORS, labels = pp.config.KINSUB_LABELS) +
        labs(x='Ground Truth Set', y='Overlapping Interactions') +
        theme(figure_size=(7, 6)) +
        theme(legend_position='bottom') +
        # expand y axis 10%
        expand_limits(y=toplot['n_overlapping_interactions'].max()*1.2)
    )
    p = p1/p2 + theme(figure_size=(10,4.5))

    return toplot, p

def plot_gt_corrs(
    df: pd.DataFrame,
    lun_gt_ints: Iterable[str],
    signor_gt_ints: Iterable[str],
    do_stats: bool = True,
) -> Tuple[pd.DataFrame, ggplot]:
    """Plot correlation distributions stratified by membership in GT sets."""

    toplot = df[['prot_a', 'prot_b', 'pearson_r']].drop_duplicates().copy()
    is_in_gt = []
    for i, row in toplot.iterrows():
        edge_str1 = row['prot_a'] + '-->' + row['prot_b']
        edge_str2 = row['prot_b'] + '-->' + row['prot_a']
        if edge_str1 in signor_gt_ints or edge_str2 in signor_gt_ints:
            is_in_gt.append('CanonicalEGF')
        elif edge_str1 in lun_gt_ints or edge_str2 in lun_gt_ints:
            is_in_gt.append('LunGT')
        else:
            is_in_gt.append('NoGT')

    toplot['is_in_gt'] = is_in_gt
    # replace with proper labels and reorder
    toplot['is_in_gt'] = toplot['is_in_gt'].replace({'NoGT': 'Not in set', 'CanonicalEGF': 'In SIGNOR', 'LunGT': 'In Overexpression'})
    toplot['is_in_gt'] = pd.Categorical(toplot['is_in_gt'], categories=[ 'Not in set', 'In SIGNOR', 'In Overexpression'], ordered=True)

    # anova result
    if do_stats:
        anova_res = stats.f_oneway(
            toplot[toplot['is_in_gt']=='Not in set']['pearson_r'],
            toplot[toplot['is_in_gt']=='In SIGNOR']['pearson_r'],
            toplot[toplot['is_in_gt']=='In Overexpression']['pearson_r']
        )
        print(f"ANOVA result: F={anova_res.statistic}, p={anova_res.pvalue}")

    p = (
        ggplot(toplot, aes(y='pearson_r', x='is_in_gt')) +
        geom_hline(yintercept=0.9, linetype='dashed', color='black', size = 1) +
        geom_hline(yintercept=0.8, linetype='dashed', color='grey', size = 1) +
        geom_hline(yintercept=0.7, linetype='dashed', color='lightgrey', size = 1) +
        geom_violin(fill = 'lightgrey', alpha = 0.5) +
        geom_boxplot(width = 0.1) +
        theme_bw() +
        labs(x = 'Ground Truth Set', y = 'Pearson Correlation') +
        labs(title = 'Kinase functional site correlation') +
        theme(figure_size=(3.5,3)) 
    )
    return toplot, p

def plot_top_corrs(cor_df: pd.DataFrame, wide_df: pd.DataFrame) -> Tuple[pd.DataFrame, ggplot]:
    """Highlight top correlated functional site pairs with supporting heatmap."""
    func_score = pp.netgt.get_ochoa_functional_scores()
    top_corr = cor_df.sort_values('pearson_r', ascending=False).head(4).copy()
    p_list = []
    toplot= []
    for i, row in top_corr.iterrows():
        vec_a = wide_df.loc[row['site_a']].values
        vec_b = wide_df.loc[row['site_b']].values

        # label is site + func score
        label_a = pp.utils.flex_site_mapping([row['site_a']])[0] + " (" + round(func_score[row['site_a']], 2).__str__() + ")"
        label_b = pp.utils.flex_site_mapping([row['site_b']])[0] + " (" + round(func_score[row['site_b']], 2).__str__() + ")"

        df = pd.DataFrame({'A': vec_a, 'B': vec_b, 'label': wide_df.columns})
        df['pair'] = row['site_a'] + ' + ' + row['site_b']
        toplot.append(df)
        p = (
            ggplot(df, aes(x='A', y='B', label = 'label')) +
            geom_point() +
            geom_text(adjust_text = {})+
            theme_bw() +
            labs(x = label_a, y = label_b) 
        )
        p_list.append(p)
    toplot = pd.concat(toplot)
    p = (p_list[0] | p_list[1]) / (p_list[2] | p_list[3])
    return toplot, p

def plot_top_lundata(lun_diffdata: pd.DataFrame) -> Tuple[pd.DataFrame, ggplot]:
    """Summarize Lun et al. differential data per knockout kinase."""
    int_kinases = ['EGFR', 'JAK1', 'SRC', 'PAK1', 'PDPK1', 'BRAF', 'MAPK1', 'AKT1', 'RPS6KA1']
    toplot = lun_diffdata[lun_diffdata['ko_symbol'].isin(int_kinases)].copy()
    toplot = toplot[toplot['target_uniprot'].isin(pp.kinsub.get_kinase_uniprot_list())]

    # ko_symbol is category in int_kinase order
    toplot['ko_symbol'] = pd.Categorical(toplot['ko_symbol'], categories=int_kinases, ordered=True)
    toplot_top = toplot.sort_values('diff', ascending=False).groupby('ko_symbol').head(1)
    p = (
        ggplot(toplot, aes(x='ko_symbol', y='diff')) +

        geom_hline(yintercept=0.18, linetype='dashed', color='black', size = 1) +
        geom_hline(yintercept=0.13, linetype='dashed', color='gray', size = 1) +
        geom_hline(yintercept=0.08, linetype='dashed', color='lightgrey', size = 1) +
        geom_boxplot(fill = 'lightgrey') + geom_point() +
        geom_label(data=toplot_top, mapping = aes(label='target_symbol'), size = 6) +
        theme_bw() +
        theme(figure_size=(5,3)) +
        labs(x= 'Overexpressed Kinase', y = 'Differential Abundance\n(BP-R2 score)') +
        labs(title = 'Mass Cytometry Data (EGF vs Control)') +
        expand_limits(y=toplot['diff'].max()*1.1) 
    )

    return toplot, p

def plot_top_kinase_egf(
    kinase_df: pd.DataFrame,
    control_studies: Iterable[str] = (
        'Chen et al. 2025 (HEK293T)',
        'Tuechler et al. 2025 (PDGFRb)'
    ),
    n_top: int = 25,
    highlight_egf: bool = True,
) -> Tuple[pd.DataFrame, ggplot]:
    """Plot top EGFR-regulated kinases with optional highlighting."""

    toplot = kinase_df[~kinase_df['study'].isin(control_studies)].copy()
    # scale the activity per study and resource to -3 to 3
    for study in toplot['study'].unique():
        for resource in toplot['resource'].unique():
            mask = (toplot['study'] == study) & (toplot['resource'] == resource)
            scaler = StandardScaler()
            toplot.loc[mask, 'activity'] = scaler.fit_transform(toplot.loc[mask, ['activity']])

    toplot['abs_activity'] = np.abs(toplot['activity'])
    toplot['symbol'] = pp.utils.flex_site_mapping(toplot['kinase'])

    toplot = toplot[['study','symbol', 'resource', 'abs_activity', 'activity']].copy()
    toplot['axis_group'] =  toplot['resource'] + '--' + toplot['symbol']
    # if highlight_egf, add an asterisk to the axis_group
    if highlight_egf:
        egf_kinases = pp.utils.flex_site_mapping(pp.netgt.get_canonical_egf_proteins())
        toplot['is_egf'] = toplot['symbol'].isin(egf_kinases)
        toplot['axis_group'] = np.where(toplot['is_egf'], toplot['axis_group'] + ' (*)', toplot['axis_group'])
    # get the top 10 deregulated kinases per axis group using abs_activity
    top_kinases = toplot.groupby(['symbol', 'resource','axis_group'])['abs_activity'].mean().reset_index()
    top_kinases = top_kinases.groupby(['resource']).apply(lambda x: x.nlargest(n_top, 'abs_activity')).reset_index(drop=True)
    top_kinase_order = top_kinases.groupby('axis_group')['abs_activity'].mean().sort_values(ascending=False).index
    toplot = toplot[toplot['axis_group'].isin(top_kinases['axis_group'])]
    
    # set axis_group as a categorical with the order from top_kinase_order
    toplot['axis_group'] = pd.Categorical(toplot['axis_group'], categories=top_kinase_order[::-1], ordered=True)
    # replace the values of resource and make it categorical
    toplot['resource'] = toplot['resource'].map(pp.config.KINSUB_LABELS)
    toplot['resource'] = pd.Categorical(toplot['resource'], categories = pp.config.KINSUB_LABELS.values(), ordered=True)

    toplot = toplot.dropna()

    toplot_mean = toplot.groupby(['resource', 'axis_group'])['activity'].mean().reset_index().rename(columns={'activity': 'mean_activity'})

    toplot = toplot.merge(toplot_mean, on=['resource', 'axis_group'])

    p = (
        ggplot(toplot, aes(x='axis_group', y='activity', fill='activity')) +
        geom_violin(mapping = aes(fill = 'mean_activity'), alpha = 0.8) +
        geom_point() +
        scale_fill_gradient2(low='blue', mid='white', high='red', midpoint=0) +
        coord_flip() +
        facet_wrap('~resource', ncol=4, scales='free') +
        # remove everything before -- from the axis
        scale_x_discrete(labels=lambda l: [x.split('--')[1] for x in l]) +
        #scale_fill_manual(values=pp.config.KINSUB_COLORS, labels=pp.config.KINSUB_LABELS)+
        theme_bw() +
        # vertical line at 0 dashed
        geom_hline(yintercept=0, linetype='dashed', color='grey') +
        labs(y='Kinase Activity (z-score)', x='Kinase (by absolute activity)') +
        theme(figure_size=(11,6), legend_position='right')
    )
    
    return toplot, p

def plot_top_kinases_control(
    kinase_df: pd.DataFrame,
    control_studies: Iterable[str] = (
        'Chen et al. 2025 (HEK293T)',
        'Tuechler et al. 2025 (PDGFRb)'
    ),
    n_top: int = 5,
    highlight_egf: bool = True,
) -> Tuple[pd.DataFrame, ggplot]:
    """Plot top kinases for control studies with optional EGF highlighting."""

    toplot = kinase_df[kinase_df['study'].isin(control_studies)].copy()
    toplot['symbol'] = pp.utils.flex_site_mapping(toplot['kinase'])
    # scale the activity per study and resource to -3 to 3
    for study in toplot['study'].unique():
        for resource in toplot['resource'].unique():
            mask = (toplot['study'] == study) & (toplot['resource'] == resource)
            scaler = StandardScaler()
            toplot.loc[mask, 'activity'] = scaler.fit_transform(toplot.loc[mask, ['activity']])
    toplot['abs_activity'] = toplot['activity'].abs()

    toplot = toplot[['study','symbol', 'resource', 'abs_activity', 'activity',]].copy()
    toplot['axis_group'] = toplot['study'] + '__' + toplot['resource'] + '--' + toplot['symbol']
    if highlight_egf:
        egf_kinases = pp.utils.flex_site_mapping(pp.netgt.get_canonical_egf_proteins())
        toplot['is_egf'] = toplot['symbol'].isin(egf_kinases)
        toplot['axis_group'] = np.where(toplot['is_egf'], toplot['axis_group'] + ' (*)', toplot['axis_group'])
    # get the top 10 deregulated kinases per axis group using abs_activity
    top_kinases = toplot.groupby(['symbol', 'resource','axis_group', 'study'])['abs_activity'].mean().reset_index()
    top_kinases = top_kinases.groupby(['resource', 'study']).apply(lambda x: x.nlargest(n_top, 'abs_activity')).reset_index(drop=True)
    top_kinase_order = top_kinases.groupby('axis_group')['abs_activity'].mean().sort_values(ascending=False).index
    toplot = toplot[toplot['axis_group'].isin(top_kinases['axis_group'])]
    
    # set axis_group as a categorical with the order from top_kinase_order
    toplot['axis_group'] = pd.Categorical(toplot['axis_group'], categories=top_kinase_order[::-1], ordered=True)
    toplot['resource'] = toplot['resource'].map(pp.config.KINSUB_LABELS)
    toplot['resource'] = pd.Categorical(toplot['resource'], categories = pp.config.KINSUB_LABELS.values(), ordered=True)

    p = (
        ggplot(toplot, aes(x='axis_group', y='activity', fill='activity')) +
        geom_col() +
        coord_flip() +
        facet_wrap('~study+resource', ncol=4, scales='free') +
        # remove everything before -- from the axis
        scale_x_discrete(labels=lambda l: [x.split('--')[1] for x in l]) +
        #scale_fill_manual(values=pp.config.KINSUB_COLORS, labels=pp.config.KINSUB_LABELS)+
        scale_fill_gradient2(low='blue', mid='white', high='red', midpoint=0) +
        theme_bw() +
        # vertical line at 0 dashed
        geom_hline(yintercept=0, linetype='dashed', color='grey') +
        labs(y='Kinase Activity (z-score)', x='Kinase (by absolute activity)') +
        theme(figure_size=(11,4), legend_position='right') +
        theme(strip_text=element_text(size=8))
    )
    
    return toplot, p

def plot_kinase_correlation(
    kinase_df: pd.DataFrame,
    control_studies: Iterable[str] = (
        'Chen et al. 2025 (HEK293T)',
        'Tuechler et al. 2025 (PDGFRb)'
    ),
) -> Tuple[pd.DataFrame, ggplot]:
    """Scatter plot comparing kinase activity correlations vs. coverage."""
    toplot = kinase_df.copy()
    cov_df = kinase_df[['study','resource', 'proportion_sites_covered']].drop_duplicates()
    cor_results = []
    for resource in toplot['resource'].unique():
        tocor = toplot[toplot['resource'] == resource][['kinase', 'study', 'activity']].pivot(index='kinase', columns='study', values='activity')
        cor_res = pp.utils.corr_from_df(tocor)
        cor_res['resource'] = resource
        cor_results.append(cor_res)
    cor_df = pd.concat(cor_results)
    toplot = cor_df.merge(cov_df, left_on=['study_a', 'resource'], right_on=['study', 'resource'])
    # is control column indicating whether study_a or study_b is a control study
    toplot['is_control'] = toplot['study_a'].isin(control_studies) | toplot['study_b'].isin(control_studies)
    toplot['is_control'] = toplot['is_control'].map({True: 'Between EGF and control studies', False: 'Between EGF studies only'})

    toplot_means = toplot.groupby(['resource', 'is_control']).agg({'spearman_rho': 'mean', 'proportion_sites_covered': 'mean'}).reset_index()

    p = (
        ggplot(toplot, aes(y='proportion_sites_covered', x='spearman_rho', color = 'resource', fill = 'resource')) +
        geom_point(color = 'black') +
        geom_point(data=toplot_means, size=4, shape='s', color = 'black') +
        facet_grid(rows = 'is_control') +
        scale_fill_manual(values=pp.config.KINSUB_COLORS, labels=pp.config.KINSUB_LABELS)+
        labs(y='Average Site Coverage', x='Spearman Rho Between studies') +
        theme_bw() +
        theme(figure_size=(4,5)) +
        theme(legend_position='none')
    )
    return toplot, p

def cor_per_timepoint(data_df: pd.DataFrame) -> Tuple[pd.DataFrame, ggplot]:
    """Visualize pairwise correlations per timepoint across studies."""

    time_point_df = data_df[data_df['study'].str.contains('This study (HEK293F', regex=False)].copy()
    time_point_df = time_point_df[time_point_df['adj.P.Val'] <= 0.05]

    toplot = time_point_df.copy()
    toplot['study'] = toplot['study'].str.replace('This study|\(|\)', '', regex = True)
    toplot['study'] = toplot['study'] + '-' + toplot['comparison']
    # store study as a string
    toplot['study'] = toplot['study'].astype(str)
    toplot_wide = toplot.pivot(index='id', columns='study', values='logFC')
    toplot = pp.utils.corr_from_df(toplot_wide)
    toplot['score'] = toplot['spearman_rho'].round(2)
    # arrange study_a and b in alphabetical order
    toplot['study_a'] = pd.Categorical(toplot['study_a'], categories=sorted(toplot['study_a'].unique()), ordered=True)
    toplot['study_b'] = pd.Categorical(toplot['study_b'], categories=sorted(toplot['study_b'].unique()), ordered=True)

    # filter to 'HEK293F TR' on study a and to 'HEK293F-' in study b
    toplot = toplot[toplot['study_a'].str.contains('HEK293F TR')]
    toplot = toplot[toplot['study_b'].str.contains('HEK293F-', regex=False)]
    # remove these and keep only the minutes
    toplot['study_a'] = toplot['study_a'].str.replace('HEK293F TR-', '', regex = True)
    toplot['study_b'] = toplot['study_b'].str.replace('HEK293F-', '', regex = True)
    # arrange the mins in numerical order (creating study_a_num_min by removing 'min')
    toplot['study_a_num_min'] = toplot['study_a'].str.replace('min', '', regex = True).astype(int)
    toplot['study_b_num_min'] = toplot['study_b'].str.replace('min', '', regex = True).astype(int)
    toplot['study_a'] = pd.Categorical(toplot['study_a'], categories=toplot.sort_values('study_a_num_min')['study_a'].unique(), ordered=True)
    toplot['study_b'] = pd.Categorical(toplot['study_b'], categories=toplot.sort_values('study_b_num_min')['study_b'].unique(), ordered=True)

    p = (
        ggplot(toplot, aes(y = 'study_a', x = 'study_b', fill = 'score', label = 'score')) +
        geom_tile(color = 'black') +
        geom_label(fill ='white', size = 8) +
        ylab('HEK293F (TR)') + xlab('HEK293F') +
        labs(fill='Spearman\nRho') +
        scale_fill_continuous(cmap_name = 'Reds') +
        theme_bw() +
        theme(figure_size=(3.5,4), axis_text_x=element_text(angle=45, hjust=1))
    )

    return toplot, p


def get_overlap(
    filt_df: pd.DataFrame,
    int_fun,
    metric_col: str = 'score'
) -> pd.DataFrame:
    """Apply ``int_fun`` across studies and tidy the resulting overlap metric."""
    overlap_df = filt_df.copy()
    toplot_wide = overlap_df[['id', 'logFC', 'study']].pivot(index='id', columns='study', values='logFC')
    overlap_df = int_fun(toplot_wide)
    cats = list(pp.config.STUDY_COLORS.keys())
    overlap_df['study_a'] = pd.Categorical(overlap_df['study_a'], categories=cats, ordered=True)
    overlap_df['study_b'] = pd.Categorical(overlap_df['study_b'], categories=cats, ordered=True)
    overlap_df[metric_col + '_round'] = overlap_df[metric_col].round(2)
    return overlap_df

def plot_combined_cor(filt_df: pd.DataFrame) -> Tuple[pd.DataFrame, ggplot]:
    """Joint visualization of overlap and correlation metrics."""
    data_jac  = get_overlap(filt_df, pp.utils.overlap_from_df)
    data_spear = get_overlap(filt_df, pp.utils.corr_from_df, metric_col='spearman_rho')
    data_spear_2 = get_overlap(filt_df[filt_df['adj.P.Val'] <= 0.05], pp.utils.corr_from_df, metric_col='spearman_rho')
    data_spear_2 = data_spear_2.rename(columns={'spearman_rho_round': 'spearman_rho_round_sig'})
    toplot_merged = pd.merge(data_jac, data_spear, on = ['study_a', 'study_b'], how='left')
    toplot_merged = pd.merge(toplot_merged, data_spear_2, on = ['study_a', 'study_b'], how='left')
    toplot_merged = toplot_merged[['study_a', 'study_b', 'score_round', 'spearman_rho_round', 'spearman_rho_round_sig']]
    toplot_merged = toplot_merged.melt(id_vars=['study_a', 'study_b'], var_name='metric', value_name='score')
    # rename values in 'value'
    toplot_merged['metric'] = toplot_merged['metric'].replace({'score_round': 'Jaccard\nIndex', 'spearman_rho_round': 'Spearman\nRho (all sites)', 'spearman_rho_round_sig': 'Spearman\nRho (adj. P-value <= 0.05)'})
    # set category order to jaccard index, spearman rho and spearman rho sig
    toplot_merged['metric'] = pd.Categorical(toplot_merged['metric'], categories=['Jaccard\nIndex', 'Spearman\nRho (all sites)', 'Spearman\nRho (adj. P-value <= 0.05)'], ordered=True)
    p = (
        ggplot(toplot_merged, aes(x='study_a', y='study_b', fill='score', label = 'score')) +
        geom_tile(color = 'black') +
        geom_label(fill ='white', size = 8) +
        scale_fill_continuous(cmap_name = 'Reds') +
        facet_grid('~metric', scales='free') +
        theme_bw() +
        labs(fill='Score') +
        theme(
            axis_text_x=element_text(angle=45, hjust=1),
            axis_title_x = element_blank(),
            axis_title_y=element_blank()
        ) +
        theme(figure_size=(10.5,4))
    )
    return toplot_merged, p

def plot_canonical_sites(filt_df: pd.DataFrame) -> Tuple[pd.DataFrame, ggplot]:
    """Plot abundance of canonical EGFR sites across studies."""
    toplot = filt_df.copy()
    toplot['is_egf'] = toplot['id'].isin(pp.netgt.get_canonical_egf_sites())
    toplot['is_egfr'] = toplot['symbol_id'].str.contains('EGFR', regex = False)
    toplot['annotation'] = 'other'
    toplot.loc[toplot['is_egf'], 'annotation'] = 'EGF pathway'
    toplot.loc[toplot['is_egfr'], 'annotation'] = 'EGFR'
    toplot = toplot[toplot['annotation'] != 'other']
    toplot['logFC'] = toplot['logFC'].clip(-3, 3)

    symbol_order = (
        toplot.groupby('symbol_id')['logFC']
        .mean()
        .sort_values(ascending=False)
        .index
    )
    toplot['symbol_id'] = pd.Categorical(toplot['symbol_id'], categories=symbol_order, ordered=True)

    p = (
        ggplot(toplot, aes(x = 'symbol_id', y = 'study', fill = 'logFC')) +
        scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
        geom_point(size = 5) +
        theme_bw() +
        facet_grid(cols = 'annotation', scales = 'free', space ='free') +
        theme(figure_size=(11, 2.5)) +
        labs(x = "Site") +
        theme(axis_text_x=element_text(angle=45, hjust=1, size = 8),
              axis_text_y = element_text(size = 8),
              axis_title_y = element_blank()) 

    )
    return toplot, p

def plot_volcanos(filt_df: pd.DataFrame) -> Tuple[pd.DataFrame, ggplot]:
    """Generate volcano plots for each study."""
    toplot = filt_df.copy()
    toplot['study'] = pd.Categorical(toplot['study'], categories=toplot['study'].cat.categories[::-1])
    
    label_df = toplot[['study', 'comparison', 'stimuli']].drop_duplicates()
    label_df['label'] = label_df['stimuli'] + '\n(' + label_df['comparison'] + ')'
    label_df['logFC'] = 0
    label_df['log_pval'] = 20

    p = (
        ggplot(toplot, aes(x='logFC', y='log_pval', fill='logFC')) +
        geom_point(alpha = 0.3, size = 2) +
        # add egf sites as yellow, larger points
        geom_point(data=toplot[toplot['is_egf']], fill='yellow', size=3) +
        scale_fill_gradient2(low='blue', mid='white', high='red', midpoint=0) +
        # add labels in the top left corner of each facet
        geom_label(data=label_df, mapping=aes(x='logFC', y='log_pval', label='label'), fill = 'white', size = 6) +
        facet_wrap('~study', nrow = 2) +
        theme_bw() +
        xlab('logFC') + ylab('-log10(P-value)') +
        theme(figure_size=(7, 3)) +
        theme(strip_text=element_text(size=6))
    )
    return toplot, p

def plot_cluster_means(cluster_df: pd.DataFrame) -> Tuple[pd.DataFrame, ggplot]:
    """Plot mean logFC trajectories per cluster."""
    # group by study and cluster, get mean and sd of logFC
    toplot = (
        cluster_df.groupby(['study', 'cluster', 'timepoint'])['logFC']
        .agg(['mean', 'std'])
        .reset_index()
    )
    
    p = (
        ggplot(toplot, aes(x='timepoint', y='mean', color='factor(cluster)', group='factor(cluster)')) +
        geom_line() +
        geom_point() +
        geom_ribbon(aes(ymin='mean - std', ymax='mean + std', fill='factor(cluster)'), alpha=0.2, color=None) +
        facet_wrap('~study', nrow=1) +
        theme_bw() +
        xlab('Timepoint (min)') + ylab('Mean logFC') +
        labs(color='Cluster', fill='Cluster') +
        theme(figure_size=(6, 3)) 
    )
    return toplot, p

def plot_sil_scores(sil_df: pd.DataFrame) -> ggplot:
    """Plot silhouette scores across candidate cluster numbers."""
    toplot = sil_df.copy()
    p = (
        ggplot(sil_df, aes(x='k', y='silhouette_score', group='study', color='study')) +
        geom_line() +
        geom_point() +
        scale_color_manual(values = pp.config.STUDY_COLORS) +
        theme_bw() +
        xlab('Number of clusters (k)') + ylab('Silhouette score') +
        labs(color='Study') +
        theme(figure_size=(5, 3)) 
    )
    return toplot, p

def plot_site_count(data_df: pd.DataFrame) -> Tuple[pd.DataFrame, ggplot]:
    """Bar plot summarizing detected site counts per study."""

    toplot = data_df[['id', 'study']].drop_duplicates()
    toplot['residue'] = toplot['id'].str.replace('.*_', '', regex = True).str.replace('[0-9]+', '', regex = True)
    per_study_count = toplot.groupby('study', observed = False).size().reset_index(name='total_sites')
    int_study_order = list(per_study_count.sort_values('total_sites', ascending=False)['study'])[::-1]
    toplot = toplot.groupby(['study', 'residue'], observed = False).size().reset_index(name='count')
    toplot = toplot[toplot['residue'].isin(['S', 'T', 'Y'])]
    toplot['study'] = pd.Categorical(toplot['study'], categories=int_study_order, ordered=True)

    p = (
        ggplot(toplot, aes(y = 'count', x = 'study', fill = 'study')) + 
        scale_fill_manual(values = pp.config.STUDY_COLORS) +
        geom_col(stat = 'identity') +
        coord_flip() +
        facet_wrap('~residue', scales='free_x') +
        ylab('Site count') +
        theme_bw() +
        theme(axis_text_x=element_text(angle=45, hjust=1)) +
        theme(figure_size=(4.5, 3)) +
        theme(axis_title_y=element_blank(), legend_position='none')
    )
    return toplot, p

def plot_target_similarity(input_df: pd.DataFrame) -> Tuple[pd.DataFrame, ggplot]:
    """Visualize sequence vs. target similarity for kinase pairs."""

    # group by resource and take averages of score and score_seq
    toplot = input_df.groupby('resource')[['score', 'score_seq']].mean().reset_index()

    # resource order as defined by lablels of kinsub colors
    toplot['resource'] = pd.Categorical(
        toplot['resource'],
        categories=list(pp.config.KINSUB_LABELS.keys()),
        ordered=True
    )

    plot = (
        ggplot(toplot, aes(y='score', x = 'resource', fill = 'resource')) +
        geom_col() +
        scale_fill_manual(values=pp.config.KINSUB_COLORS) +
        ylab('Overlap coefficient') + xlab('Resource') +
        theme_bw() +
        theme(axis_ticks_x=element_blank(), axis_text_x=element_blank()) +
        theme(figure_size=(2, 2), legend_position='none')
    )

    return toplot, plot

def plot_specialized_sites_ratio(input_df: pd.DataFrame) -> Tuple[pd.DataFrame, ggplot]:
    """Plot the fraction of specialized vs promiscuous sites per resource."""
    
    specialized_threshold = 3
    # compute, per resource, and per target, the number of rows
    per_site_sources = input_df.groupby(['resource', 'target'])['source'].count().reset_index()
    per_site_sources['is_specialized'] = per_site_sources['source'] <= specialized_threshold

    # per resource, compute the specialzied to non-specialized ratio
    toplot = (
        per_site_sources
        .groupby('resource')['is_specialized']
        .mean().reset_index()
    )

    # resource as category following KINSUB_LABELS order
    toplot['resource'] = pd.Categorical(
        toplot['resource'], 
        categories=list(pp.config.KINSUB_LABELS.keys()), 
        ordered=True
    )

    plot = (

        ggplot(toplot, aes(x='resource', y='is_specialized', fill = 'resource')) +
        geom_col() +
        scale_fill_manual(values=pp.config.KINSUB_COLORS, 
                        name='Resource', 
                        labels=list(pp.config.KINSUB_LABELS.values())) +
        labs(x='Resource', y='Specialized sites ratio') +
        # remove x axis ticks and labels
        theme_bw() +
        theme(axis_ticks_x=element_blank(), axis_text_x=element_blank()) +
        theme(figure_size=(2, 2), legend_position='none')

    )

    return toplot, plot

def plot_kinase_superfamily_proportions(int_df: pd.DataFrame) -> Tuple[pd.DataFrame, ggplot]:
    """Display kinase superfamily composition per resource."""

    kin_info = pp.kinsub.get_kinase_info_df(sequences=False)
    kin_to_family = dict(zip(kin_info['UniprotID'], kin_info['superfamily']))
    int_df['superfamily'] = int_df['source'].map(kin_to_family)
    int_df[(int_df['superfamily'] == 'Dual') & (int_df['resource'] == 'literature')]

    # Group by resource and superfamily to count interactions
    toplot = (
        int_df
        .groupby(['resource', 'superfamily'])['source']
        .count()
        .reset_index(name='count')
    )

    # Compute proportions per resource
    toplot['proportion'] = (
        toplot
        .groupby('resource')['count']
        .transform(lambda x: x / x.sum())
    )

    # set resource order to match KINSUB LABELS value order
    toplot['resource'] = pd.Categorical(
        toplot['resource'],
        categories=list(pp.config.KINSUB_LABELS.keys()),
        ordered=True
    )
    toplot['proportion_rounded'] = toplot['proportion'].round(2)

    plot = (
        ggplot(toplot, aes(
            x='superfamily', fill='proportion', y='resource', 
            label = 'proportion_rounded'
            )) +
        geom_tile() +
        # fill with inferno palette
        scale_fill_continuous(cmap_name = 'Reds')+
        geom_label( size=8, fill='white', show_legend=False) +
        scale_y_discrete(labels = lambda x: [pp.config.KINSUB_LABELS.get(i, i) for i in x]) +
        theme_bw()+
        labs(y='Resource', x='Superfamily', fill='Proportion') +
        # figure size
        theme(
            axis_text_x=element_text(rotation =45, vjust = 1, hjust = 1, size = 8), 
            figure_size=(3.5, 3)
        ) 
    )

    return toplot, plot

def plot_kinsub_barplots(int_df: pd.DataFrame) -> Tuple[pd.DataFrame, ggplot]:
    """Stacked resource bar plots summarizing interaction counts by class."""

    int_df['target_protein'] = int_df['target'].str.replace('_.*', '', regex=True)
    toplot_interactions = int_df['resource'].value_counts().reset_index()
    toplot_interactions['category'] = 'Total Interactions'
    toplot_sites = int_df.groupby('resource')['target'].nunique().reset_index().rename(columns={'target': 'count'})
    toplot_sites['category'] = 'Phosphosites'
    toplot_kins = int_df.groupby('resource')['source'].nunique().reset_index().rename(columns={'source': 'count'})
    toplot_kins['category'] = 'Kinases'
    toplot_target_prots = int_df.groupby('resource')['target_protein'].nunique().reset_index().rename(columns={'target_protein': 'count'})
    toplot_target_prots['category'] = 'Phosphoproteins'

    toplot = pd.concat([toplot_interactions, toplot_sites, toplot_kins, toplot_target_prots], ignore_index=True)

    # reorder resource to match KINSUB LABELS value order
    toplot['resource'] = pd.Categorical(
        toplot['resource'],
        categories=list(pp.config.KINSUB_LABELS.keys()),
        ordered=True
    )

    plot = (
        ggplot(toplot, aes(x='resource', y='count', fill='resource')) +
        scale_fill_manual(values=pp.config.KINSUB_COLORS) +
        geom_col() +
        facet_wrap('~category', scales='free_y', nrow = 1) +
        labs(x='Resource', y='Count', fill='Resource') +
        theme_bw() +
        scale_y_continuous(labels=scientific_format()) +
        theme(axis_text_x = element_blank(), axis_ticks_x = element_blank()) +
        theme(figure_size=(8, 2)) +
        theme(legend_position='none')
    )

    return toplot, plot

def plot_inhibition_thresholds(roc_summary: pd.DataFrame) -> ggplot:
    """Plot AUROC as a function of inhibition threshold selection."""
    toplot = roc_summary.groupby(['resource', 'threshold'])['auroc_score'].mean().reset_index()
    toplot['cutoff'] = toplot['resource'].str.split('-').str[1]
    toplot['resource'] = toplot['resource'].str.split('-').str[0]
    # if cutoff is na then it is literature
    toplot['cutoff'] = toplot['cutoff'].fillna(toplot['resource'])
    # threshold as a category
    toplot['threshold'] = toplot['threshold'].round(2).astype(str)

    plot = (
        ggplot(toplot, aes(y = 'auroc_score', x = 'threshold', fill = 'resource')) +
        geom_col(position='dodge', width=0.7 , color = 'black') +
        facet_wrap('~cutoff') +
        scale_fill_manual(values = pp.config.KINSUB_COLORS, labels=pp.config.KINSUB_LABELS) +
        theme_bw() +
        xlab('Kinase in-vitro inhibition threshold') + ylab('AUROC score') +
        theme(figure_size=(7, 5) ,axis_text = element_text(size = 8))
    )

    return toplot, plot

def plot_roc_curve(roc_data: pd.DataFrame) -> Tuple[pd.DataFrame, ggplot]:
    """Plot ROC curves for a specific inhibition threshold."""
    toplot = roc_data.copy()
    # remove threshold
    toplot['cutoff'] = toplot['resource'].str.replace('.*-', '', regex=True)
    toplot['group'] = toplot['resource'] + '_' + toplot['iteration'].astype(str)
    toplot['resource'] = toplot['resource'].str.replace('-.*', '', regex=True) 

    plot = (
        ggplot(toplot, aes(x='fpr', y='tpr', color='resource')) +
        geom_line(aes(group = 'group'), alpha = 0.1) + 
        geom_line(stat='summary', fun_y=np.mean, size=1.5) +
        facet_wrap('~cutoff', scales='free') +
        geom_abline(slope=1, intercept=0, linetype='dashed', color='gray') +
        scale_color_manual(values=pp.config.KINSUB_COLORS, labels=pp.config.KINSUB_LABELS) +
        labs(x='False Positive Rate',
                y='True Positive Rate') +
        theme_bw() +
        theme(figure_size=(7, 5) ,axis_text = element_text(size = 8))
    )
    return toplot, plot

def plot_hijazi_roc(
    roc_summary: pd.DataFrame,
    resource_coverage: pd.DataFrame,
    int_threshold: float = 0.5,
) -> Tuple[pd.DataFrame, ggplot]:
    """Summarize Hijazi ROC performance and coverage at a given threshold."""

    roc_summary = roc_summary[roc_summary['threshold'] == int_threshold]
    toplot = roc_summary.groupby(['resource', 'threshold'])['auroc_score'].mean().reset_index()
    toplot = toplot.merge(resource_coverage, on='resource')

    toplot['cutoff'] = toplot['resource'].str.split('-').str[1]
    toplot['cutoff'] = toplot['cutoff'].fillna(toplot['resource'])
    toplot['resource'] = toplot['resource'].str.split('-').str[0]

    plot = (
        ggplot(toplot, aes(x='auroc_score', y='proportion_sites_covered', color='resource',shape = 'cutoff'))+
        geom_point(size = 3) +
        labs(
            x='AUROC score',
            y='Coverage'
        ) +
        scale_color_manual(values=pp.config.KINSUB_COLORS, labels=pp.config.KINSUB_LABELS)+
        theme_bw() +
        theme( axis_text_x=element_text(rotation =60, vjust = 1, hjust = 0.5, size = 8) ,
            axis_text_y = element_text(size = 8))+
        theme(figure_size=(6, 3)) +
        theme(axis_title_y = element_text()) +
        expand_limits(x=(0.5, 0.7), y=(0, 1)) +
        theme(figure_size=(7, 5) ,axis_text = element_text(size = 8))
    )

    return toplot, plot

def reorder_df_by_resource(input_df: pd.DataFrame) -> pd.DataFrame:
    """Order resources consistently using the configured colour palette."""
    # reorder the resource column, depending on the order that it contains the KINSUB_LABELS keys
    input_df['resource_raw'] = input_df['resource'].str.replace('-.*', '', regex=True)
    input_df['resource_raw'] = pd.Categorical(
        input_df['resource_raw'],
        categories=list(pp.config.KINSUB_LABELS.keys()),
        ordered=True
    )
    input_df = input_df.sort_values('resource_raw')
    input_df['cutoff'] = input_df['resource'].str.split('-').str[1]
    input_df['cutoff'] = input_df['cutoff'].fillna(input_df['resource'])
    input_df['resource_label'] = input_df['resource_raw'].map(pp.config.KINSUB_LABELS).astype(str) + '\n(' + input_df['cutoff'] + ')'
    # order resource_label in order of appearance in the dataframe
    input_df['resource_label'] = pd.Categorical(
        input_df['resource_label'],
        categories=input_df['resource_label'].unique(),
        ordered=True
    )

    return input_df

def plot_hijazi_kinase_heatmap(kinase_df: pd.DataFrame) -> Tuple[pd.DataFrame, ggplot]:
    """Heatmap of Hijazi kinase activities with hierarchical ordering."""

    toplot = kinase_df.dropna()[['kinase', 'condition', 'activity', 'resource']].groupby(['kinase', 'condition', 'resource']).mean().reset_index()
    common_kins = toplot.groupby('kinase').filter(lambda x: len(x['resource'].unique()) == len(toplot['resource'].unique()))['kinase'].unique()
    toplot = toplot[toplot['kinase'].isin(common_kins)]

    # sort kinases by decreasing average activity
    avg_activity = toplot.groupby('kinase')['activity'].mean().sort_values(ascending=False)
    toplot['kinase'] = pd.Categorical(toplot['kinase'], categories= avg_activity.index, ordered=True)

    # sort conditions by decreasing average activity
    avg_condition = toplot.groupby('condition')['activity'].mean().sort_values(ascending=False)
    toplot['condition'] = pd.Categorical(toplot['condition'], categories= avg_condition.index, ordered=True)

    # reorder the resource column, depending on the order that it contains the KINSUB_LABELS keys
    toplot = reorder_df_by_resource(toplot)

    # clip activitities to -3, 3
    toplot['activity'] = toplot['activity'].clip(-3, 3)

    plot = (
        ggplot(toplot, aes(x = 'condition', y ='kinase', fill = 'activity')) +
        geom_tile(color = None) +
        facet_wrap('~resource_label', ncol = 5) +
        scale_fill_gradient2(low = 'blue', mid = 'white', high = 'red', limits = (-3,3))+
        # remoe x and y axes
        theme(axis_text = element_blank(), axis_ticks = element_blank()) +
        # make it wider
        theme(figure_size = (10, 3.5)) +
        labs(x = 'Drug', y = 'Kinase', fill = 'Activity') 
    )

    return toplot, plot


def nx_to_gv(G: nx.DiGraph) -> 'AGraph':
    """Convert a NetworkX DiGraph into a styled PyGraphviz graph."""
    A = nx.nx_agraph.to_agraph(G)
    A.node_attr.update(
        shape='ellipse',          # synonym of 'rectangle'
        style='filled',
        fillcolor="lightgrey",
        color="#000000FF", # here border
        penwidth=0.8,
        fontsize=35,
        fontname='Arial',
        margin="0.05,0.05",   # horizontal, vertical (inches)
        fixedsize='false',    # allow autosizing
        width='0.01',         # very small minima so margins can take effect
        height='0.01'
    )
    A.edge_attr.update(
        color='black',
        penwidth=5,
        arrowsize=2,
        fontsize=20,
    )

    # set width of graph
    A.graph_attr.update(width='s', height='10')
    # transparent bg
    A.graph_attr.update(bgcolor='white')

    # set render to LR
    A.graph_attr.update(rankdir='LR', splines='true', overlap='true', res = 300)
    return A


def gv_boolean_node_highlight(G: nx.DiGraph, int_nodes: Iterable[str]) -> 'AGraph':
    """Highlight a subset of nodes using a binary colour scheme."""
    A = nx_to_gv(G)

    for n in G.nodes():
        node = A.get_node(n)
        if n in int_nodes:
            node.attr['color'] = 'black'
            node.attr['penwidth'] = '6'
            node.attr['fillcolor'] = "#FF823A"  # solid hex (alpha not needed in Graphviz)
        else:
            node.attr['color'] = 'grey'
            node.attr['penwidth'] = '2'
            node.attr['fillcolor'] = "lightgrey"

    for u, v in G.edges():
        edge = A.get_edge(u, v)
        if u in int_nodes and v in int_nodes:
            edge.attr['color'] = 'black'
            edge.attr['penwidth'] = '6'
        else:
            edge.attr['color'] = 'grey'
            edge.attr['penwidth'] = '2'



    return A


def gv_continuous_node_highlight(
    G: nx.DiGraph,
    node_dict: Dict[str, float],
) -> 'AGraph':
    """Colour nodes by continuous scores using a red gradient."""
    A = nx_to_gv(G)

    # remove NaN values from dict
    node_dict = {k: v for k, v in node_dict.items() if pd.notna(v)}
    node_dict = {k: v for k, v in node_dict.items() if k in G.nodes()}
    if not node_dict:  # nothing to color
        return A

    raw_values = list(node_dict.values())
    vmin, vmax = 0,1  # set fixed vmin, vmax for consistency across plots

    # Prepare normalization & cmap
    norm = Normalize(vmin=vmin, vmax=vmax)
    my_cmap = cm.get_cmap("Reds")

    # Colors for nodes
    norm_values = norm(raw_values)  # in [0,1]
    colors = [my_cmap(x) for x in norm_values]
    hex_colors = [cm.colors.to_hex(c, keep_alpha=False) for c in colors]  # Graphviz wants hex RGB
    color_dict = dict(zip(node_dict.keys(), hex_colors))

    for n in G.nodes():
        node = A.get_node(n)
        if n in node_dict:
            node.attr['style'] = 'filled'
            node.attr['fillcolor'] = color_dict[n]

    return A


def render_graphviz(
    input_g: 'AGraph',
    use_symbols: bool = True,
    outf: str = '/tmp/g.png',
    width: int = 1000,
) -> Image:
    """Render a Graphviz graph to disk and return the corresponding image."""

    if use_symbols:
        for n in input_g.nodes():
            node = input_g.get_node(n)
            node.attr['label'] = pp.utils.flex_site_mapping([n])[0]

    input_g.draw(outf, prog = 'dot')
    # if is png
    if outf.endswith('.png'):
        return Image(outf, width=width)


def plot_legend(palette_name: str, title: str = 'Relative Kinase Activity') -> Tuple[plt.Figure, plt.Axes]:
    """Create a horizontal colourbar legend spanning 0–1 for a palette."""
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)

    cmap = cm.get_cmap(palette_name)
    norm = Normalize(vmin=0, vmax=1)

    cb1 = cm.ScalarMappable(norm=norm, cmap=cmap)
    cb1.set_array([])

    cbar = fig.colorbar(cb1, cax=ax, orientation='horizontal')
    cbar.set_label(title, fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    plt.show()
    return fig, ax
