"""High-level orchestrators for reproducible analysis pipelines."""

from __future__ import annotations

import os

from IPython.display import display

import phosphonetworks as pp


def _resolve_figures_dir(figures_dir: str | os.PathLike | None) -> str:
    """Return the directory where figures should be stored, creating it if needed."""

    base_dir = os.fspath(figures_dir or pp.config.FIGURES_DIR)
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def _figure_path(filename: str, figures_dir: str, subdir: str = 'individual') -> str:
    """Build the full path for a figure inside the requested subdirectory."""

    target_dir = os.path.join(figures_dir, subdir) if subdir else figures_dir
    os.makedirs(target_dir, exist_ok=True)
    return os.path.join(target_dir, filename)

def run_kinsub_pipeline(figures_dir: str | os.PathLike | None = None) -> bool:
    """Execute the kinase–substrate resource benchmarking workflow.

    Args:
        figures_dir: Optional base directory for saved figures. If ``None``,
            ``pp.config.FIGURES_DIR`` is used.
    """

    # load data
    figures_dir = _resolve_figures_dir(figures_dir)
    _ = pp.kinsub.get_kinase_info_df(sequences=True)
    kinsub_df = pp.kinsub.get_combined_kinsub()

    # compute kinase activities for Hijazi
    hijazi_data, hijazi_drug_to_target = pp.phosphodata.get_hijazi()
    hijazi_kin_act = pp.methods.hijazi_kinase_activity(hijazi_data, kinsub_df)

    # compute resource coverage
    resource_coverage = (
        hijazi_kin_act
        .groupby('resource')['proportion_sites_covered']
        .mean().reset_index()
    )

    # compute hijazi rocs
    roc_summary, roc_data = pp.methods.hijazi_roc_wrapper(hijazi_kin_act, hijazi_drug_to_target)

    # select one cutoff for the rest of analyses
    int_cutoff = 'Moderate'
    kinsub_selected = pp.kinsub.get_one_cutoff_combined_kinsub(cutoff=int_cutoff)

    # compute kinase similarities
    target_similarity_df = pp.kinsub.get_target_similarity_df(kinsub_selected)
    target_similarity_df = target_similarity_df[target_similarity_df['score_seq']>=50]
    print(f"Number of kinase pairs with sequence similarity >= 50: {target_similarity_df.shape[0]}")

    #### ---- plots ---- ####
    # kinase activity heatmaps
    data, plot = pp.viz.plot_hijazi_kinase_heatmap(hijazi_kin_act)
    plot.save(_figure_path('hijazi_kinase_activity_heatmap.png', figures_dir), dpi=300)
    plot.show()

    # Hijazi ROC summary
    data, plot = pp.viz.plot_hijazi_roc(roc_summary, resource_coverage, int_threshold=0.5)
    plot.save(_figure_path('hijazi_roc_analysis.png', figures_dir), width=6, height=3, dpi=300)
    plot.show()

    # Hijazi ROC one threshold
    data, plot = pp.viz.plot_roc_curve(roc_data[roc_data['threshold'] == 0.5])
    plot.save(_figure_path('hijazi_roc_curve.png', figures_dir), dpi=300)
    plot.show()

    # Hijazi ROC across thresholds
    data, plot = pp.viz.plot_inhibition_thresholds(roc_summary)
    plot.save(_figure_path('hijazi_inhibition_cutoffs.png', figures_dir), dpi=300)
    plot.show()

    # resource barplots
    data, plot = pp.viz.plot_kinsub_barplots(kinsub_selected)
    plot.save(_figure_path('kinsub_barplots.png', figures_dir), width=8, height=2, dpi=300)
    plot.show()

    # superfamily proportions
    data, plot = pp.viz.plot_kinase_superfamily_proportions(kinsub_selected)
    plot.save(_figure_path('kinase_superfamily_proportions.png', figures_dir), width=3.5, height=3, dpi=300)
    plot.show()

    # specialized sites ratio
    data, plot = pp.viz.plot_specialized_sites_ratio(kinsub_selected)
    plot.save(_figure_path('specialized_sites_ratio.png', figures_dir), width=2, height=2, dpi=300)
    plot.show()

    # target similarities
    data, plot = pp.viz.plot_target_similarity(target_similarity_df)
    plot.save(_figure_path('target_similarity.png', figures_dir), width=2, height=2, dpi=300)
    plot.show()

    return True

def run_site_pipeline(figures_dir: str | os.PathLike | None = None) -> None:
    """Generate summary figures for phosphosite-centric analyses.

    Args:
        figures_dir: Optional base directory for saved figures. If ``None``,
            ``pp.config.FIGURES_DIR`` is used.
    """

    # load data and filter
    figures_dir = _resolve_figures_dir(figures_dir)
    data_df = pp.phosphodata.get_all_studies()
    filt_df = pp.phosphodata.filter_studies(data_df)

    # cluster timecourse data
    sil_df, cluster_df = pp.phosphodata.get_timecourse_clusters(data_df)

    #### ---- plots ---- ####
    # site count and silhouette scores
    data, plot = pp.viz.plot_site_count(data_df)
    plot.save(_figure_path('site_count.png', figures_dir), dpi=300)
    plot.show()

    # silhouette scores
    data, plot = pp.viz.plot_sil_scores(sil_df)
    plot.save(_figure_path('silhouette_scores.png', figures_dir), dpi=300)
    plot.show()

    # cluster means
    data, plot = pp.viz.plot_cluster_means(cluster_df)
    plot.save(_figure_path('cluster_means.png', figures_dir), dpi=300)
    plot.show()

    # volcanos
    data, plot = pp.viz.plot_volcanos(filt_df)
    plot.save(_figure_path('volcanos.png', figures_dir), dpi=300)
    plot.show()

    # canonical sites
    data, plot = pp.viz.plot_canonical_sites(filt_df)
    plot.save(_figure_path('egf_sites.png', figures_dir), dpi=300)
    plot.show()

    # combined corrs
    data, plot = pp.viz.plot_combined_cor(filt_df)
    plot.save(_figure_path('overlap_correlation.png', figures_dir), dpi=300)
    plot.show()

    # corr per time point
    data, plot = pp.viz.cor_per_timepoint(data_df)
    plot.save(_figure_path('time_point_cor.png', figures_dir), dpi=300)
    plot.show()

def run_egf_kin_pipeline(figures_dir: str | os.PathLike | None = None) -> None:
    """Profile EGFR-driven kinase activities across studies.

    Args:
        figures_dir: Optional base directory for saved figures. If ``None``,
            ``pp.config.FIGURES_DIR`` is used.
    """

    # load data
    figures_dir = _resolve_figures_dir(figures_dir)
    kinsub = pp.kinsub.get_one_cutoff_combined_kinsub('Moderate')
    data_df = pp.phosphodata.filter_studies(pp.phosphodata.get_all_studies())

    # compute kinase activities
    kinase_df = pp.methods.egf_kinase_activity_analysis(kinsub, data_df)

    #### ---- plots ---- ####
    # kinase correlation
    data, plot = pp.viz.plot_kinase_correlation(kinase_df)
    plot.save(_figure_path('kinase_corr.png', figures_dir), dpi=300)
    plot.show()

    # top regulated kinases
    data, plot = pp.viz.plot_top_kinases_control(kinase_df)
    plot.save(_figure_path('top_control_kinases.png', figures_dir), dpi=300)
    plot.show()

    # top egf kinases
    data, plot = pp.viz.plot_top_kinase_egf(kinase_df)
    plot.save(_figure_path('top_egf_kinases.png', figures_dir), dpi=300)
    plot.show()

def run_egf_gt_pipeline(figures_dir: str | os.PathLike | None = None) -> None:
    """Evaluate ground-truth networks derived from EGFR perturbations.

    Args:
        figures_dir: Optional base directory for saved figures. If ``None``,
            ``pp.config.FIGURES_DIR`` is used.
    """

    # prepare data
    figures_dir = _resolve_figures_dir(figures_dir)
    kinsub = pp.kinsub.get_one_cutoff_combined_kinsub('Moderate')
    all_kinases = list(set(kinsub['source']))
    data_df = pp.phosphodata.get_all_studies()
    egf_kinases = pp.methods.egf_kinase_activity_analysis(kinsub, data_df)

    # prepare kinase-kinase PKN
    pkn = pp.utils.kinsub_to_pkn(kinsub, all_kinases, int_resource = 'literature')
    lit_kins = egf_kinases[egf_kinases['resource']=='literature'].copy()
    terminals = pp.utils.kinact_to_terminals(lit_kins, int_study = 'This study (HEK293T)')
    pkn, terminals = pp.utils.prepare_pkn_and_terminals(pkn, terminals)

    # get illustrative nets
    mean_net = pp.network_methods.mean_selection(pkn, terminals, n_edges=10)
    pagerank_net = pp.network_methods.pagerank_selection(pkn, terminals, n_edges=10)
    # commented to prevent GUROBI complaints during testing, can be removed once gurobi is available
    #rpcst_net = pp.network_methods.rpcst_selection(pkn, terminals, n_edges = 10, mip = 0.01)
    net_dict = {'mean_net': mean_net, 
                #'rpcst_net': rpcst_net,
                'pagerank_net': pagerank_net} 

    # get lun data
    lun_diffdata = pp.netgt.get_lun_diffdata()

    # corr of hek data
    wide_hek = pp.netgt.get_wide_hek293f_tr()
    filt_corr = pp.netgt.corr_func_sites(wide_hek, threshold=0.5)

    # get gt ints
    lun_gt_ints = pp.netgt.get_lun_gt_interactions(threshold = 0.13)
    signor_gt_ints = pp.netgt.get_canonical_egf_interactions()

    # get all gt ints and PKN
    all_ints = pp.netgt.prepare_all_gt_ints()
    kk_df = pp.kinsub.get_kk_pkn()
    kk_df['int_id'] = kk_df['source'] + '-->' + kk_df['target']

    #### ---- plots ---- ####
    # illustrative networks
    for k,v in net_dict.items():
        print(f"Network has {v.number_of_nodes()} nodes and {v.number_of_edges()} edges")
        net_agraph = pp.viz.gv_continuous_node_highlight(v, terminals)
        display(pp.viz.render_graphviz(net_agraph, width = 300))

    # plot top hits per KO kinase
    toplot, plot = pp.viz.plot_top_lundata(lun_diffdata)
    plot.save(_figure_path('egf_netgt_lun.png', figures_dir), dpi=300)
    plot.show()

    # top correlations
    data, plot = pp.viz.plot_top_corrs(filt_corr, wide_hek)
    plot.save(_figure_path('egf_netgt_corrs.png', figures_dir), dpi=300)
    display(plot.draw())

    # plot ground truth correlation values
    data, plot = pp.viz.plot_gt_corrs(filt_corr, lun_gt_ints, signor_gt_ints)
    plot.save(_figure_path('egf_netgt_gtcorrs.png', figures_dir), dpi=300)
    plot.show()

    # plot ground truth counts
    data, plot = pp.viz.plot_gt_counts(kk_df, all_ints)
    plot.save(_figure_path('egf_netgt_gtcounts.png', figures_dir), dpi=300)
    display(plot.draw())

    # plot per cut-off top n 
    data, plot = pp.viz.plot_n_top_cutoffs(kinsub)
    plot.save(_figure_path('egf_netgt_ntopcutoff.png', figures_dir), dpi=300)
    plot.show()


def run_net_benchmark_pipeline(figures_dir: str | os.PathLike | None = None) -> None:
    """Benchmark network reconstruction algorithms against curated GT sets.

    Args:
        figures_dir: Optional base directory for saved figures. If ``None``,
            ``pp.config.FIGURES_DIR`` is used.
    """

    # load kinase activities
    figures_dir = _resolve_figures_dir(figures_dir)
    kinsub = pp.kinsub.get_one_cutoff_combined_kinsub('Moderate')
    data_df = pp.phosphodata.filter_studies(pp.phosphodata.get_all_studies())
    kinase_df = pp.methods.egf_kinase_activity_analysis(kinsub, data_df)
    kinase_df['abs_activity'] = kinase_df['activity'].abs()

    # prepare inputs
    edge_lengths = [50, 75, 100, 125, 150, 175, 200, 225, 250]
    input_dicts = pp.network_methods.prepare_input_dict(kinase_df, edge_lengths)

    # run network analysis
    results = pp.network_methods.solve_networks(
        pp.config.CACHE_DIR + 'intermediate_files/network_input_dicts.pkl',
        output_path=os.path.join(pp.config.CACHE_DIR, 'intermediate_files/result_networks.pkl')
    )

    # prepare ground truth interactions
    gt_ints = pp.netgt.prepare_all_gt_ints()

    # compute overlaps
    res_df = pp.network_methods.compute_overlap_with_gt(gt_ints, results)

    #### ---- plots ---- ####
    data, plot = pp.viz.plot_net_results_heatmap(res_df)
    plot.save(_figure_path('egf_netgt_heatmap.png', figures_dir), dpi=300)
    plot.show()
