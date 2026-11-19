from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.check_dir = '/root/autodl-tmp/TBSI_gai/TBSI/output'
    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/root/autodl-tmp/TBSI_gai/TBSI/data/got10k_lmdb'
    settings.got10k_path = '/root/autodl-tmp/TBSI_gai/TBSI/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/root/autodl-tmp/TBSI_gai/TBSI/data/itb'
    settings.lasher_path = '/root/autodl-tmp/TBSI_gai/data/lasher'
    settings.lasot_extension_subset_path_path = '/root/autodl-tmp/TBSI_gai/TBSI/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/root/autodl-tmp/TBSI_gai/TBSI/data/lasot_lmdb'
    settings.lasot_path = '/root/autodl-tmp/TBSI_gai/TBSI/data/lasot'
    settings.network_path = '/root/autodl-tmp/TBSI_gai/TBSI/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/root/autodl-tmp/TBSI_gai/TBSI/data/nfs'
    settings.otb_path = '/root/autodl-tmp/TBSI_gai/TBSI/data/otb'
    settings.prj_dir = '/root/autodl-tmp/TBSI_gai/TBSI'
    settings.result_plot_path = '/root/autodl-tmp/TBSI_gai/TBSI/output/test/result_plots'
    settings.results_path = '/root/autodl-tmp/TBSI_gai/TBSI/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/root/autodl-tmp/TBSI_gai/TBSI/output'
    settings.segmentation_path = '/root/autodl-tmp/TBSI_gai/TBSI/output/test/segmentation_results'
    settings.tc128_path = '/root/autodl-tmp/TBSI_gai/TBSI/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/root/autodl-tmp/TBSI_gai/TBSI/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/root/autodl-tmp/TBSI_gai/TBSI/data/trackingnet'
    settings.uav_path = '/root/autodl-tmp/TBSI_gai/TBSI/data/uav'
    settings.vot18_path = '/root/autodl-tmp/TBSI_gai/TBSI/data/vot2018'
    settings.vot22_path = '/root/autodl-tmp/TBSI_gai/TBSI/data/vot2022'
    settings.vot_path = '/root/autodl-tmp/TBSI_gai/TBSI/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

