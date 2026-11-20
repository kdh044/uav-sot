from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.biodrone_path = __import__('os').environ.get('SOT_BIODRONE', '/home/airlab/user/kdh/datasets/sot/train/biodrone_val')
    settings.davis_dir = ''
    settings.dtb70_path = '/home/airlab/user/kdh/datasets/sot/eval/dtb70'
    settings.got10k_lmdb_path = '/home/airlab/user/kdh/datasets/sot/got10k_lmdb'
    settings.got10k_path = '/home/airlab/user/kdh/datasets/sot/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/airlab/user/kdh/datasets/sot/itb'
    settings.lasot_extension_subset_path_path = '/home/airlab/user/kdh/datasets/sot/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/airlab/user/kdh/datasets/sot/lasot_lmdb'
    settings.lasot_path = '/home/airlab/user/kdh/datasets/sot/lasot'
    settings.network_path = '/home/airlab/user/kdh/SOT/ORTrack/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/airlab/user/kdh/datasets/sot/nfs'
    settings.otb_path = '/home/airlab/user/kdh/datasets/sot/otb'
    settings.prj_dir = '/home/airlab/user/kdh/SOT/ORTrack'
    settings.result_plot_path = '/home/airlab/user/kdh/SOT/ORTrack/output/test/result_plots'
    settings.results_path = '/home/airlab/user/kdh/SOT/ORTrack/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/airlab/user/kdh/SOT/ORTrack/output'
    settings.segmentation_path = '/home/airlab/user/kdh/SOT/ORTrack/output/test/segmentation_results'
    settings.tc128_path = '/home/airlab/user/kdh/datasets/sot/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/airlab/user/kdh/datasets/sot/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/airlab/user/kdh/datasets/sot/trackingnet'
    settings.uav123_path = '/home/airlab/user/kdh/datasets/sot/eval/uav123'
    settings.uav_path = '/home/airlab/user/kdh/datasets/sot/eval/uav123'
    settings.uavdt_path = '/home/airlab/user/kdh/datasets/sot/eval/uavdt'
    settings.visdrone2018_path = '/home/airlab/user/kdh/datasets/sot/eval/visdrone2018'
    settings.vot18_path = '/home/airlab/user/kdh/datasets/sot/vot2018'
    settings.vot22_path = '/home/airlab/user/kdh/datasets/sot/vot2022'
    settings.vot_path = '/home/airlab/user/kdh/datasets/sot/VOT2019'
    settings.youtubevos_dir = ''

    settings.uav123_10fps_path = '/home/airlab/user/kdh/datasets/sot/eval/uav123/UAV123_10fps'
    return settings

