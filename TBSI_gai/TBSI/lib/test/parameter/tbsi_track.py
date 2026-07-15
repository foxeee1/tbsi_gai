from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.tbsi_track.config import cfg, update_config_from_file


def parameters(yaml_name: str):
    params = TrackerParams()
    save_dir = env_settings().save_dir
    # update default config from yaml file
    prj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    yaml_file = os.path.join(prj_dir, 'experiments/tbsi_track/%s.yaml' % yaml_name)

    update_config_from_file(yaml_file)
    params.cfg = cfg
    print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path: matches training save path
    # Priority: new structure (experiments/{name}/checkpoints/) first, fallback to old structure
    new_ckpt = os.path.join(save_dir, "experiments/%s/checkpoints/TBSITrack_ep%04d.pth.tar" %
                            (yaml_name, cfg.TEST.EPOCH))
    old_ckpt = os.path.join(save_dir, "checkpoints/train/tbsi_track/%s/TBSITrack_ep%04d.pth.tar" %
                            (yaml_name, cfg.TEST.EPOCH))
    if os.path.exists(new_ckpt):
        params.checkpoint = new_ckpt
    else:
        params.checkpoint = old_ckpt

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
