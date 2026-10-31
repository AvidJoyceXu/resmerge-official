import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import pathlib

from psi_policy.workspace.base_workspace import BaseWorkspace

from utils import setup_path_variables
PROJECT_ROOT, SRC_ROOT = setup_path_variables()

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=os.path.join(SRC_ROOT, 'psi_policy', 'config'),
    config_name="psi_policy_workspace"  
)
def train(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    if cfg.task.dataset_path is None:
            raise ValueError(
                "For training, dataset_path cannot be null. "
                "Please specify a valid dataset path."
            )
        
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    """
    Usage:
    Training:
    DATASET_ZARR=/path/to/demos.zarr python train.py logging.mode=disabled
    """
    train()
