import logging
import os
import time
import hydra
from omegaconf import OmegaConf
from dotenv import load_dotenv

from src.jobs.fairness_parity import FairnessParity
from src.config.schema import Config
from src.utils.general_utils import _innit_logger

load_dotenv()
hydra_config_path = os.getenv("HYDRA_CONFIG_PATH")
hydra_config_name = os.getenv("HYDRA_CONFIG_NAME")


@hydra.main(config_path=hydra_config_path, config_name=hydra_config_name, version_base=None)
def main(config: Config):
    logging.info("\n"+OmegaConf.to_yaml(config))
    fp = FairnessParity(config)
    fp.run_fairness_par()


if __name__ == "__main__":
    _innit_logger()
    main()
