from dataclasses import dataclass

import draccus

from cosmos_policy.experiments.robot.aloha.deploy import DeployConfig as BaseDeployConfig
from cosmos_policy.experiments.robot.aloha.deploy import deploy as base_deploy


@dataclass
class DeployConfig(BaseDeployConfig):
    suite: str = "vlabench"
    num_wrist_images: int = 1
    num_third_person_images: int = 1
    chunk_size: int = 8
    num_open_loop_steps: int = 8


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    raw_base_deploy = getattr(base_deploy, "__wrapped__", base_deploy)
    raw_base_deploy(cfg)


if __name__ == "__main__":
    deploy()