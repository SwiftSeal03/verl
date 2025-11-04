# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Split GPU Placement

This module extends the standard PPO trainer to support split GPU placement,
where actor/rollout/ref models use one set of GPUs and critic uses another.

Split Configuration:
- Actor/Rollout/Ref: First half of GPUs (e.g., GPUs 0-1 in a 4-GPU setup)
- Critic: Second half of GPUs (e.g., GPUs 2-3 in a 4-GPU setup)
"""

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer import main_ppo
from verl.utils.device import is_cuda_available


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Main entry point for PPO training with split GPU placement.

    Args:
        config: Hydra configuration dictionary containing training parameters.
    """
    run_ppo_split(config)


def run_ppo_split(config) -> None:
    """Initialize Ray cluster and run distributed PPO training with split placement.

    Args:
        config: Training configuration object containing all necessary parameters.
    """
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    # Create a remote instance of TaskRunnerSplit instead of TaskRunner
    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = TaskRunnerSplit.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunnerSplit.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration
    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)
class TaskRunnerSplit(main_ppo.TaskRunner.__ray_actor_class__):
    """TaskRunner with split GPU placement for actor and critic.

    Inherits from the unwrapped TaskRunner class and only overrides init_resource_pool_mgr
    to create separate resource pools for actor/rollout/ref and critic models.
    """

    def init_resource_pool_mgr(self, config):
        """Initialize resource pool manager with separate pools for actor and critic.

        Creates two resource pools:
        - actor_rollout_ref_pool: For Actor, Rollout, and Reference model
        - critic_pool: For Critic model

        Args:
            config: Training configuration object

        Returns:
            ResourcePoolManager with separate pools for actor and critic
        """
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        n_gpus_per_node = config.trainer.n_gpus_per_node
        nnodes = config.trainer.nnodes

        # Split GPUs: first half for actor/rollout/ref, second half for critic
        actor_rollout_ref_pool_id = "actor_rollout_ref_pool"
        critic_pool_id = "critic_pool"

        # Calculate GPU allocation
        if n_gpus_per_node % 2 == 0:
            # Even number of GPUs per node - split evenly
            gpus_per_pool = n_gpus_per_node // 2
            resource_pool_spec = {
                actor_rollout_ref_pool_id: [gpus_per_pool] * nnodes,
                critic_pool_id: [gpus_per_pool] * nnodes,
            }
        else:
            # Odd number of GPUs - give extra GPU to actor pool
            actor_gpus = (n_gpus_per_node + 1) // 2
            critic_gpus = n_gpus_per_node // 2
            resource_pool_spec = {
                actor_rollout_ref_pool_id: [actor_gpus] * nnodes,
                critic_pool_id: [critic_gpus] * nnodes,
            }

        print(f"Split GPU placement - resource_pool_spec: {resource_pool_spec}")
        print(f"  - Actor/Rollout/Ref pool: {resource_pool_spec[actor_rollout_ref_pool_id]}")
        print(f"  - Critic pool: {resource_pool_spec[critic_pool_id]}")

        # Map roles to resource pools
        self.mapping[Role.ActorRollout] = actor_rollout_ref_pool_id
        self.mapping[Role.Critic] = critic_pool_id

        # Handle reward model placement
        if config.reward_model.enable_resource_pool:
            if config.reward_model.n_gpus_per_node <= 0:
                raise ValueError("config.reward_model.n_gpus_per_node must be greater than 0")
            if config.reward_model.nnodes <= 0:
                raise ValueError("config.reward_model.nnodes must be greater than 0")

            # Create separate pool for reward model
            reward_pool = [config.reward_model.n_gpus_per_node] * config.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=self.mapping
        )

        return resource_pool_manager


if __name__ == "__main__":
    main()
