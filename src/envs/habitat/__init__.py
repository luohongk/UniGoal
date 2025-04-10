import os
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
import torch

from habitat.config.default import get_config
from habitat import make_dataset

from .instanceimagegoal_env import InstanceImageGoal_Env

def make_env_fn(args, config_env, rank=0):
    dataset = make_dataset(config_env.habitat.dataset.type, config=config_env.habitat.dataset)
    OmegaConf.set_readonly(config_env, False)
    config_env.habitat.simulator.scene = dataset.episodes[0].scene_id
    OmegaConf.set_readonly(config_env, True)

    env = InstanceImageGoal_Env(args=args, rank=rank,
                            config_env=config_env,
                            dataset=dataset
                            )

    env.seed(rank)
    return env


def construct_envs(args):
    basic_config = get_config(config_path="src/envs/habitat/configs/"
                                         + args.task_config)
    OmegaConf.set_readonly(basic_config, False)
    basic_config.habitat.dataset.split = args.split

    OmegaConf.set_readonly(basic_config, True)

    dataset = make_dataset(basic_config.habitat.dataset.type)
    scenes = basic_config.habitat.dataset.content_scenes
    if "*" in basic_config.habitat.dataset.content_scenes:
        scenes = dataset.get_scenes_to_load(basic_config.habitat.dataset)



    if len(scenes) > 0:
        assert len(scenes) >= args.num_processes, (
            "reduce the number of processes as there "
            "aren't enough number of scenes"
        )

        scene_split_sizes = [int(np.floor(len(scenes) / args.num_processes))
                             for _ in range(args.num_processes)]
        for i in range(len(scenes) % args.num_processes):
            scene_split_sizes[i] += 1

    env_idx = 0
    config_env = get_config(config_path="src/envs/habitat/configs/"
                                        + args.task_config)
    OmegaConf.set_readonly(config_env, False)

    if len(scenes) > 0:
        contentss = scenes[
            sum(scene_split_sizes[:env_idx]):
            sum(scene_split_sizes[:env_idx + 1])
        ]

        bad_scense = []
        good_scense = [contentss[i] for i in range(len(contentss)) if contentss[i] not in bad_scense]

        config_env.habitat.dataset.content_scenes = good_scense

    gpu_id = 0

    config_env.habitat.simulator.habitat_sim_v0.gpu_device_id = gpu_id

    config_env.habitat.environment.iterator_options.shuffle = False

    config_env.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width = args.env_frame_width
    config_env.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height = args.env_frame_height
    config_env.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov = args.hfov
    config_env.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.position = [0, args.camera_height, 0]

    config_env.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width = args.env_frame_width
    config_env.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height = args.env_frame_height
    config_env.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov = args.hfov
    config_env.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.min_depth = args.min_depth
    config_env.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth = args.max_depth
    config_env.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.position = [0, args.camera_height, 0]

    config_env.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor.width = args.env_frame_width
    config_env.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor.height = args.env_frame_height
    config_env.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor.hfov = args.hfov
    config_env.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor.position = [0, args.camera_height, 0]

    config_env.habitat.simulator.agents.main_agent.height = args.camera_height
    config_env.habitat.simulator.turn_angle = args.turn_angle

    config_env.habitat.dataset.split = args.split

    envs = make_env_fn(args, config_env)

    return envs
