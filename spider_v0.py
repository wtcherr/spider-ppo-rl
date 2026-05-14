__credits__ = ["wtcherr"]

import mujoco
import numpy as np
from ant_v5 import AntEnv

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class SpiderEnv(AntEnv):
    r""" """

    def __init__(
        self,
        xml_file: str = "scene.xml",
        frame_skip: int = 5,
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 5.0,
        ctrl_cost_weight: float = 0.03,
        contact_cost_weight: float = 5e-4,
        z_orientation_cost_weight: float = 30.0,
        healthy_reward: float = 1.0,
        main_body: int | str = 1,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: tuple[float, float] = (-0.05, 0.5),
        contact_force_range: tuple[float, float] = (-1.0, 1.0),
        reset_noise_scale: float = 0.1,
        exclude_current_positions_from_observation: bool = True,
        include_cfrc_ext_in_observation: bool = False,
        **kwargs,
    ):
        super().__init__(
            xml_file,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            contact_cost_weight,
            healthy_reward,
            main_body,
            terminate_when_unhealthy,
            healthy_z_range,
            contact_force_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            include_cfrc_ext_in_observation,
            **kwargs,
        )
        self._z_orientation_cost_weight = z_orientation_cost_weight

    @property
    def z_orientation(self):
        return self.data.body(self._main_body).xmat.reshape(3, 3)[:, 2].copy()

    @property
    def is_flipped(self):
        is_flipped = self.z_orientation[2] < 0
        return is_flipped

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = (
            np.isfinite(state).all()
            and min_z <= state[2] <= max_z
            and not self.is_flipped
        )
        return is_healthy

    def step(self, action):
        xy_position_before = self.data.body(self._main_body).xpos[:2].copy()
        z_orientation_before = self.z_orientation
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.data.body(self._main_body).xpos[:2].copy()
        z_orientation_after = self.z_orientation

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        z_orientation_similarity = np.dot(z_orientation_before, z_orientation_after)

        observation = self._get_obs()
        reward, reward_info = self._get_rew(
            x_velocity, z_orientation_similarity, action
        )
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "z_orientation": self.z_orientation,
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "z_orientation_similarity": z_orientation_similarity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def _get_rew(self, x_velocity: float, z_orientation_similarity: float, action):
        forward_reward = x_velocity * self._forward_reward_weight
        healthy_reward = self.healthy_reward
        rewards = forward_reward + healthy_reward

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost
        z_orientation_cost = (
            1 - z_orientation_similarity
        ) * self._z_orientation_cost_weight

        costs = ctrl_cost + contact_cost + z_orientation_cost

        reward = rewards - costs

        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_z_orientation": -z_orientation_cost,
            "reward_survive": healthy_reward,
        }

        return reward, reward_info
