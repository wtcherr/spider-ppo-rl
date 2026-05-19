__credits__ = ["wtcherr"]

import numpy as np
from ant_v5 import AntEnv

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class SpiderEnv(AntEnv):
    r"""
    ## Description
    This environment is based on ant_v5[AntEnv] the one implemented in Gymnasium library Mujoco environments in [Gymnasium[mujoco][Ant]](https://gymnasium.farama.org/environments/mujoco/ant/).
    The spider is a 3D quadruped robot consisting of a torso (free rotational body) with four legs attached to it, where each leg has three body parts.
    The goal is to coordinate the four legs to move in the forward (right) direction by applying torque to the four balls and eight hinges connecting the three body parts of each leg and the torso (thirteen body parts, four balls and eight hinges).

    Note: Although the robot is called "Spider", it is actually 75cm tall and weighs 1239.76g, with the torso being 620.52g and each leg being 154.81g.

    ## Action Space
    ```{figure} action_space_figures/ant.png
    :name: ant
    ```

    The action space is a `Box(-1, 1, (8,), float32)`. An action represents the torques applied at the hinge joints.

    | Num | Action                                                                | Control Min | Control Max | Name            | Joint | Type (Unit)  |
    | --- | --------------------------------------------------------------------- | ----------- | ----------- | --------------- | ----- | ------------ |
    | 0   | Torque applied on the rotor between the torso and front left hip      | -1          | 1           | FL_Leg          | ball  | torque (N m) |
    | 1   | Torque applied on the rotor between the front left 1st and 2nd links  | -1          | 1           | FL_Leg_001      | hinge | torque (N m) |
    | 2   | Torque applied on the rotor between the front left 2nd and 3rd links  | -1          | 1           | FL_Leg_002      | hinge | torque (N m) |
    | 3   | Torque applied on the rotor between the torso and front right hip     | -1          | 1           | FR_Leg          | ball  | torque (N m) |
    | 4   | Torque applied on the rotor between the front left 1st and 2nd links  | -1          | 1           | FR_Leg_001      | hinge | torque (N m) |
    | 5   | Torque applied on the rotor between the front left 2nd and 3rd links  | -1          | 1           | FR_Leg_002      | hinge | torque (N m) |
    | 6   | Torque applied on the rotor between the torso and back right hip      | -1          | 1           | BR_Leg          | ball  | torque (N m) |
    | 7   | Torque applied on the rotor between the back right 1st and 2nd links  | -1          | 1           | BR_Leg_001      | hinge | torque (N m) |
    | 8   | Torque applied on the rotor between the back right 2nd and 3rd links  | -1          | 1           | BR_Leg_002      | hinge | torque (N m) |
    | 9   | Torque applied on the rotor between the torso and back left hip       | -1          | 1           | BL_Leg          | ball  | torque (N m) |
    | 10  | Torque applied on the rotor between the back left 1st and 2nd links   | -1          | 1           | BL_Leg_001      | hinge | torque (N m) |
    | 11  | Torque applied on the rotor between the back left 2nd and 3rd links   | -1          | 1           | BL_Leg_002      | hinge | torque (N m) |


    ## Observation Space
    The observation space consists of the following parts (in order):

    - *qpos (31 elements by default):* Position values of the robot's body parts 13 * 3 - 2 (*nbody * 3 - root.xy).
    - *qvel (26 elements):* The velocities of these individual body parts (their derivatives).
    - *cfrc_ext (78 elements):* This is the center of mass based external forces on the body parts.
    It has shape 13 * 6 (*nbody * 6*) and hence adds another 78 elements to the state space.
    (external forces - force x, y, z and torque x, y, z)

    By default, the observation does not include the x- and y-coordinates of the torso.
    These can be included by passing `exclude_current_positions_from_observation=False` during construction.
    In this case, the observation space will be a `Box(-Inf, Inf, (135,), float64)`, where the first two observations are the x- and y-coordinates of the torso.
    Regardless of whether `exclude_current_positions_from_observation` is set to `True` or `False`, the x- and y-coordinates are returned in `info` with the keys `"x_position"` and `"y_position"`, respectively.

    By default, however, the observation space is a `Box(-Inf, Inf, (133,), float64)`, where the position and velocity elements are as follows:

    | Num | Observation                                                     | Min      | Max      | Name                             | Joint | Type (Unit)                    |
    |-----|-----------------------------------------------------------------|----------|----------|----------------------------------|-------|--------------------------------|
    | 0   | z-component of the torso
    | 1   | w-component quaternion of torso                                 | -Inf     | Inf      | Torso                            | free  | quaternion                     |
    | 2   | x-component quaternion of torso                                 | -Inf     | Inf      | Torso                            | free  | quaternion                     |
    | 3   | y-component quaternion of torso                                 | -Inf     | Inf      | Torso                            | free  | quaternion                     |
    | 4   | z-component quaternion of torso                                 | -Inf     | Inf      | Torso                            | free  | quaternion                     |

    | 5   | w-component front-left hip ball joint quaternion                | -Inf     | Inf      | FL_Leg_Joint                     | ball  | quaternion                     |
    | 6   | x-component front-left hip ball joint quaternion                | -Inf     | Inf      | FL_Leg_Joint                     | ball  | quaternion                     |
    | 7   | y-component front-left hip ball joint quaternion                | -Inf     | Inf      | FL_Leg_Joint                     | ball  | quaternion                     |
    | 8   | z-component front-left hip ball joint quaternion                | -Inf     | Inf      | FL_Leg_Joint                     | ball  | quaternion                     |
    | 9   | angle between front-left upper and middle leg links             | -Inf     | Inf      | FL_Leg_001_Joint                 | hinge | angle (rad)                    |
    | 10  | angle between front-left middle and lower leg links             | -Inf     | Inf      | FL_Leg_002_Joint                 | hinge | angle (rad)                    |

    | 11  | w-component front-right hip ball joint quaternion               | -Inf     | Inf      | FR_Leg_Joint                     | ball  | quaternion                     |
    | 12  | x-component front-right hip ball joint quaternion               | -Inf     | Inf      | FR_Leg_Joint                     | ball  | quaternion                     |
    | 13  | y-component front-right hip ball joint quaternion               | -Inf     | Inf      | FR_Leg_Joint                     | ball  | quaternion                     |
    | 14  | z-component front-right hip ball joint quaternion               | -Inf     | Inf      | FR_Leg_Joint                     | ball  | quaternion                     |
    | 15  | angle between front-right upper and middle leg links            | -Inf     | Inf      | FR_Leg_001_Joint                 | hinge | angle (rad)                    |
    | 16  | angle between front-right middle and lower leg links            | -Inf     | Inf      | FR_Leg_002_Joint                 | hinge | angle (rad)                    |

    | 17  | w-component back-right hip ball joint quaternion                | -Inf     | Inf      | BR_Leg_Joint                     | ball  | quaternion                     |
    | 18  | x-component back-right hip ball joint quaternion                | -Inf     | Inf      | BR_Leg_Joint                     | ball  | quaternion                     |
    | 19  | y-component back-right hip ball joint quaternion                | -Inf     | Inf      | BR_Leg_Joint                     | ball  | quaternion                     |
    | 20  | z-component back-right hip ball joint quaternion                | -Inf     | Inf      | BR_Leg_Joint                     | ball  | quaternion                     |
    | 21  | angle between back-right upper and middle leg links             | -Inf     | Inf      | BR_Leg_001_Joint                 | hinge | angle (rad)                    |
    | 22  | angle between back-right middle and lower leg links             | -Inf     | Inf      | BR_Leg_002_Joint                 | hinge | angle (rad)                    |

    | 23  | w-component back-left hip ball joint quaternion                 | -Inf     | Inf      | BL_Leg_Joint                     | ball  | quaternion                     |
    | 24  | x-component back-left hip ball joint quaternion                 | -Inf     | Inf      | BL_Leg_Joint                     | ball  | quaternion                     |
    | 25  | y-component back-left hip ball joint quaternion                 | -Inf     | Inf      | BL_Leg_Joint                     | ball  | quaternion                     |
    | 26  | z-component back-left hip ball joint quaternion                 | -Inf     | Inf      | BL_Leg_Joint                     | ball  | quaternion                     |
    | 27  | angle between back-left upper and middle leg links              | -Inf     | Inf      | BL_Leg_001_Joint                 | hinge | angle (rad)                    |
    | 28  | angle between back-left middle and lower leg links              | -Inf     | Inf      | BL_Leg_002_Joint                 | hinge | angle (rad)                    |

    | 29  | x-component velocity of torso                                   | -Inf     | Inf      | Torso                            | free  | linear velocity (m/s)          |
    | 30  | y-component velocity of torso                                   | -Inf     | Inf      | Torso                            | free  | linear velocity (m/s)          |
    | 31  | z-component velocity of torso                                   | -Inf     | Inf      | Torso                            | free  | linear velocity (m/s)          |

    | 32  | x-axis angular velocity of torso                                | -Inf     | Inf      | Torso                            | free  | angular velocity (rad/s)       |
    | 33  | y-axis angular velocity of torso                                | -Inf     | Inf      | Torso                            | free  | angular velocity (rad/s)       |
    | 34  | z-axis angular velocity of torso                                | -Inf     | Inf      | Torso                            | free  | angular velocity (rad/s)       |

    | 35  | x-component front-left hip joint angular velocity               | -Inf     | Inf      | FL_Leg_Joint                     | ball  | angular velocity (rad/s)       |
    | 36  | y-component front-left hip joint angular velocity               | -Inf     | Inf      | FL_Leg_Joint                     | ball  | angular velocity (rad/s)       |
    | 37  | z-component front-left hip joint angular velocity               | -Inf     | Inf      | FL_Leg_Joint                     | ball  | angular velocity (rad/s)       |
    | 38  | angular velocity of front-left middle joint                     | -Inf     | Inf      | FL_Leg_001_Joint                 | hinge | angular velocity (rad/s)       |
    | 39  | angular velocity of front-left lower joint                      | -Inf     | Inf      | FL_Leg_002_Joint                 | hinge | angular velocity (rad/s)       |

    | 40  | x-component front-right hip joint angular velocity              | -Inf     | Inf      | FR_Leg_Joint                     | ball  | angular velocity (rad/s)       |
    | 41  | y-component front-right hip joint angular velocity              | -Inf     | Inf      | FR_Leg_Joint                     | ball  | angular velocity (rad/s)       |
    | 42  | z-component front-right hip joint angular velocity              | -Inf     | Inf      | FR_Leg_Joint                     | ball  | angular velocity (rad/s)       |
    | 43  | angular velocity of front-right middle joint                    | -Inf     | Inf      | FR_Leg_001_Joint                 | hinge | angular velocity (rad/s)       |
    | 44  | angular velocity of front-right lower joint                     | -Inf     | Inf      | FR_Leg_002_Joint                 | hinge | angular velocity (rad/s)       |

    | 45  | x-component back-right hip joint angular velocity               | -Inf     | Inf      | BR_Leg_Joint                     | ball  | angular velocity (rad/s)       |
    | 46  | y-component back-right hip joint angular velocity               | -Inf     | Inf      | BR_Leg_Joint                     | ball  | angular velocity (rad/s)       |
    | 47  | z-component back-right hip joint angular velocity               | -Inf     | Inf      | BR_Leg_Joint                     | ball  | angular velocity (rad/s)       |
    | 48  | angular velocity of back-right middle joint                     | -Inf     | Inf      | BR_Leg_001_Joint                 | hinge | angular velocity (rad/s)       |
    | 49  | angular velocity of back-right lower joint                      | -Inf     | Inf      | BR_Leg_002_Joint                 | hinge | angular velocity (rad/s)       |

    | 50  | x-component back-left hip joint angular velocity                | -Inf     | Inf      | BL_Leg_Joint                     | ball  | angular velocity (rad/s)       |
    | 51  | y-component back-left hip joint angular velocity                | -Inf     | Inf      | BL_Leg_Joint                     | ball  | angular velocity (rad/s)       |
    | 52  | z-component back-left hip joint angular velocity                | -Inf     | Inf      | BL_Leg_Joint                     | ball  | angular velocity (rad/s)       |
    | 53  | angular velocity of back-left middle joint                      | -Inf     | Inf      | BL_Leg_001_Joint                 | hinge | angular velocity (rad/s)       |
    | 54  | angular velocity of back-left lower joint                       | -Inf     | Inf      | BL_Leg_002_Joint                 | hinge | angular velocity (rad/s)       |

    | excluded | x-component of torso position                             | -Inf     | Inf      | Torso                            | free  | position (m)                   |
    | excluded | y-component of torso position                             | -Inf     | Inf      | Torso                            | free  | position (m)                   |


    The body parts are:
    | Body Part / XML Name                                      | Body ID (`v0`) |
    |------------------------------------------------------------|----------------|
    | worldbody (constant reference body)                        | 0              |
    | Torso (main body / centre body)                            | 1              |

    | FL_Leg (front-left upper leg / hip segment)                | 2              |
    | FL_Leg_001 (front-left middle leg segment)                 | 3              |
    | FL_Leg_002 (front-left lower leg / foot segment)           | 4              |

    | FR_Leg (front-right upper leg / hip segment)               | 5              |
    | FR_Leg_001 (front-right middle leg segment)                | 6              |
    | FR_Leg_002 (front-right lower leg / foot segment)          | 7              |

    | BR_Leg (back-right upper leg / hip segment)                | 8              |
    | BR_Leg_001 (back-right middle leg segment)                 | 9              |
    | BR_Leg_002 (back-right lower leg / foot segment)           | 10             |

    | BL_Leg (back-left upper leg / hip segment)                 | 11             |
    | BL_Leg_001 (back-left middle leg segment)                  | 12             |
    | BL_Leg_002 (back-left lower leg / foot segment)            | 13             |


    The (x,y,z) coordinates are translational DOFs, while the orientations are rotational DOFs expressed as quaternions.
    One can read more about free joints in the [MuJoCo documentation](https://mujoco.readthedocs.io/en/latest/XMLreference.html).


    ## Rewards
    The total reward is ***reward*** *=* *healthy_reward + forward_reward - ctrl_cost - contact_cost - z_orientation_cost*.

    - *healthy_reward*:
    Every timestep that the Spider is healthy (see definition in section "Episode End"),
    it gets a reward of fixed value `healthy_reward` (default is $1$).
    - *forward_reward*:
    A reward for moving forward,
    this reward would be positive if the Spider moves forward (in the positive $x$ direction / in the right direction).
    $w_{forward} \times \frac{dx}{dt}$, where
    $dx$ is the displacement of the `main_body` ($x_{after-action} - x_{before-action}$),
    $dt$ is the time between actions, which depends on the `frame_skip` parameter (default is $5$),
    and `frametime`, which is $0.01$ - so the default is $dt = 5 \times 0.01 = 0.05$,
    $w_{forward}$ is the `forward_reward_weight` (default is $1$).
    - *ctrl_cost*:
    A negative reward to penalize the Spider for taking actions that are too large.
    $w_{control} \times \|action\|_2^2$,
    where $w_{control}$ is `ctrl_cost_weight` (default is $0.5$).
    - *contact_cost*:
    A negative reward to penalize the Spider if the external contact forces are too large.
    $w_{contact} \times \|F_{contact}\|_2^2$, where
    $w_{contact}$ is `contact_cost_weight` (default is $5\times10^{-4}$),
    $F_{contact}$ are the external contact forces clipped by `contact_force_range` (see `cfrc_ext` section on Observation Space).
    - *z_orientation_cost*:
    A negative reward to penalize the Spider if the local z-axis of the torso changes too much between steps.
    //TODO add info

    `info` contains the individual reward terms.

    The total reward returned is ***reward*** *=* *healthy_reward + forward_reward - ctrl_cost*.





    ## Episode End
    ### Termination
    If `terminate_when_unhealthy is True` (the default), the environment terminates when the Spider is unhealthy.
    the Spider is unhealthy if any of the following happens:

    1. Any of the state space values is no longer finite.
    2. The z-coordinate of the torso (the height) is **not** in the closed interval given by the `healthy_z_range` argument (default is $[0.2, 1.0]$).
    3. The z-component of the local z-axis of the torso (the uprightness) is less than 0.

    ### Truncation
    The default duration of an episode is 1000 timesteps.


    ## Arguments
    Spider provides a range of parameters to modify the observation space, reward function, initial state, and termination condition.
    These parameters can be applied during `gymnasium.make` after registering the environment first in the following way:

    ```python
    import gymnasium as gym
    gym.register(id="mujoco_env/Spider-v0", entry_point=SpiderEnv, max_episode_steps=1000)
    env = gym.make('spider_v0', ctrl_cost_weight=0.5, ...)
    ```

    | Parameter                                  | Type       | Default        |Description                    |
    |--------------------------------------------|------------|----------------|-------------------------------|
    |`xml_file`                                  | **str**    | `"scene.xml"`  | Path to a MuJoCo model                                                                                                                                                                                      |
    |`forward_reward_weight`                     | **float**  | `5`            | Weight for _forward_reward_ term (see `Rewards` section)                                                                                                                                                    |
    |`ctrl_cost_weight`                          | **float**  | `0.035`          | Weight for _ctrl_cost_ term (see `Rewards` section)                                                                                                                                                         |
    |`contact_cost_weight`                       | **float**  | `5e-4`         | Weight for _contact_cost_ term (see `Rewards` section)                                                                                                                                                      |
    |`z_orientation_cost_weight`                 | **float**  | `30`           | Weight for _z_orientation_cost_ term (see `Rewards` section)                                                                                                                                                |
    |`healthy_reward`                            | **float**  | `1`            | Weight for _healthy_reward_ term (see `Rewards` section)                                                                                                                                                    |
    |`main_body`                                 |**str\|int**| `1`("Torso")   | Name or ID of the body, whose displacement is used to calculate the *dx*/_forward_reward_ (useful for custom MuJoCo models) (see `Rewards` section)                                                         |
    |`terminate_when_unhealthy`                  | **bool**   | `True`         | If `True`, issue a `terminated` signal is unhealthy (see `Episode End` section)                                                                                                                             |
    |`healthy_z_range`                           | **tuple**  | `(-0.05, 0.5)` | The spider is considered healthy if the z-coordinate of the torso is in this range (see `Episode End` section)                                                                                              |
    |`contact_force_range`                       | **tuple**  | `(-1, 1)`      | Contact forces are clipped to this range in the computation of *contact_cost* (see `Rewards` section)                                                                                                       |
    |`reset_noise_scale`                         | **float**  | `0.1`          | Scale of random perturbations of initial position and velocity (see `Starting State` section)                                                                                                               |
    |`exclude_current_positions_from_observation`| **bool**   | `True`         | Whether or not to omit the x- and y-coordinates from observations. Excluding the position can serve as an inductive bias to induce position-agnostic behavior in policies (see `Observation State` section) |
    |`include_cfrc_ext_in_observation`           | **bool**   | `True`         | Whether to include *cfrc_ext* elements in the observations (see `Observation State` section)                                                                                                                |

    ## Version History
    * v0: Initial versions release
    """

    def __init__(
        self,
        xml_file: str = "scene.xml",
        frame_skip: int = 5,
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 5.0,
        ctrl_cost_weight: float = 0.035,
        contact_cost_weight: float = 5e-4,
        z_orientation_cost_weight: float = 30.0,
        healthy_reward: float = 1.0,
        main_body: int | str = 1,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: tuple[float, float] = (-0.05, 0.5),
        contact_force_range: tuple[float, float] = (-1.0, 1.0),
        reset_noise_scale: float = 0.1,
        exclude_current_positions_from_observation: bool = True,
        include_cfrc_ext_in_observation: bool = True,
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
