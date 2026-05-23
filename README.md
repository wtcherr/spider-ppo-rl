# Quadrupedal Spider PPO RL (Spider Robot)

This project implements a Reinforcement Learning environment and training pipeline for a quadrupedal "Spider" robot using MuJoCo and Gymnasium. It features a custom PPO (Proximal Policy Optimization) implementation to train the robot to walk forward.

## Preview

|                                 Top View                                 |                                       Top View (Close-up)                                        |
| :----------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------: |
|  [![Top Walk](videos/spider_walk_top.gif)](videos/spider_walk_top.mp4)   |  [![Top Walk Closeup](videos/spider_walk_top_closeup.gif)](videos/spider_walk_top_closeup.mp4)   |
|                              **Side View**                               |                                     **Side View (Close-up)**                                     |
| [![Side Walk](videos/spider_walk_side.gif)](videos/spider_walk_side.mp4) | [![Side Walk Closeup](videos/spider_walk_side_closeup.gif)](videos/spider_walk_side_closeup.mp4) |

---

## Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Environment Details: Spider-v0](#-environment-details-spider-v0)
- [Development Process](#-development-process)
- [Training & Implementation](#-training--implementation)
- [Results](#-results)
- [Future Work](#-future-work)
- [References & Credits](#-references--credits)

---

## Features

- **Custom MuJoCo Environment**: `SpiderEnv` (in `spider_v0.py`), based on Gymnasium's `AntEnv`, tailored for a spider-like robot model.
- **PPO Training**: A robust implementation of Proximal Policy Optimization in `ppo_spider.py`.
- **Visualization**: Support for real-time MuJoCo visualization and training progress tracking.
- **Tracking**: Integrated with Tensorboard and Weights & Biases (W&B) for experiment tracking.
- **Assets**: Using high-quality 3D models and textures for the spider robot.

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU with 2GB+ memory (recommended)
- 8GB+ RAM
- [MuJoCo](https://github.com/google-deepmind/mujoco)
- [PyTorch](https://pytorch.org/)

### Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/spider-ppo-rl.git
   cd spider-ppo-rl
   ```

2. **Create a virtual environment and install dependencies:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

   _Note: Install PyTorch with the appropriate CUDA version from [pytorch.org](https://pytorch.org/)._

---

## 🕷 Environment Details: `Spider-v0`

### Description

This environment is based on `Ant-v5` from the Gymnasium library. The spider is a 3D quadruped robot consisting of a torso (free rotational body) with four legs. Each leg has three segments connected by a ball joint (hip) and two hinge joints (middle/lower).

- **Robot Stats**: 20cm tall, 1239.76g total weight (Torso: 620.52g, Legs: 154.81g each).
- **Goal**: Coordinate the four legs to move in the forward ($+x$) direction.

### Action Space

The action space is a `Box(-1, 1, (12,), float32)`. Actions represent normalized torques applied to the joints.

| Num | Name       | Joint | Type         | Description              |
| :-- | :--------- | :---- | :----------- | :----------------------- |
| 0   | FL_Leg     | Ball  | torque (N m) | Front Left Hip           |
| 1   | FL_Leg_001 | Hinge | torque (N m) | Front Left Middle Joint  |
| 2   | FL_Leg_002 | Hinge | torque (N m) | Front Left Lower Joint   |
| 3   | FR_Leg     | Ball  | torque (N m) | Front Right Hip          |
| 4   | FR_Leg_001 | Hinge | torque (N m) | Front Right Middle Joint |
| 5   | FR_Leg_002 | Hinge | torque (N m) | Front Right Lower Joint  |
| 6   | BR_Leg     | Ball  | torque (N m) | Back Right Hip           |
| 7   | BR_Leg_001 | Hinge | torque (N m) | Back Right Middle Joint  |
| 8   | BR_Leg_002 | Hinge | torque (N m) | Back Right Lower Joint   |
| 9   | BL_Leg     | Ball  | torque (N m) | Back Left Hip            |
| 10  | BL_Leg_001 | Hinge | torque (N m) | Back Left Middle Joint   |
| 11  | BL_Leg_002 | Hinge | torque (N m) | Back Left Lower Joint    |

### Observation Space

The observation space varies based on configuration, ranging from 55 to 135 elements:

- **qpos (31 elements)**: Positions/orientations of the 13 body parts.
- **qvel (26 elements)**: Linear and angular velocities of body parts.
- **cfrc_ext (78 elements)**: Center-of-mass based external forces (optional).

_By default, x/y torso positions are excluded to encourage position-agnostic behavior._

These can be included by passing `exclude_current_positions_from_observation=False` during construction.
In this case, the observation space will be a `Box(-Inf, Inf, (135,), float64)`, where the first two observations are the x- and y-coordinates of the torso.
Regardless of whether `exclude_current_positions_from_observation` is set to `True` or `False`, the x- and y-coordinates are returned in `info` with the keys `"x_position"` and `"y_position"`, respectively.

By default, however, the observation space is a `Box(-Inf, Inf, (133,), float64)`, where the position and velocity elements are as follows:

| Num      | Observation                                          | Min  | Max | Name             | Joint | Type (Unit)              |
| -------- | ---------------------------------------------------- | ---- | --- | ---------------- | ----- | ------------------------ |
| 1        | w-component quaternion of torso                      | -Inf | Inf | Torso            | free  | quaternion               |
| 2        | x-component quaternion of torso                      | -Inf | Inf | Torso            | free  | quaternion               |
| 3        | y-component quaternion of torso                      | -Inf | Inf | Torso            | free  | quaternion               |
| 4        | z-component quaternion of torso                      | -Inf | Inf | Torso            | free  | quaternion               |
| 5        | w-component front-left hip ball joint quaternion     | -Inf | Inf | FL_Leg_Joint     | ball  | quaternion               |
| 6        | x-component front-left hip ball joint quaternion     | -Inf | Inf | FL_Leg_Joint     | ball  | quaternion               |
| 7        | y-component front-left hip ball joint quaternion     | -Inf | Inf | FL_Leg_Joint     | ball  | quaternion               |
| 8        | z-component front-left hip ball joint quaternion     | -Inf | Inf | FL_Leg_Joint     | ball  | quaternion               |
| 9        | angle between front-left upper and middle leg links  | -Inf | Inf | FL_Leg_001_Joint | hinge | angle (rad)              |
| 10       | angle between front-left middle and lower leg links  | -Inf | Inf | FL_Leg_002_Joint | hinge | angle (rad)              |
| 11       | w-component front-right hip ball joint quaternion    | -Inf | Inf | FR_Leg_Joint     | ball  | quaternion               |
| 12       | x-component front-right hip ball joint quaternion    | -Inf | Inf | FR_Leg_Joint     | ball  | quaternion               |
| 13       | y-component front-right hip ball joint quaternion    | -Inf | Inf | FR_Leg_Joint     | ball  | quaternion               |
| 14       | z-component front-right hip ball joint quaternion    | -Inf | Inf | FR_Leg_Joint     | ball  | quaternion               |
| 15       | angle between front-right upper and middle leg links | -Inf | Inf | FR_Leg_001_Joint | hinge | angle (rad)              |
| 16       | angle between front-right middle and lower leg links | -Inf | Inf | FR_Leg_002_Joint | hinge | angle (rad)              |
| 17       | w-component back-right hip ball joint quaternion     | -Inf | Inf | BR_Leg_Joint     | ball  | quaternion               |
| 18       | x-component back-right hip ball joint quaternion     | -Inf | Inf | BR_Leg_Joint     | ball  | quaternion               |
| 19       | y-component back-right hip ball joint quaternion     | -Inf | Inf | BR_Leg_Joint     | ball  | quaternion               |
| 20       | z-component back-right hip ball joint quaternion     | -Inf | Inf | BR_Leg_Joint     | ball  | quaternion               |
| 21       | angle between back-right upper and middle leg links  | -Inf | Inf | BR_Leg_001_Joint | hinge | angle (rad)              |
| 22       | angle between back-right middle and lower leg links  | -Inf | Inf | BR_Leg_002_Joint | hinge | angle (rad)              |
| 23       | w-component back-left hip ball joint quaternion      | -Inf | Inf | BL_Leg_Joint     | ball  | quaternion               |
| 24       | x-component back-left hip ball joint quaternion      | -Inf | Inf | BL_Leg_Joint     | ball  | quaternion               |
| 25       | y-component back-left hip ball joint quaternion      | -Inf | Inf | BL_Leg_Joint     | ball  | quaternion               |
| 26       | z-component back-left hip ball joint quaternion      | -Inf | Inf | BL_Leg_Joint     | ball  | quaternion               |
| 27       | angle between back-left upper and middle leg links   | -Inf | Inf | BL_Leg_001_Joint | hinge | angle (rad)              |
| 28       | angle between back-left middle and lower leg links   | -Inf | Inf | BL_Leg_002_Joint | hinge | angle (rad)              |
| 29       | x-component velocity of torso                        | -Inf | Inf | Torso            | free  | linear velocity (m/s)    |
| 30       | y-component velocity of torso                        | -Inf | Inf | Torso            | free  | linear velocity (m/s)    |
| 31       | z-component velocity of torso                        | -Inf | Inf | Torso            | free  | linear velocity (m/s)    |
| 32       | x-axis angular velocity of torso                     | -Inf | Inf | Torso            | free  | angular velocity (rad/s) |
| 33       | y-axis angular velocity of torso                     | -Inf | Inf | Torso            | free  | angular velocity (rad/s) |
| 34       | z-axis angular velocity of torso                     | -Inf | Inf | Torso            | free  | angular velocity (rad/s) |
| 35       | x-component front-left hip joint angular velocity    | -Inf | Inf | FL_Leg_Joint     | ball  | angular velocity (rad/s) |
| 36       | y-component front-left hip joint angular velocity    | -Inf | Inf | FL_Leg_Joint     | ball  | angular velocity (rad/s) |
| 37       | z-component front-left hip joint angular velocity    | -Inf | Inf | FL_Leg_Joint     | ball  | angular velocity (rad/s) |
| 38       | angular velocity of front-left middle joint          | -Inf | Inf | FL_Leg_001_Joint | hinge | angular velocity (rad/s) |
| 39       | angular velocity of front-left lower joint           | -Inf | Inf | FL_Leg_002_Joint | hinge | angular velocity (rad/s) |
| 40       | x-component front-right hip joint angular velocity   | -Inf | Inf | FR_Leg_Joint     | ball  | angular velocity (rad/s) |
| 41       | y-component front-right hip joint angular velocity   | -Inf | Inf | FR_Leg_Joint     | ball  | angular velocity (rad/s) |
| 42       | z-component front-right hip joint angular velocity   | -Inf | Inf | FR_Leg_Joint     | ball  | angular velocity (rad/s) |
| 43       | angular velocity of front-right middle joint         | -Inf | Inf | FR_Leg_001_Joint | hinge | angular velocity (rad/s) |
| 44       | angular velocity of front-right lower joint          | -Inf | Inf | FR_Leg_002_Joint | hinge | angular velocity (rad/s) |
| 45       | x-component back-right hip joint angular velocity    | -Inf | Inf | BR_Leg_Joint     | ball  | angular velocity (rad/s) |
| 46       | y-component back-right hip joint angular velocity    | -Inf | Inf | BR_Leg_Joint     | ball  | angular velocity (rad/s) |
| 47       | z-component back-right hip joint angular velocity    | -Inf | Inf | BR_Leg_Joint     | ball  | angular velocity (rad/s) |
| 48       | angular velocity of back-right middle joint          | -Inf | Inf | BR_Leg_001_Joint | hinge | angular velocity (rad/s) |
| 49       | angular velocity of back-right lower joint           | -Inf | Inf | BR_Leg_002_Joint | hinge | angular velocity (rad/s) |
| 50       | x-component back-left hip joint angular velocity     | -Inf | Inf | BL_Leg_Joint     | ball  | angular velocity (rad/s) |
| 51       | y-component back-left hip joint angular velocity     | -Inf | Inf | BL_Leg_Joint     | ball  | angular velocity (rad/s) |
| 52       | z-component back-left hip joint angular velocity     | -Inf | Inf | BL_Leg_Joint     | ball  | angular velocity (rad/s) |
| 53       | angular velocity of back-left middle joint           | -Inf | Inf | BL_Leg_001_Joint | hinge | angular velocity (rad/s) |
| 54       | angular velocity of back-left lower joint            | -Inf | Inf | BL_Leg_002_Joint | hinge | angular velocity (rad/s) |
| excluded | x-component of torso position                        | -Inf | Inf | Torso            | free  | position (m)             |
| excluded | y-component of torso position                        | -Inf | Inf | Torso            | free  | position (m)             |

The body parts are:

| Body Part / XML Name                              | Body ID (`v0`) |
| ------------------------------------------------- | -------------- |
| worldbody (constant reference body)               | 0              |
| Torso (main body / centre body)                   | 1              |
| FL_Leg (front-left upper leg / hip segment)       | 2              |
| FL_Leg_001 (front-left middle leg segment)        | 3              |
| FL_Leg_002 (front-left lower leg / foot segment)  | 4              |
| FR_Leg (front-right upper leg / hip segment)      | 5              |
| FR_Leg_001 (front-right middle leg segment)       | 6              |
| FR_Leg_002 (front-right lower leg / foot segment) | 7              |
| BR_Leg (back-right upper leg / hip segment)       | 8              |
| BR_Leg_001 (back-right middle leg segment)        | 9              |
| BR_Leg_002 (back-right lower leg / foot segment)  | 10             |
| BL_Leg (back-left upper leg / hip segment)        | 11             |
| BL_Leg_001 (back-left middle leg segment)         | 12             |
| BL_Leg_002 (back-left lower leg / foot segment)   | 13             |

The (x,y,z) coordinates are translational DOFs, while the orientations are rotational DOFs expressed as quaternions.
One can read more about free joints in the [MuJoCo documentation](https://mujoco.readthedocs.io/en/latest/XMLreference.html).

### Rewards

The total reward is calculated as:
$Reward = Healthy\_reward + Forward\_reward - Ctrl\_cost - Contact\_cost - Z\_orientation\_cost$

- **Forward Reward**: $w_{forward} \times \frac{dx}{dt}$ (Progress along the x-axis).
- **Healthy Reward**: Fixed reward ($+1.0$) awarded for every step the robot is "healthy".
- **Control Cost**: $w_{control} \times \|action\|_2^2$ (Penalizes excessive joint torque).
- **Contact Cost**: $w_{contact} \times \|F_{contact}\|_2^2$ (Penalizes high impact forces).
- **Z-Orientation Cost**: $w_{z\_orient} \times (1 - \vec{z}_{before} \cdot \vec{z}_{after})$ (Penalizes rapid changes or wobbling in the torso's upright orientation, where $\vec{z}$ is the local z-axis vector).

### Episode End

**Termination (Unhealthy State):**

1. Any state value becomes non-finite (NaN/Inf).
2. Torso height ($z$) falls outside the `healthy_z_range` (default: $[-0.05, 0.5]$).
3. The torso flips over (z-component of local z-axis < 0).

**Truncation:**

- Reaching the maximum step limit (default: 1000).

### Configuration Arguments

Parameters can be modified during `gymnasium.make` after registration:

```python
gym.register(id="mujoco_env/Spider-v0", entry_point=SpiderEnv, max_episode_steps=1000)
env = gym.make('mujoco_env/Spider-v0',
    forward_reward_weight=5.0,
    ctrl_cost_weight=0.035,
    healthy_z_range=(-0.05, 0.5),
    terminate_when_unhealthy=True
)
```

| Parameter                                    | Type         | Default        | Description                                                                                                                                                                                                 |
| -------------------------------------------- | ------------ | -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `xml_file`                                   | **str**      | `"scene.xml"`  | Path to a MuJoCo model                                                                                                                                                                                      |
| `forward_reward_weight`                      | **float**    | `5`            | Weight for _forward_reward_ term (see `Rewards` section)                                                                                                                                                    |
| `ctrl_cost_weight`                           | **float**    | `0.035`        | Weight for _ctrl_cost_ term (see `Rewards` section)                                                                                                                                                         |
| `contact_cost_weight`                        | **float**    | `5e-4`         | Weight for _contact_cost_ term (see `Rewards` section)                                                                                                                                                      |
| `z_orientation_cost_weight`                  | **float**    | `30`           | Weight for _z_orientation_cost_ term (see `Rewards` section)                                                                                                                                                |
| `healthy_reward`                             | **float**    | `1`            | Weight for _healthy_reward_ term (see `Rewards` section)                                                                                                                                                    |
| `main_body`                                  | **str\|int** | `1`("Torso")   | Name or ID of the body, whose displacement is used to calculate the _dx_/_forward_reward_ (useful for custom MuJoCo models) (see `Rewards` section)                                                         |
| `terminate_when_unhealthy`                   | **bool**     | `True`         | If `True`, issue a `terminated` signal is unhealthy (see `Episode End` section)                                                                                                                             |
| `healthy_z_range`                            | **tuple**    | `(-0.05, 0.5)` | The spider is considered healthy if the z-coordinate of the torso is in this range (see `Episode End` section)                                                                                              |
| `contact_force_range`                        | **tuple**    | `(-1, 1)`      | Contact forces are clipped to this range in the computation of _contact_cost_ (see `Rewards` section)                                                                                                       |
| `reset_noise_scale`                          | **float**    | `0.1`          | Scale of random perturbations of initial position and velocity (see `Starting State` section)                                                                                                               |
| `exclude_current_positions_from_observation` | **bool**     | `True`         | Whether or not to omit the x- and y-coordinates from observations. Excluding the position can serve as an inductive bias to induce position-agnostic behavior in policies (see `Observation State` section) |
| `include_cfrc_ext_in_observation`            | **bool**     | `True`         | Whether to include _cfrc_ext_ elements in the observations (see `Observation State` section)                                                                                                                |

## Development Process

1. **3D Modeling & Conversion**:
   - Sourced model from [Unity Asset Store](https://assetstore.unity.com/packages/3d/characters/robots/spider-orange-181154).
   - Segmented in Blender for `obj2mjcf` compatibility.
   - Processed with `obj2mjcf` to generate MJCF XML and meshes.
2. **MuJoCo Model Refinement**:
   - Cleaned XML, established hierarchy, and placed joints based on Blender bone positions.
   - Added actuators and collision meshes.
3. **Environment Integration**:
   - Developed `SpiderEnv` wrapper for Gymnasium.
   - Verified with random agents.
4. **PPO Implementation**:
   - Custom implementation optimized for continuous control.
   - Integrated observation and reward normalization.

---

## Training & Implementation

### Network Architecture

**Actor (Policy Network):**

```
Input (55) → FC(256) → Tanh → FC(256) → Tanh → FC(128) → Tanh → FC(12)
                                                                    ↓
                                                            Action Mean + Log Std
```

- **Action Distribution**: Diagonal Gaussian with learned log standard deviation.

**Critic (Value Network):**

```
Input (55) → FC(256) → Tanh → FC(256) → Tanh → FC(128) → Tanh → FC(1)
                                                                    ↓
                                                               State Value
```

### Key Techniques

- Vectorized Environments (16 parallel envs)
- Observation & Reward Normalization
- Generalized Advantage Estimation (GAE)
- Adam Learning Rate Annealing
- Global Gradient Clipping (0.5)
- Orthogonal Initialization

### Hyperparameters

| Parameter                 | First Run | Second Run (Fine-tuning) |
| :------------------------ | :-------- | :----------------------- |
| **Learning Rate**         | 3e-4      | 1.5e-5                   |
| **Entropy Coeff**         | 0.01      | 0.004                    |
| **Forward Reward Weight** | 5.0       | 7.0                      |
| **Ctrl Cost Weight**      | 0.05      | 0.035                    |
| **Healthy Reward**        | 1.0       | 0.5                      |
| **Motor Gear**            | 10        | 16                       |
| **Z-Orientation Penalty** | None      | 30.0                     |
| **Total Timesteps**       | 8M        | +8M                      |

---

## Results

### First Training Run: Stability & Survival

The initial run focused on survival. The model learned a stable, symmetric gait but moved slowly.

| Metric              |                              Chart                               |
| :------------------ | :--------------------------------------------------------------: |
| **Episodic Return** |     ![Episodic Return](charts/first-run/episodic_return.svg)     |
| **Forward Reward**  | ![Forward Reward](charts/first-run/episodic_reward_forward.svg)  |
| **Survival Reward** | ![Survival Reward](charts/first-run/episodic_reward_survive.svg) |

### Second Training Run: Speed & Optimization

Fine-tuning from the first checkpoint with increased motor gear and higher forward reward weight. This resulted in significantly faster locomotion while maintaining stability.

#### Performance Metrics

| Metric                   |                                  Chart                                  |
| :----------------------- | :---------------------------------------------------------------------: |
| **Episodic Return**      |     ![Episodic Return](charts/second-run/stats/episodic_return.svg)     |
| **Forward Reward**       | ![Forward Reward](charts/second-run/stats/episodic_reward_forward.svg)  |
| **Survival Reward**      | ![Survival Reward](charts/second-run/stats/episodic_reward_survive.svg) |
| **Forward Velocity**     |      ![XVelocity](charts/second-run/stats/episodic_x_velocity.svg)      |
| **Distance from Origin** | ![Distance](charts/second-run/stats/episodic_distance_from_origin.svg)  |

#### Training Stability

| Metric          |                          Chart                           |
| :-------------- | :------------------------------------------------------: |
| **Policy Loss** | ![Policy Loss](charts/second-run/losses/policy_loss.svg) |
| **Value Loss**  |  ![Value Loss](charts/second-run/losses/value_loss.svg)  |
| **Approx KL**   |   ![Approx KL](charts/second-run/losses/approx_kl.svg)   |

---

## Future Work

- [ ] Multi-task learning (turning, uneven terrain).
- [ ] Obstacle avoidance in cluttered environments.
- [ ] Advanced Adaptive Reward Shaping.
- [ ] Automated hyperparameter tuning.
- [ ] Neuroevolutionary approaches.

---

## References & Credits

### Core Algorithms

- **PPO**: [Schulman et al., 2017](https://arxiv.org/abs/1707.06347)
- **GAE**: [Schulman et al., 2016](https://arxiv.org/abs/1506.02438)
- **MuJoCo**: [Todorov et al., 2012](https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf)

### Credits

- Developed by **wtcherr**.
- Inspired by [CleanRL](https://github.com/vwxyzjn/cleanrl) and Gymnasium's MuJoCo environments.
