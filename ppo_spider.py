import argparse
import os
import random
import time
import datetime
from distutils.util import strtobool
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from spider_v0 import SpiderEnv

# Register the environment
gym.register(id="mujoco_env/Spider-v0", entry_point=SpiderEnv, max_episode_steps=1000)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name",
        type=str,
        default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment",
    )
    parser.add_argument(
        "--gym-id",
        type=str,
        default="mujoco_env/Spider-v0",
        help="the id of the gym environment",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="the learning rate of the optimizer",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed of the random generator",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=8_000_000,
        help="the total time steps of the experiment",
    )
    parser.add_argument(
        "--torch-deterministic",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=True` (default:True)",
    )
    parser.add_argument(
        "--cuda",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, cuda will be enabled by default (default:True)",
    )
    parser.add_argument(
        "--track",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="if toggled, this experiment will be tracked with W&B (default:False)",
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default="spider-rl",
        help="the wandb's project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="the entity (team) of wandb's project",
    )
    parser.add_argument(
        "--capture-video",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="whether to capture videos of the agent performances (videos folder) (default:False)",
    )
    parser.add_argument(
        "--render",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="whether to render the environment",
    )
    parser.add_argument(
        "--save-model",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="whether to save the model",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="the path to the model to load",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1_000_000,
        help="the interval (in timesteps) to save a checkpoint of the model",
    )

    # Algorithm specific arguments
    parser.add_argument(
        "--num-envs",
        type=int,
        default=16,
        help="the number of parallel environments",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=512,
        help="the number of steps to run in each environment per policy rollout",
    )
    parser.add_argument(
        "--anneal-lr",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="toggle learning rate annealing for policy and value networks (default:True)",
    )
    parser.add_argument(
        "--gae",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="use generalized advantage estimation (gae) for advantage computation (default:True)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="the discount factor gamma",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="the lambda fo the generalized advantage estimation",
    )
    parser.add_argument(
        "--num-minibatchs",
        type=int,
        default=32,
        help="the number of mini-batch to use in update",
    )
    parser.add_argument(
        "--update-epochs",
        type=int,
        default=10,
        help="the K epochs to update the policy",
    )
    parser.add_argument(
        "--norm-adv",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="whether to perform normalization to the advantages (default:True)",
    )
    parser.add_argument(
        "--clip-coef",
        type=float,
        default=0.2,
        help="the surrogate clipping coefficient",
    )
    parser.add_argument(
        "--clip-vloss",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="whether to use clipped loss for the value function (default:True)",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="coefficient of the entropy",
    )
    parser.add_argument(
        "--vf-coef",
        type=float,
        default=0.5,
        help="coefficient of the value function",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="the maximum norm for the gradient clipping",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=None,
        help="the target KL divergence threshold",
    )

    # Environment specific arguments
    parser.add_argument(
        "--xml-file",
        type=str,
        default="./scene.xml",
        help="the path to a MuJoCo model (default:'./scene.xml')",
    )
    parser.add_argument(
        "--forward-reward-weight",
        type=float,
        default=5.0,
        help="weight for forward reward",
    )
    parser.add_argument(
        "--ctrl-cost-weight",
        type=float,
        default=0.035,
        help="weight for control cost",
    )
    parser.add_argument(
        "--contact-cost-weight",
        type=float,
        default=5e-4,
        help="weight for contact cost",
    )
    parser.add_argument(
        "--z-orientation-cost-weight",
        type=float,
        default=30.0,
        help="weight for z-orientation variance cost",
    )
    parser.add_argument(
        "--healthy-reward",
        type=float,
        default=1.0,
        help="weight for healthy reward (survival reward) (default:1.0)",
    )
    parser.add_argument(
        "--main-body-id",
        type=int,
        default=1,
        help="ID of the body, whose displacement is used to calculate the dx/_forward_reward_ (default:`1`('Torso')) ignored if --main-body-name is not `None`",
    )
    parser.add_argument(
        "--main-body-name",
        type=str,
        default=None,
        help="name of the body, whose displacement is used to calculate the *dx*/_forward_reward_",
    )
    parser.add_argument(
        "--terminate-when-unhealthy",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if `True`, issue a `terminated` signal is unhealthy",
    )
    parser.add_argument(
        "--healthy-z-range",
        type=tuple,
        default=(-0.05, 0.5),
        help="the spider is considered healthy if the z-coordinate of the torso is in this range",
    )
    parser.add_argument(
        "--contact-force-range",
        type=tuple,
        default=(-1, 1),
        help="contact forces are clipped to this range in the computation of contact_cost",
    )
    parser.add_argument(
        "--reset-noise-scale",
        type=float,
        default=0.1,
        help="scale of random perturbations of initial position and velocity (default:0.1)",
    )
    parser.add_argument(
        "--exclude-current-positions-from-observation",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="whether or not to omit the x- and y-coordinates from observations. Excluding the position can serve as an inductive bias to induce position-agnostic behavior in policies",
    )
    parser.add_argument(
        "--include-cfrc-ext-in-observation",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="whether to include cfrc_ext elements in the observations (default:False)",
    )
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatchs)
    return args


def make_env(gym_id, seed, idx, capture_video, run_name, render):
    environment_args = {
        "xml_file": args.xml_file,
        "forward_reward_weight": args.forward_reward_weight,
        "ctrl_cost_weight": args.ctrl_cost_weight,
        "contact_cost_weight": args.contact_cost_weight,
        "z_orientation_cost_weight": args.z_orientation_cost_weight,
        "healthy_reward": args.healthy_reward,
        "main_body": (
            args.main_body_name
            if args.main_body_name is not None
            else args.main_body_id
        ),
        "terminate_when_unhealthy": args.terminate_when_unhealthy,
        "healthy_z_range": args.healthy_z_range,
        "contact_force_range": args.contact_force_range,
        "reset_noise_scale": args.reset_noise_scale,
        "exclude_current_positions_from_observation": args.exclude_current_positions_from_observation,
        "include_cfrc_ext_in_observation": args.include_cfrc_ext_in_observation,
    }

    def thunk():
        if render and idx == 0:
            env = gym.make(gym_id, render_mode="human", **environment_args)
        elif capture_video and idx == 0:
            env = gym.make(gym_id, render_mode="rgb_array", **environment_args)
            env = gym.wrappers.RecordVideo(
                env, f"videos/{run_name}", step_trigger=lambda t: t % 50000 == 0
            )
        else:
            env = gym.make(gym_id, **environment_args)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env, lambda obs: np.clip(obs, -10, 10), env.observation_space
        )
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(
            env, lambda reward: np.clip(reward, -10.0, 10.0)
        )

        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 128)),
            nn.Tanh(),
            layer_init(
                nn.Linear(128, np.prod(envs.single_action_space.shape)), std=0.01
            ),
        )
        self.actor_logstd = nn.Parameter(
            torch.zeros(1, np.prod(envs.single_action_space.shape))
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device):
        self.load_state_dict(torch.load(path, map_location=device))


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.gym_id.replace('/', '_')}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=False,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    log_file_path = f"./runs/{run_name}/log.txt"

    def log_to_file(message):
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")
        print(message)

    log_to_file("Hyperparameters:")
    for key, value in vars(args).items():
        log_to_file(f"  {key}: {value}")
    log_to_file(f"\n{'=' * 60}\n")

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.gym_id, args.seed + i, i, args.capture_video, run_name, args.render
            )
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    log_to_file(f"Observation space: {envs.single_observation_space.shape}")
    log_to_file(f"Action space: {envs.single_action_space.shape}")
    log_to_file(f"\n{'=' * 60}\n")

    agent = Agent(envs).to(device)
    if args.model_path:
        agent.load(args.model_path, device)
        log_to_file(f"Loaded model from {args.model_path}")

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    log_to_file(f"Critic model: {agent.critic}")
    log_to_file(f"Actor mean model: {agent.actor_mean}")
    log_to_file(f"Actor logstd model: {agent.actor_logstd}")
    log_to_file(f"\n{'=' * 60}\n")

    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    log_probs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, info = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    # Reward component tracking
    episodic_reward_forward = np.zeros(args.num_envs)
    episodic_reward_ctrl = np.zeros(args.num_envs)
    episodic_reward_contact = np.zeros(args.num_envs)
    episodic_reward_z_orientation = np.zeros(args.num_envs)
    episodic_reward_survive = np.zeros(args.num_envs)
    episodic_distance_from_origin = np.zeros(args.num_envs)
    episodic_x_velocity = np.zeros(args.num_envs)
    episodic_y_velocity = np.zeros(args.num_envs)

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, log_prob, entropy, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            log_probs[step] = log_prob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, termination, truncation, info = envs.step(
                action.cpu().numpy()
            )
            episodic_reward_forward += info["reward_forward"]
            episodic_reward_ctrl += info["reward_ctrl"]
            episodic_reward_contact += info["reward_contact"]
            episodic_reward_z_orientation += info["reward_z_orientation"]
            episodic_reward_survive += info["reward_survive"]
            episodic_distance_from_origin += info["distance_from_origin"]
            episodic_x_velocity += info["x_velocity"]
            episodic_y_velocity += info["y_velocity"]

            rewards[step] = torch.Tensor(reward).to(device)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(termination | truncation).to(device)

            if any(next_done):
                finished_envs = next_done.cpu().numpy().astype(bool)
                # Select the best environment among those that finished
                finished_indices = np.where(finished_envs)[0]
                best_idx_in_finished = np.argmax(info["episode"]["r"][finished_envs])
                best_env_idx = finished_indices[best_idx_in_finished]

                episodic_return = info["episode"]["r"][best_env_idx]
                episodic_length = info["episode"]["l"][best_env_idx]
                log_to_file(
                    f"global_step={global_step}, episodic_return={episodic_return}, episodic_length={episodic_length}"
                )
                writer.add_scalar(
                    "charts/episodic_return", episodic_return, global_step
                )
                writer.add_scalar(
                    "charts/episodic_length", episodic_length, global_step
                )
                writer.add_scalar(
                    "charts/episodic_reward_forward",
                    episodic_reward_forward[best_env_idx],
                    global_step,
                )
                writer.add_scalar(
                    "charts/episodic_reward_ctrl",
                    episodic_reward_ctrl[best_env_idx],
                    global_step,
                )
                writer.add_scalar(
                    "charts/episodic_reward_contact",
                    episodic_reward_contact[best_env_idx],
                    global_step,
                )
                writer.add_scalar(
                    "charts/episodic_reward_z_orientation",
                    episodic_reward_z_orientation[best_env_idx],
                    global_step,
                )
                writer.add_scalar(
                    "charts/episodic_reward_survive",
                    episodic_reward_survive[best_env_idx],
                    global_step,
                )
                writer.add_scalar(
                    "charts/episodic_distance_from_origin",
                    episodic_distance_from_origin[best_env_idx],
                    global_step,
                )
                writer.add_scalar(
                    "charts/episodic_x_velocity",
                    episodic_x_velocity[best_env_idx],
                    global_step,
                )
                writer.add_scalar(
                    "charts/episodic_y_velocity",
                    episodic_y_velocity[best_env_idx],
                    global_step,
                )
                # Reset statistics for all finished environments
                episodic_reward_forward[finished_envs] = 0
                episodic_reward_ctrl[finished_envs] = 0
                episodic_reward_contact[finished_envs] = 0
                episodic_reward_z_orientation[finished_envs] = 0
                episodic_reward_survive[finished_envs] = 0
                episodic_distance_from_origin[finished_envs] = 0
                episodic_x_velocity[finished_envs] = 0
                episodic_y_velocity[finished_envs] = 0

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                last_gaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        next_nonterminal = 1.0 - next_done
                        next_values = next_value
                    else:
                        next_nonterminal = 1.0 - dones[t + 1]
                        next_values = values[t + 1]
                    delta = (
                        rewards[t]
                        + args.gamma * next_values * next_nonterminal
                        - values[t]
                    )
                    advantages[t] = last_gaelam = (
                        delta
                        + args.gamma * args.gae_lambda * next_nonterminal * last_gaelam
                    )
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        next_nonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = (
                        rewards[t] + args.gamma * next_nonterminal * next_return
                    )
                advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_log_probs = log_probs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clip_fracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                _, newlog_probs, entropy, new_values = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                log_ratios = newlog_probs - b_log_probs[mb_inds]
                ratios = log_ratios.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-log_ratios).mean()
                    approx_kl = ((ratios - 1.0) - log_ratios).mean()
                    clip_fracs += [
                        ((ratios - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # policy loss
                pg_loss1 = -mb_advantages * ratios
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratios, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # value loss
                new_values = new_values.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (new_values - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        new_values - b_values[mb_inds], -args.clip_coef, args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((new_values - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                # global gradient clipping
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clip_fracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        elapsed_time = int(time.time() - start_time)
        remaining_time = (
            int(elapsed_time / global_step * (args.total_timesteps - global_step))
            if global_step > 0
            else 0
        )
        writer.add_scalar(
            "charts/SPS", int(global_step / max(elapsed_time, 1)), global_step
        )
        log_to_file(
            f"Steps/Second: {int(global_step / max(elapsed_time, 1))}, "
            f"Elapsed time: {datetime.timedelta(seconds=elapsed_time)}, "
            f"Remaining time: {datetime.timedelta(seconds=remaining_time)}"
        )

        # Checkpoint saving
        if args.save_model and args.checkpoint_interval > 0:
            if (global_step // args.checkpoint_interval) > (
                (global_step - args.batch_size) // args.checkpoint_interval
            ):
                checkpoint_path = f"runs/{run_name}/{args.exp_name}_{global_step}.pt"
                agent.save(checkpoint_path)
                log_to_file(f"checkpoint saved to {checkpoint_path}")

                # Save RMS statistics for checkpoints
                def get_rms_stats(env):
                    obs_rms = None
                    ret_rms = None
                    curr_env = env
                    while hasattr(curr_env, "env"):
                        if isinstance(curr_env, gym.wrappers.NormalizeObservation):
                            obs_rms = curr_env.obs_rms
                        if isinstance(curr_env, gym.wrappers.NormalizeReward):
                            ret_rms = curr_env.return_rms
                        curr_env = curr_env.env
                    return obs_rms, ret_rms

                obs_rms, ret_rms = get_rms_stats(envs.envs[0])
                stats_base_path = checkpoint_path.replace(".pt", "")
                if obs_rms is not None:
                    torch.save(obs_rms, f"{stats_base_path}_obs_rms.pt")
                if ret_rms is not None:
                    torch.save(ret_rms, f"{stats_base_path}_ret_rms.pt")

    envs.close()
    writer.close()

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}"
        agent.save(model_path)
        log_to_file(f"model saved to {model_path}")

        # Save RMS statistics
        # Since we use SyncVectorEnv, we can take stats from the first environment
        # as they should be similar across all parallel environments.
        def get_rms_stats(env):
            obs_rms = None
            ret_rms = None
            curr_env = env
            while hasattr(curr_env, "env"):
                if isinstance(curr_env, gym.wrappers.NormalizeObservation):
                    obs_rms = curr_env.obs_rms
                if isinstance(curr_env, gym.wrappers.NormalizeReward):
                    ret_rms = curr_env.return_rms
                curr_env = curr_env.env
            return obs_rms, ret_rms

        obs_rms, ret_rms = get_rms_stats(envs.envs[0])
        if obs_rms is not None:
            torch.save(obs_rms, f"{model_path}_obs_rms.pt")
            log_to_file(f"observation RMS saved to {model_path}_obs_rms.pt")
        if ret_rms is not None:
            torch.save(ret_rms, f"{model_path}_ret_rms.pt")
            log_to_file(f"reward RMS saved to {model_path}_ret_rms.pt")
