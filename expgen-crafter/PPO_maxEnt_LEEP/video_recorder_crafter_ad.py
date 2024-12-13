import argparse
import os
import random

import numpy as np
import torch

from crafter.env import Env
from crafter.recorder import VideoRecorder
from expgen.PPO_maxEnt_LEEP.model import Policy, ImpalaModel
from expgen.PPO_maxEnt_LEEP.arguments import get_args
from expgen.PPO_maxEnt_LEEP.procgen_wrappers import ModifiedObservationWrapper


def main(args):
    # Fix Random seed
    random.seed(args.eval_seed)
    np.random.seed(args.eval_seed)
    torch.manual_seed(args.eval_seed)
    torch.cuda.manual_seed_all(args.eval_seed)

    # CUDA setting
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define checkpoint directory
    ckpt_dir = "/home/user_118/proj/crafter/expgen/logs/craftercrafter_ppo_seed_0_02-12-2024_10-57-12"

    # Create environment
    env = Env(seed=args.eval_seed)
    env = ModifiedObservationWrapper(env)
    env = VideoRecorder(env, directory=f"./videos")

    # Create model
    args = get_args()
    actor_critic = Policy(
        env.observation_space.shape,
        env.action_space,
        base=ImpalaModel,
        base_kwargs={'recurrent': args.recurrent_policy, 'hidden_size': args.recurrent_hidden_size,
                     'gray_scale': args.gray_scale}
    )
    actor_critic.to(device)

    # Load checkpoint
    ckpt_path = os.path.join(ckpt_dir, f"crafter-epoch-{args.saved_epoch}.pt")
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    actor_critic.load_state_dict(state_dict['state_dict'])

    # Eval
    actor_critic.eval()
    obs = env.reset()
    done = False
    total_reward = 0
    eval_recurrent_hidden_states = torch.zeros(
        1, 256, device=device)
    eval_masks = torch.zeros(1, 1, device=device)

    while not done:
        with torch.no_grad():
            obs = torch.FloatTensor(obs.transpose(2, 0, 1) / 255.0).unsqueeze(0).to(device)
            value, action, action_log_prob, recurrent_hidden_states, _ = actor_critic.act(
                obs, eval_recurrent_hidden_states, eval_masks, deterministic=False
            )

        obs, reward, done, _ = env.step(action.item())
        total_reward += reward

    print(f"Total reward: {total_reward}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--exp_name", type=str, required=True)
    # parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--train_seed", type=int, default=0)
    parser.add_argument("--eval_seed", type=int, default=205)
    parser.add_argument("--saved_epoch", type=int, default=250)
    args = parser.parse_args()

    main(args)
