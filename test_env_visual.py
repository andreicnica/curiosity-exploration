import torch
import cv2
import time
import numpy as np

from gathering_mae import GatheringEnv
from configs import get_config

np.set_printoptions(threshold=np.nan, linewidth=np.nan, precision=2)


if __name__ == '__main__':

    # Get default config
    cfg = get_config("default")

    EVAL_STEPS = 100000

    no_agents = cfg.no_agents
    # Force visual
    cfg.visualize = True
    visualize = cfg.visualize

    env = GatheringEnv(cfg)
    env_step = 0
    ep_r = 0

    done = torch.ones(1).byte()

    agent0_r = []
    start_time = time.time()
    while env_step < EVAL_STEPS:
        # check if env needs reset
        if done.any():
            obs, r, done = env.restart_game()
            if visualize:
                env.render(True)

            print("Episode finished:")
            print("Return per episode: {}".format(sum(agent0_r)))
            if len(agent0_r) > 0:
                print("Agent0 reward: {}".format(np.mean(agent0_r)))

            agent0_r = []

        actions = np.random.randint(7, size=no_agents)

        if visualize:
            # actions[1] = 4
            actions[0] = -1
            while actions[0] == -1 or actions[0] == 255:
                key = cv2.waitKey(0) & 0xFF

                # if the 'ESC' key is pressed, Quit
                if key == 27:
                    quit()
                elif key == 82:   # Arrow up
                    actions[0] = 0
                elif key == 83:  # Arrow right
                    actions[0] = 1
                elif key == 84:  # Arrow down
                    actions[0] = 2
                elif key == 81:  # Arrow left
                    actions[0] = 3
                elif key == 113:  # Null action q
                    actions[0] = 4
                elif key == 119:  # Turn clockwise w
                    actions[0] = 5
                elif key == 101:  # Turn counterclockwise e
                    actions[0] = 6
                elif key == 114:  # Action - laser r
                    actions[0] = 7
                elif key != 255:
                    print("Unknown key: {}".format(key))

        obs, r, done, _ = env.step(actions)

        env_step += 1
        agent0_r.append(r[0])

        if visualize:
            print(f"Step: {env_step};\t Reward: {r.numpy()}")
            env.render(True)

        if env_step % 10000 == 0:
            print(time.time() - start_time)
            start_time = time.time()