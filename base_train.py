import ray
from ray.tune.logger import pretty_print

from agents.rllib_a3c import a3c
from agents.rllib_a3c.config import DEFAULT_CONFIG

ray.init()
config = DEFAULT_CONFIG.copy()
config["num_gpus"] = 1
config["num_workers"] = 1
print(config)
agent = a3c.A3CAgent(config=config, env="CartPole-v0")

# Can optionally call agent.restore(path) to load a checkpoint.

for i in range(1000):
   # Perform one iteration of training the policy with PPO
   result = agent.train()
   print(pretty_print(result))

   if i % 100 == 0:
       checkpoint = agent.save()
       print("checkpoint saved at", checkpoint)
