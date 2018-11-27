from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from ray.rllib.agents.agent import Agent
from ray.rllib.optimizers import AsyncGradientsOptimizer

from .a3c_torch_policy_graph import A3CTorchPolicyGraph
from .config import DEFAULT_CONFIG


class A3CAgent(Agent):
    """A3C implementations in TensorFlow and PyTorch."""

    _agent_name = "A3C"
    _default_config = DEFAULT_CONFIG
    _policy_graph = A3CTorchPolicyGraph

    def _init(self):
        policy_cls = A3CTorchPolicyGraph

        self.local_evaluator = self.make_local_evaluator(
            self.env_creator, policy_cls)
        self.remote_evaluators = self.make_remote_evaluators(
            self.env_creator, policy_cls, self.config["num_workers"])
        self.optimizer = self._make_optimizer()

    def _make_optimizer(self):
        return AsyncGradientsOptimizer(self.local_evaluator,
                                       self.remote_evaluators,
                                       self.config["optimizer"])

    def _train(self):
        prev_steps = self.optimizer.num_steps_sampled
        start = time.time()
        while time.time() - start < self.config["min_iter_time_s"]:
            self.optimizer.step()
        result = self.optimizer.collect_metrics(
            self.config["collect_metrics_timeout"])
        result.update(timesteps_this_iter=self.optimizer.num_steps_sampled -
                      prev_steps)
        return result
