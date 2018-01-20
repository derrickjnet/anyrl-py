"""
Ways of gathering state transitions to store in a replay
buffer.
"""

from abc import ABC, abstractmethod
from collections import OrderedDict
import time

# pylint: disable=R0902,R0903

class Player(ABC):
    """
    An object that runs a model on an environment and
    produces transitions that can be stored in a replay
    buffer.

    A Player serves a similar role in DQN as a Roller plays in
    policy-gradient algorithms.
    """
    @abstractmethod
    def play(self):
        """
        Gather a sequence of transition dicts.

        Each transition dict should have these keys:
          'obs': the starting observation.
          'action': the chosen action.
          'rewards': the rewards after taking the action.
            For n-step Q-learning, there are n rewards.
          'new_obs': the new observation, or None if the
            transition is terminal.
          'info': the info dictionary from the step when
            the action was taken.
          'start_state': the model's state when it chose
            to take the given action. This may be None.
            This is represented as a batch-size 1 value.
            For example, it might be [1 x n].
          'episode_id': a comparable value that identifies
            the particular episode for this timestep.
          'episode_step': the timestep of the initial
            observation, starting from 0.
          'end_time': the UNIX timestamp when the action
            was finished being applied taken.
          'is_last': a boolean indicating if this is the
            last transition for the episode. This may not
            be True even if new_obs is None, since n-step
            Q-learning can result in multiple terminal
            transitions.
          'total_reward': the total reward for the episode
            up to and including this transition.
            If 'new_obs' is None, then this is the total
            reward for the episode.

        All transitions must be ordered such that, if you
        lay out the transitions first according to play()
        call index and then according to the order within
        a batch, episode_step should always be increasing
        for the corresponding episode_id.
        """
        pass

class BasicPlayer(Player):
    """
    A Player that uses a single Gym environment to gather
    sequential batches of 1-step transitions.
    """
    def __init__(self, env, model, batch_size=1):
        self.env = env
        self.model = model
        self.batch_size = batch_size
        self._needs_reset = True
        self._cur_state = None
        self._last_obs = None
        self._episode_id = 0
        self._episode_step = 0
        self._total_reward = 0.0

    def play(self):
        return [self._gather_transition() for _ in range(self.batch_size)]

    def _gather_transition(self):
        if self._needs_reset:
            self._needs_reset = False
            self._cur_state = self.model.start_state(1)
            self._last_obs = self.env.reset()
            self._episode_id += 1
            self._episode_step = 0
            self._total_reward = 0.0
        output = self.model.step([self._last_obs], self._cur_state)
        new_obs, rew, self._needs_reset, info = self.env.step(output['actions'][0])
        self._total_reward += rew
        res = {
            'obs': self._last_obs,
            'action': output['actions'][0],
            'rewards': [rew],
            'new_obs': (new_obs if not self._needs_reset else None),
            'info': info,
            'start_state': self._cur_state,
            'episode_id': self._episode_id,
            'episode_step': self._episode_step,
            'end_time': time.time(),
            'is_last': self._needs_reset,
            'total_reward': self._total_reward
        }
        self._cur_state = output['states']
        self._last_obs = new_obs
        self._episode_step += 1
        return res

class NStepPlayer(Player):
    """
    A Player that wraps another Player and uses n-step
    transitions instead of 1-step transitions.
    """
    # pylint: disable=R0913
    def __init__(self, player, num_steps):
        self.player = player
        self.num_steps = num_steps
        self._ep_to_history = OrderedDict()

    def play(self):
        # Let the buffers fill up until we get actual
        # n-step transitions.
        while True:
            transes = self._play_once()
            if transes:
                return transes

    def _play_once(self):
        for trans in self.player.play():
            assert len(trans['rewards']) == 1
            ep_id = trans['episode_id']
            if ep_id in self._ep_to_history:
                self._ep_to_history[ep_id].append(trans)
            else:
                self._ep_to_history[ep_id] = [trans]
        res = []
        for ep_id, history in list(self._ep_to_history.items()):
            while history:
                trans = self._next_transition(history)
                if trans is None:
                    break
                res.append(trans)
            if not history:
                del self._ep_to_history[ep_id]
        return res

    def _next_transition(self, history):
        if len(history) < self.num_steps:
            if not history[-1]['is_last']:
                return None
        res = history[0].copy()
        res['rewards'] = [h['rewards'][0] for h in history[:self.num_steps]]
        res['total_reward'] += sum(h['rewards'][0] for h in history[1:self.num_steps])
        if len(history) >= self.num_steps:
            res['new_obs'] = history[self.num_steps-1]['new_obs']
        else:
            res['new_obs'] = None
        del history[0]
        return res
