from typing import (
    Tuple,
)

import torch
import numpy as np

from utils_types import (
    BatchAction,
    BatchDone,
    BatchNext,
    BatchReward,
    BatchState,
    BatchIndice,
    Batchweight,
    BatchPriority,
    TensorStack5,
    TorchDevice,
)


class Prioritized_ReplayMemory(object):

    def __init__(
            self,
            channels: int,
            capacity: int,
            alpha   : float,
            device: TorchDevice
    ) -> None:
        self.__capacity = capacity
        self.__alpha = alpha
        self.__device = device
        
        self.__size = 0
        self.__pos = 0

        self.__m_states = torch.zeros(
            (capacity, channels, 84, 84), dtype=torch.uint8)
        self.__m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.__m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.__m_dones = torch.zeros((capacity, 1), dtype=torch.bool)
        self.__m_priorities = torch.zeros((capacity, 1), dtype=torch.float64)

    def push(
            self,
            folded_state: TensorStack5,
            action: int,
            reward: int,
            done: bool
    ) -> None:
        self.__m_states[self.__pos] = folded_state
        self.__m_actions[self.__pos, 0] = action
        self.__m_rewards[self.__pos, 0] = reward
        self.__m_dones[self.__pos, 0] = done
        max_priority = self.__m_priorities.max() if  self.__size else 1.0
        self.__m_priorities[self.__pos, 0] = max_priority
        self.__pos = (self.__pos + 1) % self.__capacity
        self.__size = max(self.__size, self.__pos)


    def sample(self, batch_size: int, beta: float) -> Tuple[
            BatchState,
            BatchAction,
            BatchReward,
            BatchNext,
            BatchDone,
            BatchIndice,
            Batchweight
    ]:
        if self.__size == self.capacity:
            priorities = self.__m_priorities
        else:
            priorities = self.__m_priorities[:self.__pos]

        probabilities = priorities[:,0] ** self.__alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(self.__size, batch_size, p=probabilities)

        weights  = (self.__size * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        indices =  torch.from_numpy(indices)
        weights =  torch.tensor(weights, dtype=torch.float64)

        b_state = self.__m_states[indices, :4].to(self.__device).float()
        b_next = self.__m_states[indices, 1:].to(self.__device).float()
        b_action = self.__m_actions[indices].to(self.__device)
        b_reward = self.__m_rewards[indices].to(self.__device).float()
        b_done = self.__m_dones[indices].to(self.__device).float()

        return b_state, b_action, b_reward, b_next, b_done, indices, weights

    def update_priorities(self, batch_indice: BatchIndice, batch_priority: BatchPriority):
        for idx, priority in zip(batch_indice, batch_priority):
            self.__m_priorities[idx] = priority

    def __len__(self) -> int:
        return self.__size
