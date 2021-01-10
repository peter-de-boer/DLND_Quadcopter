# TODO: your agent here!
import numpy as np
from task import Task
from agents.actor import Actor
from agents.critic import Critic
from agents.ounoise import OUNoise
from replaybuffer import ReplayBuffer
        
class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.exploration_mu = 0.0
        self.exploration_theta = 0.5 
        self.exploration_sigma = 0.1
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size =  1000
        self.batch_size =  64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = 0.0 # 0.99# discount factor
        self.tau = 0.001 # 0.001 was best for contrained speeds 0.01  # for soft update of target parameters
        
        self.best_score = -np.inf

    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0
        
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        #self.max_reward=-1000
        return state

    def step(self, action, reward, next_state, done):
        # Save experience / reward       
        self.total_reward += reward
        #print(reward, self.total_reward)
        self.count += 1
        self.score = self.total_reward / float(self.count) #if self.count else 0.0
        #print(reward, self.total_reward, self.count, self.score)
        if self.score > self.best_score:
            self.best_score = self.score
        #if reward > self.max_reward:
        #    self.max_reward = reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > 2*self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state

    def act(self, states):
        """Returns actions for given state(s) as per current policy."""
        # original:
        # state = np.reshape(state, [-1, self.state_size])
        state = np.reshape(states, [-1, self.state_size])
        action_array = self.actor_local.model.predict(state)
        action = self.actor_local.model.predict(state)[0]
        Q_value = self.critic_target.model.predict([state, action_array])[0]
        return list(action + self.noise.sample()), Q_value  # add some noise for exploration

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        # shouldn't we better calculate Q_targets_next while acting (and store it in experiences)
        # but anyway, critic seems to converge well
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        # orig:
        #action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        # use critic_target insteas of critic_local?
        # actions is from experiences, but shouldn't we take the actions from the current actor?
        # also, maybe independent sample for the actor?
        actions_current = self.actor_target.model.predict_on_batch(states)
        action_gradients = np.reshape(self.critic_target.get_action_gradients([states, actions_current, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)   

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)