import numpy as np

class KinsEnv:
    def __init__(self, num_lines, max_len, line_params):
        self.num_lines = num_lines
        self.max_len = max_len
        self.line_params = line_params

    # Actions are all the lines that don't have max_len plus the action of not adding anybody    
    def actions(self, state):
        if np.sum(state) == self.num_lines * self.max_len:
            return [self.num_lines]
        return np.hstack((np.array([i for i in range(self.num_lines) if state[i] < self.max_len]), np.array([self.num_lines])))
    
    # Returns the reward and the new state
    # Reward is the number of processed people
    def step(self, state, action):
        s = np.copy(state)
        if action != self.num_lines:
            s[action] += 1
        
        old_total = np.sum(s)
        s -= np.array([1 if np.random.rand() < self.line_params[i] and s[i] != 0 else 0 for i in range(self.num_lines)])
        # reward = old_total - np.sum(state) if action != self.num_lines else -1
        # reward = old_total - np.sum(s)
        reward = np.sum(s)

        return reward, np.copy(s)
    

class SarsaAgent:
    def __init__(self, env, alpha, gamma, epsilon, n=1, episode_length=1000):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n = n
        self.episode_length = episode_length

        self.Q = np.zeros(((env.max_len+1)**env.num_lines, env.num_lines+1))
        self.states = [0 for i in range(episode_length)]
        self.actions = [0 for i in range(episode_length)]
        self.rewards = [0 for i in range(episode_length)]
        self.sigmas = [0 for i in range(episode_length)]
        self.importance_factors = [0 for i in range(episode_length)]

        self.coverage_map = np.zeros([env.max_len+1 for i in range(env.num_lines)])

    def state_to_index(self, state):
        return sum([state[i] * (self.env.max_len+1)**i for i in range(self.env.num_lines)])
    
    def index_to_state(self, index):
        state = np.zeros(self.env.num_lines)
        for i in range(self.env.num_lines):
            state[i] = index % (self.env.max_len+1)
            index //= self.env.max_len+1
        return state.astype(int)
    
    def greedy_policy(self, state):
        valid_actions = self.env.actions(state)
        return valid_actions[np.argmax(self.Q[self.state_to_index(state), valid_actions])]
    
    def epsilon_greedy_policy(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.actions(state))
        else:
            return self.greedy_policy(state)
    
    def importance_factor(self, state, action):
        num_actions = len(self.env.actions(state))
        if action == self.greedy_policy(state):
            return 1 - self.epsilon * (1 - 1/num_actions)
        else:
            return self.epsilon/num_actions
        
    def learn(self, num_episodes=100):
        rewards = []
        coverage = []
        for episode in range(num_episodes):
            T = np.inf
            tau = 0
            t = 0
            G = 0
            V = 0

            self.states[0] = np.array([np.random.randint(0,self.env.max_len+1) for i in range(self.env.num_lines)])
            self.actions[0] = self.epsilon_greedy_policy(self.states[0])
            self.coverage_map[tuple(self.states[0])] = 1
            while tau != T - 1:
                if t < T:
                    self.rewards[t+1], self.states[t+1] = self.env.step(self.states[t], self.actions[t])
                    self.coverage_map[tuple(self.states[t+1])] = 1
                    # The max length step is always terminal, so we check if t+1==max_length-1
                    if t == self.episode_length - 2:
                        T = t + 1
                    else:
                        self.actions[t+1] = self.epsilon_greedy_policy(self.states[t+1])
                        self.sigmas[t+1] = 0
                        self.importance_factors[t+1] = self.importance_factor(self.states[t+1], self.actions[t+1])
                tau = t - self.n + 1
                if tau >= 0:
                    if t+1 < T:
                        G = self.Q[self.state_to_index(self.states[t+1]), self.actions[t+1]]
                    for k in range(min(t+1, T), tau, -1):
                        if k == T:
                            G = self.rewards[T]
                        else:
                            a = self.greedy_policy(self.states[k])
                            s = self.states[k]
                            r = self.rewards[k]
                            sigma = self.sigmas[k]
                            importance_factor = self.importance_factors[k]
                            V = Q = self.Q[self.state_to_index(s), a]

                            G = r + self.gamma * (sigma*importance_factor + (1-sigma))*(G - Q) + self.gamma * V
                    self.Q[self.state_to_index(self.states[tau]), self.actions[tau]] += self.alpha * (G - self.Q[self.state_to_index(self.states[tau]), self.actions[tau]])
                t += 1
            rewards.append(np.sum(self.rewards))
            coverage.append(np.sum(self.coverage_map)/(self.env.max_len+1)**self.env.num_lines)
            print("Episode: ", episode, " Reward: ", np.sum(self.rewards), " Coverage: ", np.sum(self.coverage_map)/(self.env.max_len+1)**self.env.num_lines)
        return rewards, coverage

    def pretty_Q(self):
        Q_by_action = [i for i in range(self.env.num_lines+1)]
        for i in range(self.env.num_lines+1):
            pretty_Q = np.zeros([self.env.max_len+1 for i in range(self.env.num_lines)])
            for j in range((self.env.max_len+1)**self.env.num_lines):
                state = tuple(self.index_to_state(j))
                pretty_Q[state] = self.Q[j, i]
            Q_by_action[i] = np.copy(pretty_Q)
        return Q_by_action
