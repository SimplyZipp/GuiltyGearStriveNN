

class Memory:
    def __init__(self):
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropy = None

    def add(self, action, log_prob, value, reward):
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)

    def reset(self):
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.entropy = None

    def _zip(self):
        return zip(self.actions, self.log_probs, self.values, self.rewards)

    def __iter__(self):
        for z in self._zip():
            yield z

    def __reversed__(self):
        for z in reversed(list(self._zip())):
            yield z

    def __len__(self):
        return len(self.rewards)

    def __getitem__(self, i):
        return self.actions[i], self.log_probs[i], self.values[i], self.rewards[i]
