import random

class SumTree:
    def __init__(self, capacity: int):
        # Use power of two for simpler index math
        N = 1
        while N < capacity:
            N <<= 1
        self.N = N
        self.capacity = capacity
        self.tree = [0.0] * (2 * N)
        self.size = 0
        self.data = [None] * capacity
        self.write = 0

    @property
    def total(self) -> float:
        return self.tree[1]

    def update(self, data_idx: int, priority: float):
        # Set leaf and push changes upward
        i = self.N + data_idx
        delta = priority - self.tree[i]
        self.tree[i] = priority
        i //= 2
        while i >= 1:
            self.tree[i] += delta
            i //= 2

    def add(self, priority: float, item):
        # Insert/overwrite in cyclic buffer fashion
        i = self.write
        self.data[i] = item
        self.update(i, priority)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return i  # return data index so caller can update priority later

    def get_prefix(self, mass: float):
        """Return (data_idx, priority, item) s.t. cumulative sum crosses 'mass'."""
        # Guard for numerical drift
        mass = max(0.0, min(mass, self.total))
        i = 1
        while i < self.N:
            left = 2 * i
            if self.tree[left] >= mass:
                i = left
            else:
                mass -= self.tree[left]
                i = left + 1
        data_idx = i - self.N
        # If capacity not power-of-two, data beyond self.capacity may be dummy zeros.
        if data_idx >= self.capacity:
            data_idx = self.capacity - 1
        return data_idx, self.tree[i], self.data[data_idx]
    
class PERBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_end=1.0, beta_steps=1_000_000, eps=1e-6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.eps = eps
        self.max_priority = 1.0

        # IS weights anneal
        self.beta = beta_start
        self.beta_start, self.beta_end = beta_start, beta_end
        self.beta_steps = beta_steps
        self.step = 0

    def _to_stored_priority(self, td_error_abs):
        # store p^alpha in the tree
        return (td_error_abs + self.eps) ** self.alpha

    def add(self, state, action, reward, next_state, done):
        priority = self._to_stored_priority(self.max_priority)  # ensure new samples get seen
        return self.tree.add(priority, (state, action, reward, next_state, done))

    def sample(self, batch_size):
        total = self.tree.total
        seg = total / batch_size
        samples, idxs, priors = [], [], []

        for i in range(batch_size):
            left, right = seg * i, seg * (i + 1)
            mass = random.uniform(left, right)
            idx, p, item = self.tree.get_prefix(mass)
            samples.append(item)
            idxs.append(idx)
            priors.append(p)

        # anneal beta
        self.step += 1
        t = min(1.0, self.step / self.beta_steps)
        self.beta = self.beta_start + t * (self.beta_end - self.beta_start)

        # importance sampling weights
        probs = [p / total for p in priors]
        N = self.tree.size
        weights = [(N * pr) ** (-self.beta) for pr in probs]
        max_w = max(weights) if weights else 1.0
        weights = [w / (max_w + 1e-8) for w in weights]

        return samples, idxs, weights

    def update_priorities(self, idxs, td_errors_abs):
        for idx, e in zip(idxs, td_errors_abs):
            p = self._to_stored_priority(float(e))
            self.tree.update(idx, p)
            # track max *base* priority (before ^alpha) so fresh adds get high priority
            self.max_priority = max(self.max_priority, float(e))

    def __len__(self):
        return self.tree.size