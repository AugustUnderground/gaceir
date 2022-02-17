# GAC²EIR

Clean and Self Contained implementations of Reinforcement Learning Agents for
solving [GAC²E](https://github.com/augustunderground/gace).

- [X] PPO (Probabilistic)
- [X] TD3 (Deterministic)
- [X] SAC (Probabilistic)

## Usage

Cleanup:

```
$ make clean wipe kill
```

Will clean model directory, wipe run directory and kill all spectre
instances.

# Notes on Algorithms

My personal notes on implementing the algorithms.

## Proximal Policy Optimization (PPO)

[Proximal Policy Optimization Algorithm](https://arxiv.org/abs/1707.06347)

- Keep track of small, fixed length batch of trajectories (s,a,r,d,v,l)
- Multiple epochs for each batch
- batch sized chunks of memories
- Critic **only** criticises states
- Actor outputs probabilities for taking an action (probabilistic)

### Hyper Parameters

- Memory Size
- Batch size
- Number of Epochs

### Network Updates

#### Actor

Conservative Policy Iteration (CPI):

$$ L^{CPI} (\theta) = E_{t} ( \frac{\pi_{\theta} (a_{t} | s_{t})}{\pi_{\theta,old} (a_{t} | s_{t})} \cdot A_{t} ) = E_{t} (r_{t}(\theta) \cdot A_{t}) $$

Where
- A: Advantage
- E: Expectation
- π Actor Network returning Probability of an action a for a given state s at
  a given time t
- θ: Current network parameters

$$ L^{CLIP} = E_{t} ( min(r_{t}(\theta) \cdot A_{t}, clip(r_{t}(\theta), 1 - \epsilon, 1 + \epsilon) \cdot A_{t} ) ) $$

Where
- ε ≈ 0.2

**Pessimistic lower bound of loss**

##### Advantage

Gives benefit of new state over previous state

$$ A_{t} = \delta_{t} + (\gamma \lambda) \cdot \delta_{t + 1} + ... + (\gamma \lambda)^{T - (t + 1)} \cdot \delta_{T - 1} $$
with

$$ \delta_{t} = r_{t} + \gamma \cdot V(s_{t + 1}) - V(s_{t}) $$

Where
- V(s_t): Critic output, aka Estimated Value (stored in memory)
- γ ≈ 0.95

#### Critic

return = advantage + value

Where value is critic output stored in memory

$$ L^{VF} = MSE(return - value) $$

#### Total Loss

$$ L^{CLIP + VF + S}_{t} (\theta) = E_{t} [ L^{CLIP}_{t} (\theta) - c_{1} \cdot L^{VF}_{t} (\theta) + c_{2} \cdot S[\pi_{\theta}](s_{t}) ] $$

Gradient Ascent, **not** Descent!

- S: only used for shared AC Network
- c1 = 0.5

## Twin Delayed Double Dueling Policy Gradient (TD3)

[Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)

### Hyper Parameters

- Update Intervall
- Number of Epochs
- Number of Samples

### Loss

$$ \nabla_{\phi} J_(\phi) = \frac{1}{N} \sum \nabla_{a} Q_{\theta 1} (s,a) |_{a = \pi_{\phi} (s)} \cdot \nabla_{\phi} \pi_{\phi} (s) $$

Where
- π: Policy Network with parameters φ
- Gradient of first critic w.r.t. actions chosen by critic
- Gradient of policy network w.r.t. it's own parameters

Chain rule applied to loss function

### Network Updates

Initialize Target Networks with parameters from online networks.

$$ \theta \leftarrow \tau \cdot \theta_{i} + (1 - \tau) \cdot \theta_{i}' $$
$$ \phi \leftarrow \tau \cdot \phi{i} + (1 - \tau) \cdot \phi{i}' $$

Where 
- τ ≈ 0.005

_Soft_ update with heavy weight on current target parameters vs. heavily
discounted parameters of online network.

Not every step, only after actor update.

#### Actor

- Randomly sample trajectories from replay buffer (s,a,r,s')
- Use actor to determine actions for sampled states (don't use actions from memory)
- Use sampled states and newly found actions to get values from critic
    + Only the first critic, never the second!
- Take gradient w.r.t. actor network parameters
- Every nth step (hyper parameter of algorithm)

#### Critic

- Randomly sample trajectories from replay buffer (s,a,r,s')
- New states run Ï'(s') where Ï' is target actor
- Add noise and clip

$$ a^{~} \leftarrow \pi_{\phi'} (s') + \epsilon $$

with

$$ \epsilon ~ clip(N(0, \sigma), -c, c) $$

Where
- σ ≈ 0.2, noise standard deviation
- c ≈ 0.5, noise clipping
- γ ≈ 0.99, discount factor

$$ y \leftarrow r + \gamma \cdot min( Q'_{\theta1}(s', a^{~}), Q'_{\theta1}(s', a^{~})) $$

$$ \theta_{i} \leftarrow argmin_{\theta i} ( N^{-1} \cdot \sum ( y - Q_{\theta i} (s,a))^{2} ) $$

## Soft Actor Critic (SAC)

[Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)

**Note**: Entropy in this case means something like _Randomness of actions_,
and is modeled by reward scaling.

$$ log( \pi (a|s) ) = log (\mu (a|s)) - \sum^{D}_{i=1} log ( 1 - tanh^{2} (a_{i}) ) $$

Where
- μ: Sample of a distribution (**NOT MEAN**)
- π: Probability of selecting this particular action a given state s

### Hyper Parameters

- Target smoothing coefficient
- target update interval
- replay buffer size
- gradient steps

### Actor Update

$$ J = N^{-1} \cdot \sum log(\pi (a_{t} | s_{t})) - Q_{min}(s_{t}, a_{t}) $$

Where
- s_t is sampled from replay buffer / memory
- a_t is generated with actor network given sampled states
- Qmin is minimum of 2 critics

### Value Update

$$ J = N^{-1} \cdot \sum \frac{1}{2} \cdot ( V(s_{t}) - Qmin (s_{t}, a_{t}) - log(\pi (a_{t} | s_{t})) ) $$

Where
- V(s_t): sampled values from memory
- s_t: sampled states from memory
- a_t: newly computed actions

### Critic

$$ J_{1} = N^{-1} \sum \frac{1}{2} \cdot ( Q_{1}(s_{t}, a_{t}) - Q'_{1}(s_{t}, a_{t}))^{2} $$

$$ J_{2} = N^{-1} \sum \frac{1}{2} \cdot ( Q_{2}(s_{t}, a_{t}) - Q'_{2}(s_{t}, a_{t}))^{2} $$

$$ Q'= r_{scaled} + \gamma \cdot V'(s_{t + 1}) $$

Where
- **Both** critics get updated
- _Both_ actions and states are sampled from memory

### Network Updates

$$ \psi \leftarrow \tau \cdot \psi + (1 - \tau) \cdot \psi' $$

Where
- τ ≈ 0.005

## Things TODO

- [X] Implement replay buffer / memory as algebraic data type
- [X] Include step count in reward
- [ ] Try Discrete action spaces
- [ ] Normalize and/or reduce observation space
- [ ] consider previous reward
- [X] return trained models instead of loss
- [ ] handling of `done` for parallel envs
- [ ] higher reward for finishsing earlier
