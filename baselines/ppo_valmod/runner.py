import numpy as np
import gym

class Runner(object):

    def __init__(self, *, env, model, nsteps, gamma, lam, config):
        self.env = env
        self.model = model
        nenv = env.num_envs
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=model.train_model.X.dtype.name)
        self.obs[:] = env.reset()
        self.state = [self.env.envs[0].env.env.sim.data.qpos, self.env.envs[0].env.env.sim.data.qvel]
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

        self.config = config

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        sp_states = []
        mb_states = self.states
        epinfos = []
        for _ in range(self.nsteps):
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            sp_states.append(self.state.copy())
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            self.state = [self.env.envs[0].env.env.sim.data.qpos, self.env.envs[0].env.env.sim.data.qvel]
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)
        # ideal model prediction
        sp_obs = mb_obs.copy()
        sp_obs.append(self.obs.copy())
        sp_states.append(self.state)
        sp_values = self.ideal_pred(sp_states, sp_obs, self.env.envs[0].env.env.spec.id, self.model.act_model)
        if self.config.img_switch:
            last_values = sp_values[-1]
            mb_values = sp_values[:-1]
        #discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)
        # obs, returns, masks, actions, values, neglogpacs, states = runner.run()

    def ideal_pred(self, states, obses, env_id, act_model):
        env = gym.make(env_id)
        # calculate the ideal predictions
        value_ip = []
        for i in range(len(states)):
            # mean of many trials
            vals = []
            for j in range(self.config.img_trials):
                env.set_state(*(states[i]))
                obs = obses[i]
                val = v = k = 0
                # gamma discounted reward.
                for k in range(self.config.img_n_steps):
                    a, v, _, _ = act_model.step(obs)
                    obs, r, d, _ = env.step(a)
                    val += r * self.config.gamma ** k
                    if d:
                        val -= v ** k
                        break
                val += v ** k
                vals.append(val)
            value_ip.append(np.mean(vals))
        return np.asarray(value_ip)




def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])