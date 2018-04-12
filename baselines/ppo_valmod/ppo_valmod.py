import os
import time
import numpy as np
import os.path as osp
from baselines import logger
from collections import deque
from baselines.common import explained_variance
from baselines.ppo_valmod.model import Model
from baselines.ppo_valmod.runner import Runner

def constfn(val):
    def f(_):
        return val
    return f

#def learn(*, policy, env, nsteps, total_timesteps, ent_coef, lr,
#            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
#            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
#            save_interval=0):
def learn(env, config):

    policy = config.policy
    nsteps = config.nsteps
    total_timesteps = config.total_timesteps
    ent_coef = config.ent_coef
    lr = config.lr
    vf_coef = config.vf_coef
    max_grad_norm = config.max_grad_norm
    gamma = config.gamma
    lam = config.lam
    log_interval = config.log_interval
    nminibatches = config.nminibatches
    noptepochs = config.noptepochs
    cliprange = config.cliprange
    save_interval = config.save_interval

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    make_model = lambda : Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm)
    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()

    nupdates = total_timesteps//nbatch
    assert nbatch % nminibatches == 0
    nbatch_train = nbatch // nminibatches

    # policy iterations
    for update in range(1, nupdates+1):
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)

        # Runner collects trajectories & train model
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632

        epinfobuf.extend(epinfos)
        mblossvals = []
        if states is None: # nonrecurrent version
            inds = np.arange(nbatch)
            # multi-step update on actor, critic
            for _ in range(noptepochs):
                np.random.shuffle(inds)
                # slicing into batches
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    # batch training
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))

        else: # recurrent version
            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            envsperbatch = nbatch_train // nsteps
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, returns)
            logger.logkv("serial_timesteps", update*nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            logger.dumpkvs()
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i'%update)
            print('Saving to', savepath)
            model.save(savepath)
    env.close()

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)
