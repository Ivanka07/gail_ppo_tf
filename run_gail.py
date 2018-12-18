#!/usr/bin/python3
import argparse
import gym
import numpy as np
import datetime
import tensorflow as tf
import gym_fetch_base_motions
from network_models.policy_net import Policy_net
from network_models.discriminator import Discriminator
from algo.ppo import PPOTrain


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='log directory', default='log/train/gail')
    parser.add_argument('--savedir', help='save directory', default='trained_models/gail')
    parser.add_argument('--gamma', default=0.95)
    parser.add_argument('--iteration', default=int(5e4))
    parser.add_argument('--env', default='FetchBase-v0')
    parser.add_argument('--obs', default='observations.csv')
    parser.add_argument('--acs', default='actions.csv')
    parser.add_argument('--log_actions', default='log_actions')
    parser.add_argument('--max_reward', default=3, type=int)
    parser.add_argument('--success_num', default=20, type=int)
    return parser.parse_args()


def store_actions(iteration, inner_iter,  actions, dir='log_actions'):
    filename = dir + '/' + str(iteration) + '_' + str(inner_iter)
    print('file name=', filename)
    np.savez(filename, acs=actions)


def main(args):
    env = gym.make(args.env)
    env.seed(0)
    ob_space = env.observation_space
    ac_space = env.action_space
    
    Policy = Policy_net('policy', env)
    Old_Policy = Policy_net('old_policy', env)
    PPO = PPOTrain(Policy, Old_Policy, env.action_space.shape, gamma=args.gamma)
    D = Discriminator(env)

    expert_observations = np.genfromtxt('trajectory/'+ args.obs)
    expert_actions = np.genfromtxt('trajectory/'+ args.acs, dtype=np.int32)
    print('Expert actions shape=', expert_actions.shape)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(args.logdir, sess.graph)
        sess.run(tf.global_variables_initializer())

        obs = env.reset()
        success_num = 0
        actions_to_log = []

        for iteration in range(args.iteration):
            print('current iteration=', iteration)
            observations = []
            actions = []
            # do NOT use rewards to update policy
            rewards = []
            v_preds = []
            run_policy_steps = 0
            inner_iter = 0
            while True:
                #env.render()
                if run_policy_steps % 10000 == 0:
                    print('current policy step=', run_policy_steps)
                run_policy_steps += 1
                obs = np.stack([obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs
                act, v_pred = Policy.act(obs=obs, stochastic=True)
                print('Action = ', act, 'State Value', v_pred)
                #act = np.asscalar(act)
                v_pred = np.asscalar(v_pred)
                #we use constant velocity 2.5
                _act = [act[0],act[1], act[2], 2.5]
                next_obs, reward, done, info = env.step(act)
                
                observations.append(obs)
                actions.append(act)
                rewards.append(reward)
                v_preds.append(v_pred)
                actions_to_log.append(act)

                done = env._is_success(None, None)
                print('Is done?=', done)
                if run_policy_steps % 100000 == 0:
                    print('current obs = ', obs)
                    print('current reward = ', reward)
                    print('current action = ', act)
                    print('Action = ', _act, 'State Value', v_pred)
                    print('Success?', env._is_success(None, None))
                    print('Sum of rewards = ', sum(rewards))

                if env._is_success(None, None):
                    print('Got enough reward. done! party times :D')
                
                if run_policy_steps % 10000 == 0:
                    inner_iter+=1
                    store_actions(iteration, inner_iter, actions_to_log)
                    actions_to_log = []

                if done:
                    print('Done and prepare feeding the Value-NN')
                    next_obs = np.stack([next_obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs
                    _, v_pred = Policy.act(obs=next_obs, stochastic=True)
                    v_preds_next = v_preds[1:] + [np.asscalar(v_pred)]
                    print('Predicted next values length=', len(v_preds_next))
                    obs = env.reset()
                    break
                else:
                    obs = next_obs

            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_length', simple_value=run_policy_steps)])
                               , iteration)
            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_reward', simple_value=sum(rewards))])
                               , iteration)

            if sum(rewards) >= args.max_reward:
                success_num += 1
                if success_num >= args.success_num:
                    saver.save(sess, args.savedir + '/model_' + str(datetime.date.today()) +  '.ckpt')
                    print('**************** Clear!! Model saved.*********************')
                    break
            else:
                success_num = 0

            # convert list to numpy array for feeding tf.placeholder
            observations = np.reshape(observations, newshape=[-1] + list(ob_space.shape))
            actions = np.array(actions).astype(dtype=np.int32)

            # train discriminator

            for i in range(2):
                print('~~~~~~~~~~~~~~~~~~~~Training the discriminator now ~~~~~~~~~~~~~~~~~~~~')
                print('Length of the expert observations=', expert_observations.shape)
                print('Length of the expert actions=', expert_actions.shape)
                print('Length of the learner observations=', len(observations))
                print('Length of the learner actions=', len(actions))
              #  reshaped = expert_actions.reshape(4500, 4)
                D.train(expert_s=expert_observations,
                        expert_a=expert_actions,
                        agent_s=observations,
                        agent_a=actions)

            # output of this discriminator is reward
            d_rewards = D.get_rewards(agent_s=observations, agent_a=actions)
            d_rewards = np.reshape(d_rewards, newshape=[-1]).astype(dtype=np.float32)
            print('rewards got from the discriminator', d_rewards)
            gaes = PPO.get_gaes(rewards=d_rewards, v_preds=v_preds, v_preds_next=v_preds_next)
            gaes = np.array(gaes).astype(dtype=np.float32)
            # gaes = (gaes - gaes.mean()) / gaes.std()
            v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)

            # train policy
            inp = [observations, actions, gaes, d_rewards, v_preds_next]
            PPO.assign_policy_parameters()
            for epoch in range(6):
                print('*******************Optimizing policy on epoch={}**************************'.format(epoch))
                sample_indices = np.random.randint(low=0, high=observations.shape[0],
                                                   size=32)  # indices are in [low, high)
                sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                PPO.train(obs=sampled_inp[0],
                          actions=sampled_inp[1],
                          gaes=sampled_inp[2],
                          rewards=sampled_inp[3],
                          v_preds_next=sampled_inp[4])

            summary = PPO.get_summary(obs=inp[0],
                                      actions=inp[1],
                                      gaes=inp[2],
                                      rewards=inp[3],
                                      v_preds_next=inp[4])

            writer.add_summary(summary, iteration)
        writer.close()


if __name__ == '__main__':
    args = argparser()
    main(args)
