import gym
import numpy as np
import tensorflow as tf
import argparse
import gym_fetch_base_motions
from network_models.policy_net import Policy_net


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', help='directory of model', default='trained_models')
    parser.add_argument('--alg', help='chose algorithm one of gail, ppo, bc', default='gail')
    parser.add_argument('--model', help='model name', default='')
    parser.add_argument('--logdir', help='log directory', default='log/test')
    parser.add_argument('--iteration', default=int(1e3))
    parser.add_argument('--stochastic', action='store_false')
    parser.add_argument('--env', help='openai enviroment', default='FetchBase-v0')
    return parser.parse_args()


def main(args):
    env = gym.make(args.env)
    env.seed(0)
    Policy = Policy_net('policy', env)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(args.logdir+'/'+args.alg, sess.graph)
        sess.run(tf.global_variables_initializer())
        if args.model == '':
            saver.restore(sess, args.modeldir+'/'+args.alg+'/'+'model.ckpt')
        else:
            saver.restore(sess, args.modeldir+'/'+args.alg+'/'+ args.model + '.ckpt')
        obs = env.reset()
        reward = 0
        success_num = 0

        for iteration in range(args.iteration):
            print('Test on iteration = ', iteration)
            rewards = []
            run_policy_steps = 0
            while True:  # run policy RUN_POLICY_STEPS which is much less than episode length
                env.render()
                run_policy_steps += 1
                obs = np.stack([obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs
                act, _ = Policy.act(obs=obs, stochastic=args.stochastic)

               # act = np.asscalar(act)
                print('Action from model=', act)

                rewards.append(reward)
                
                next_obs, reward, done, info = env.step(act)
                #done = env._success(None, None)
                if done:
                    obs = env.reset()
                    reward = -1
                    break
                else:
                    obs = next_obs

            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_length', simple_value=run_policy_steps)])
                               , iteration)
            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_reward', simple_value=sum(rewards))])
                               , iteration)

            # end condition of test
            if sum(rewards) >= 195:
                success_num += 1
                if success_num >= 100:
                    print('Iteration: ', iteration)
                    print('Clear!!')
                    break
            else:
                success_num = 0

        writer.close()


if __name__ == '__main__':
    args = argparser()
    main(args)
