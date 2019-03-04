import tensorflow as tf
import gym


class Discriminator:
    def __init__(self, env, env_obs_shape):
        """
        :param env:
        Output of this Discriminator is reward for learning agent. Not the cost.
        Because discriminator predicts  P(expert|s,a) = 1 - P(agent|s,a).
        """
        print(env)
        print('[Discriminator] observation_space={}'.format(env_obs_shape))



        with tf.variable_scope('discriminator'):
           
            self.scope = tf.get_variable_scope().name
            self.expert_s = tf.placeholder(dtype=tf.float32, shape=[None, env_obs_shape], name='expert_obs')
            self.expert_a = tf.placeholder(dtype=tf.int32, shape=[None] , name='expert_act')

            self.agent_s = tf.placeholder(dtype=tf.float32, shape=[None, env_obs_shape], name='agent_obs')
            self.agent_a = tf.placeholder(dtype=tf.int32, shape=[None], name='agent_act')

            _depth = 0

            if  isinstance(env.action_space, gym.spaces.Box):
                _depth = env.action_space.shape[0]
                #create new placeholder according to the Box space
                self.expert_a = tf.placeholder(dtype=tf.float32, shape=[None] + list(env.action_space.shape), name='expert_act')
                self.agent_a = tf.placeholder(dtype=tf.float32, shape=[None] + list(env.action_space.shape), name='agent_act')

            elif isinstance(env.action_space, gym.spaces.Discrete):
                _depth = env.action_space.n

            print('Creating expert observation placeholder=', self.expert_s)
            print('Creating expert actions placeholder=', self.expert_a)
            #expert_a_one_hot = tf.one_hot(self.expert_a, depth=_depth)
            #print('Creating expert one hot tensor placeholder=', expert_a_one_hot)
            
            # add noise for stabilise training
            self.expert_a += tf.random_normal(tf.shape(self.expert_a), mean=0.1, stddev=0.1, dtype=tf.float32)/1.2
            expert_s_a = tf.concat([self.expert_s, self.expert_a], axis=1)
            print('Concatenate two placeholders -> s_a=', expert_s_a)


           # agent_a_one_hot = tf.one_hot(self.agent_a, depth=_depth)
            
            # add noise for stabilise training
            #agent_a_one_hot += tf.random_normal(tf.shape(agent_a_one_hot), mean=0.1, stddev=0.1, dtype=tf.float32)/1.2
            self.agent_a += tf.random_normal(tf.shape(self.agent_a), mean=0.1, stddev=0.1, dtype=tf.float32)/1.2
            agent_s_a = tf.concat([self.agent_s, self.agent_a], axis=1)


            with tf.variable_scope('network') as network_scope:
                prob_1 = self.construct_network(input=expert_s_a)
                network_scope.reuse_variables()  # share parameter
                prob_2 = self.construct_network(input=agent_s_a)

            with tf.variable_scope('loss'):
                loss_expert = tf.reduce_mean(tf.log(tf.clip_by_value(prob_1, 0.01, 1)))
                loss_agent = tf.reduce_mean(tf.log(tf.clip_by_value(1 - prob_2, 0.01, 1)))
                loss = loss_expert + loss_agent
                loss = -loss
                tf.summary.scalar('discriminator_loss', loss)

            self.merged = tf.summary.merge_all()
            optimizer = tf.train.AdamOptimizer()
            self.train_op = optimizer.minimize(loss)
            self.rewards = tf.log(tf.clip_by_value(prob_2, 1e-10, 1))  # log(P(expert|s,a)) larger is better for agent

    def construct_network(self, input):
        layer_1 = tf.layers.dense(inputs=input, units=20, activation=tf.nn.leaky_relu, name='layer1')
        layer_2 = tf.layers.dense(inputs=layer_1, units=20, activation=tf.nn.leaky_relu, name='layer2')
        layer_3 = tf.layers.dense(inputs=layer_2, units=20, activation=tf.nn.leaky_relu, name='layer3')
        prob = tf.layers.dense(inputs=layer_3, units=1, activation=tf.sigmoid, name='prob')
        return prob

    def train(self, expert_s, expert_a, agent_s, agent_a):
        return tf.get_default_session().run(self.train_op, feed_dict={self.expert_s: expert_s,
                                                                      self.expert_a: expert_a,
                                                                      self.agent_s: agent_s,
                                                                      self.agent_a: agent_a})

    def get_rewards(self, agent_s, agent_a):
        print("+++++++++++ calculating  discriminator rewards ++++++++++")
        return tf.get_default_session().run(self.rewards, feed_dict={self.agent_s: agent_s,
                                                                     self.agent_a: agent_a})

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_summary(self, ex_obs, ex_acs, a_obs, a_acs):
        return tf.get_default_session().run(self.merged, feed_dict={self.expert_s: ex_obs,
                                                                    self.expert_a: ex_acs,
                                                                    self.agent_s: a_obs,
                                                                    self.agent_a: a_acs})

