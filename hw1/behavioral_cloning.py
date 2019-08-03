#!/usr/bin/env python

"""
Example usage:
    python behavioral_cloning.py expert_data/Humanoid-v2.pkl Humanoid-v2 --first_train
"""

import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
from math import ceil

class SimpleCloner:
    def __init__(self, args, input_size, output_size):
        self.args = args
        self.input_size = input_size
        self.output_size = output_size
        self.create_model()
    
    def forward(self, inputs):
        with tf.variable_scope("Input", reuse=tf.AUTO_REUSE):
            outputs = tf.layers.dense(inputs=inputs, units=self.args.hidden_layer_size, 
                                    kernel_initializer=tf.initializers.variance_scaling(
                                    scale=2., mode="fan_in", distribution="truncated_normal", seed=None), 
                                    activation=tf.nn.relu)
        with tf.variable_scope("Hidden", reuse=tf.AUTO_REUSE):
            for _ in range(self.args.hidden_layer_depth):
                outputs = tf.layers.dense(inputs=outputs, units=self.args.hidden_layer_size, 
                                        kernel_initializer=tf.initializers.variance_scaling(
                                        scale=2., mode="fan_in", distribution="truncated_normal", seed=None), 
                                        activation=tf.nn.relu)
        with tf.variable_scope("Output", reuse=tf.AUTO_REUSE):
            outputs = tf.layers.dense(inputs=outputs, units=self.output_size, 
                                        activation=None)
        return outputs

    def cost(self, outputs, labels):
        loss = tf.losses.mean_squared_error(outputs, labels)

        # TODO: Implement 0-1 loss function
        # loss = tf.math.not_equal(output, tf.cast(labels, tf.float64))
        # loss = tf.reduce_sum(tf.cast(loss, tf.float64))

        return loss

    def create_model(self):
        self.inputs_ph = tf.placeholder(dtype=tf.float64, shape=[None, self.input_size], name='inputs')
        self.labels_ph = tf.placeholder(dtype=tf.float64, shape=[None, self.output_size], name='labels')
        self.outputs_ph = self.forward(self.inputs_ph)
        self.loss_ph = self.cost(self.outputs_ph, self.labels_ph)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate)

    def get_model(self):
        return self.inputs_ph, self.outputs_ph

    def train(self, dataset):
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(self.args.batch_size)

        iterator = dataset.make_initializable_iterator()
        observation, action = iterator.get_next()

        optimizer_op = self.optimizer.minimize(self.loss_ph)

        init = tf.global_variables_initializer()
        # vl = [v for v in tf.global_variables() if "Adam" not in v.name] 
        # saver = tf.train.Saver(var_list=vl)
        saver = tf.train.Saver()

        with tf.Session() as sess:
            if self.args.first_train:
                sess.run(init)
                self.args.first_train = False
            else:
                sess.run(init)
                saver.restore(sess, './tmp/{}'.format(self.args.model_name))
                
            for i_e in range(self.args.num_epochs):
                i_i = 0
                sess.run(iterator.initializer)
                while True:
                    try:
                        o, a = sess.run([observation, action])
                        loss_value, _ = sess.run([self.loss_ph, optimizer_op], 
                                                feed_dict={self.inputs_ph:o, self.labels_ph:a})
                        i_i += 1
                    except tf.errors.OutOfRangeError:
                        print("Epoch: {}, Iter: {}, Loss: {:.4f}".format(i_e, i_i, loss_value))
                        break
            saver.save(sess, './tmp/{}'.format(self.args.model_name))

def evaluate_model(args, model):
    import gym
    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit

    returns = []
    observations = []
    actions = []

    inputs, outputs = model.get_model()
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, './tmp/{}'.format(args.model_name))
    for i in range(args.num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = sess.run(outputs, feed_dict={inputs:obs[None,:]})
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if args.render:
                env.render()
            if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))
    sess.close()
    return observations

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_data_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--first_train', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--hidden_layer_size', type=int, default=512,
                        help='Size of hidden layers')
    parser.add_argument('--hidden_layer_depth', type=int, default=2,
                        help='Depth of hidden layers')
    parser.add_argument('--learning_rate', type=int, default=0.001,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Size of mini-batch')
    parser.add_argument('--model_name', type=str, default='BehavioralCloning',
                        help='Name of this model')

    args = parser.parse_args()

    print('loading expert data')
    with open(args.expert_data_file, 'rb') as f:
            expert_data = pickle.loads(f.read())
    print('expert data loaded:')

    for key,data in expert_data.items():
        if key=='observations':
            observations = data
            input_size = data.shape[-1]
        if key=='actions':
            data = data.squeeze()
            actions = data
            output_size = data.shape[-1]
        print(key+':'+str(data.shape))
    
    dataset = tf.data.Dataset.from_tensor_slices((observations, actions))

    model = SimpleCloner(args, input_size, output_size)
    model.train(dataset)

    _ = evaluate_model(args, model)

if __name__ == "__main__":
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    main()
