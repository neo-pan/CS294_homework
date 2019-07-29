#!/usr/bin/env python

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
    
    def forward(self, input):
        with tf.variable_scope("Input"):
            output = tf.layers.dense(inputs=input, units=self.args.hidden_layer_size, 
                                    activation=tf.nn.relu)
        with tf.variable_scope("Hidden"):
            for _ in range(self.args.hidden_layer_depth):
                output = tf.layers.dense(inputs=output, units=self.args.hidden_layer_size, 
                                        activation=tf.nn.relu)
        with tf.variable_scope("Output"):
            output = tf.layers.dense(inputs=output, units=self.output_size, 
                                        activation=None)
        return output

    def cost(self, output, labels):
        loss = tf.losses.mean_squared_error(output, labels)

        # TODO: Implement 0-1 loss function
        # loss = tf.math.not_equal(output, tf.cast(labels, tf.float64))
        # loss = tf.reduce_sum(tf.cast(loss, tf.float64))

        return loss

    def create_model(self):
        pass

    def train(self, dataset, num_data):
        iters_per_epoch = ceil(num_data / self.args.batch_size)

        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(self.args.batch_size)
        dataset = dataset.repeat(self.args.num_epochs)

        iterator = dataset.make_one_shot_iterator()
        observations, actions  = iterator.get_next()

        output = self.forward(observations)
        loss = self.cost(output, actions)
        optimizer_op = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate).minimize(loss)
        
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()
        if self.args.first_train:
            with tf.Session() as sess:
                sess.run(init)
                for i in range(self.args.num_epochs):
                    print('--------EPOCH----{}--------'.format(i))
                    for i in range(iters_per_epoch):
                        loss_value, _ = sess.run([loss, optimizer_op])
                        print("Iter: {}, Loss: {:.4f}".format(i, loss_value))
                saver.save(sess, '/tmp/model.ckpt')

        return observations, output


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
    parser.add_argument('--hidden_layer_size', type=int, default=100,
                        help='Size of hidden layers')
    parser.add_argument('--hidden_layer_depth', type=int, default=2,
                        help='Depth of hidden layers')
    parser.add_argument('--learning_rate', type=int, default=0.001,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Size of mini-batch')

    args = parser.parse_args()

    print('loading expert data')
    with open(args.expert_data_file, 'rb') as f:
            expert_data = pickle.loads(f.read())
    print('expert data loaded:')

    for key,data in expert_data.items():
        if key=='observations':
            observations = data
            input_size = data.shape[-1]
            num_data = data.shape[0]
        if key=='actions':
            data = data.squeeze()
            actions = data
            output_size = data.shape[-1]
        print(key+':'+str(data.shape))
    
    dataset = tf.data.Dataset.from_tensor_slices((observations, actions))

    model = SimpleCloner(args, input_size, output_size)
    input, output = model.train(dataset, num_data)

    import gym
    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit

    returns = []
    observations = []
    actions = []

    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, '/tmp/model.ckpt')

    for i in range(args.num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = sess.run(output, feed_dict={input:obs[None,:]})
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

if __name__ == "__main__":
    main()
