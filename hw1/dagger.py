#!/usr/bin/env python

"""
Example usage:
    python dagger.py experts/Humanoid-v2.pkl expert_data/Humanoid-v2.pkl Humanoid-v2 --first_train
"""

import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
from behavioral_cloning import SimpleCloner, evaluate_model

def labeling(args, observations, policy_fn):
    print("Begin Labeling")
    with tf.Session():
        tf_util.initialize()
        actions = []
        for ob in observations:
            action = policy_fn(ob[None,:])
            actions.append(action)

    assert len(observations) == len(actions)
    new_dataset = tf.data.Dataset.from_tensor_slices((np.array(observations), np.array(actions).squeeze()))
    print('End Labeling')
    return new_dataset

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('expert_data_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--first_train', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--hidden_layer_size', type=int, default=512,
                        help='Size of hidden layers')
    parser.add_argument('--hidden_layer_depth', type=int, default=2,
                        help='Depth of hidden layers')
    parser.add_argument('--learning_rate', type=int, default=0.001,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Size of mini-batch')
    parser.add_argument('--dagger_epochs', type=int, default=20, 
                        help='Number of DAgger epochs')
    parser.add_argument('--model_name', type=str, default='DAgger',
                        help='Name of this model')

    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

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

    for i in range(args.dagger_epochs):
        print('--------No {}: DAgger Epoch--------'.format(i))
        model.train(dataset)
        observations = evaluate_model(args, model)
        print('Number of new data: {}'.format(len(observations)))
        new_dataset = labeling(args, observations, policy_fn)
        dataset = dataset.concatenate(new_dataset)
        print('--------No {}: DAgger Epoch--------'.format(i))

def dataset_test(dataset):
    with tf.Session() as sess:
        dataset = dataset.shuffle(buffer_size=10000)
        # dataset = dataset.batch(128)
        iterator = dataset.make_one_shot_iterator()
        observation, action = iterator.get_next()
        batch = 0
        while True:
            try:
                sess.run([observation, action])
                batch += 1
            except tf.errors.OutOfRangeError:
                print("Number of data in dataset: {}".format(batch))
                break

if __name__ == "__main__":
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    main()
