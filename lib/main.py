#!/usr/bin/env python

'''
main code for running
'''

import sys

sys.path.append("/u/lambalex/DeepLearning/rl_hw_5")
sys.path.append("/u/lambalex/DeepLearning/rl_hw_5/lib")

import theano
import theano.tensor as T
from nn_layers import fflayer, param_init_fflayer
from utils import init_tparams, join2, srng, dropout, inverse_sigmoid
import lasagne
import numpy.random as rng
import numpy as np
#import matplotlib.pyplot as plt

from collections import OrderedDict
import gym
import logging
import cPickle
import numpy as np
import argparse

action_size = 1
state_size = 3
reward_size = 1
nfp = 128
nfe = 128
ns = 200
mb = 64
simulated_reward = simulated_loss = -999

print "ns", ns

#state -> action
def init_params_policy(p):

    p = param_init_fflayer(options={},params=p,prefix='pn_1',nin=state_size,nout=nfp,ortho=False,batch_norm=False)

    p = param_init_fflayer(options={},params=p,prefix='pn_2',nin=nfp,nout=nfp,ortho=False,batch_norm=False)

    p = param_init_fflayer(options={},params=p,prefix='pn_3_mu',nin=nfp,nout=action_size,ortho=False,batch_norm=False)
    p = param_init_fflayer(options={},params=p,prefix='pn_3_sigma',nin=nfp,nout=action_size,ortho=False,batch_norm=False)

    return init_tparams(p)

#state, action -> next_state, reward
def init_params_envsim(p):

    p = param_init_fflayer(options={},params=p,prefix='es_1',nin=action_size+state_size,nout=nfe,ortho=False,batch_norm=True)
    p = param_init_fflayer(options={},params=p,prefix='es_2',nin=nfe,nout=nfe,ortho=False,batch_norm=True)
    p = param_init_fflayer(options={},params=p,prefix='es_state',nin=nfe,nout=state_size,ortho=False,batch_norm=False)
    p = param_init_fflayer(options={},params=p,prefix='es_reward',nin=nfe,nout=reward_size,ortho=False,batch_norm=False)

    return init_tparams(p)

def policy_network(p,state):

    inp = state

    h1 = fflayer(tparams=p,state_below=inp,options={},prefix='pn_1',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',layer_norm=True)

    h2 = fflayer(tparams=p,state_below=h1,options={},prefix='pn_2',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',layer_norm=False)

    action_mu = fflayer(tparams=p,state_below=h2,options={},prefix='pn_3_mu',activ='lambda x: x',batch_norm=False)

    action_sigma = fflayer(tparams=p,state_below=h2,options={},prefix='pn_3_sigma',activ='lambda x: x',batch_norm=False)

    action = action_mu + action_sigma#T.tanh(action_mu) + 1.0 * T.tanh(action_sigma)# * srng.normal(size=action_sigma.shape)

    return action

def envsim_network(p,state,action):

    inp = join2(state,action)

    h1 = fflayer(tparams=p,state_below=inp,options={},prefix='es_1',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',layer_norm=True)

    h2 = fflayer(tparams=p,state_below=h1,options={},prefix='es_2',activ='lambda x: tensor.nnet.relu(x,alpha=0.02)',layer_norm=True)

    next_state = fflayer(tparams=p,state_below=h2,options={},prefix='es_state',activ='lambda x: x',batch_norm=False)

    reward = fflayer(tparams=p,state_below=h2,options={},prefix='es_reward',activ='lambda x: x',batch_norm=False)

    return next_state, reward


def net_simulated_chain(params_envsim, params_policy, initial_state, num_steps):


    initial_reward = theano.shared(np.zeros(shape=(mb,1)).astype('float32'))

    def one_step(last_state,last_reward):

        next_action = policy_network(params_policy, last_state)
        next_state, next_reward = envsim_network(params_envsim, last_state, next_action)
        
        return next_state, next_reward

    [state_lst, reward_lst], _ = theano.scan(fn=one_step,outputs_info=[initial_state, initial_reward],n_steps=num_steps)

    return reward_lst

params_policy = init_params_policy({})
params_envsim  = init_params_envsim({})

env = gym.make('Pendulum-v0')

state = T.matrix()
action = policy_network(params_policy, state)
#action = theano.function([state, tparams_policy],[next_action])
compute_action = theano.function([state], outputs = action)

def real_chain(init_state,policy_noise, num_steps):

    # sample action from the policy network
    # pass the action to the simulator network

    action_lst = []
    reward_lst = []

    last_state = init_state.reshape((1, state_size))

    state_lst = [last_state]

    for i in range(num_steps):
        action = compute_action(last_state)
        # env.render()
        action = action.reshape((action_size))
        action = action*0.0 + rng.uniform(-2.0, 2.0,size=action.shape).astype('float32')
        state, reward, done, info = env.step(action)
        
        state = state.reshape((1,state_size)).astype('float32')
        reward = reward.reshape((1,reward_size)).astype('float32')
        action_lst.append(action)
        state_lst.append(state)
        reward_lst.append(reward)
        last_state = state

    return action_lst, state_lst, reward_lst


#<<<<<<< HEAD

#tstate = T.matrix()
#taction = T.matrix()


#next_action = policy_network(params_policy, state)

#next_state, reward = envsim_network(params_envsim, state, action)


#=======


#print 'action list ', action_lst
#print 'state_list, ', state_lst
#print 'rewrd_list, ', reward_lst



initial_state_sim = T.matrix()

simulated_reward_lst = net_simulated_chain(params_envsim, params_policy, initial_state_sim,ns)

simulated_total_reward = T.mean(simulated_reward_lst)
simulation_loss = -1.0 * simulated_total_reward

reward_delta = (simulated_reward_lst[-1] - simulated_reward_lst[0]).mean()

simulation_updates = lasagne.updates.adam(simulation_loss, params_policy.values(), learning_rate=0.0001)

policy_grad = T.sum(T.grad(simulation_loss, params_policy.values()[0]))

simulation_function = theano.function(inputs=[initial_state_sim],outputs=[simulated_total_reward, simulation_loss,reward_delta, policy_grad],updates=simulation_updates)

#next_state_real, rewards_real = envsim_network(params_envism, state, action)

#next_state_truth = T.vector()
#rewards_truth = T.scalar()

#real_loss = (next_state_truth - next_state_real) ** 2 + (rewards_truth - rewards_real) ** 2

#real_updates = lasagne.updates.adam(real_loss, params_envsim.values())

#compute_rewards = theano.function(inputs=[state, action], outputs=[rewards, next_state])

#num_steps = 20
#state = intial_state
#loss_lst = []

# compute loss for envoriment simulator for the real chain
#for i in range(num_steps):
#    reward, state = compute_rewards(state,action_list[0])
#    loss = (state_list[i]- state) ** 2 + (reward_list[i] - reward) ** 2
#    loss_lst.append(loss)
    

state = T.matrix()
action = T.matrix()


########################################################################
#Build method for training the environment simulator
########################################################################
last_state_envtr = T.matrix()
action_envtr = T.matrix()
next_state_envtr = T.matrix()
reward_envtr = T.matrix()

next_state_pred, next_reward_pred = envsim_network(params_envsim, last_state_envtr, action_envtr)

envtr_loss = T.mean(T.sqr(next_state_pred - next_state_envtr)) + T.mean(T.sqr(next_reward_pred - reward_envtr))

envtr_updates = lasagne.updates.adam(envtr_loss, params_envsim.values(), learning_rate=0.0001,beta1=0.5)

train_envsim = theano.function(inputs = [last_state_envtr,action_envtr,next_state_envtr,reward_envtr], outputs = [envtr_loss,next_state_pred,next_reward_pred,T.grad(T.sum(next_reward_pred),last_state_envtr),T.grad(T.sum(next_reward_pred),action_envtr)], updates = envtr_updates)

run_envsim = theano.function(inputs = [last_state_envtr,action_envtr], outputs = [next_reward_pred, T.grad(T.sum(next_reward_pred),last_state_envtr),T.grad(T.sum(next_reward_pred),action_envtr)])

all_loss_lst = []
all_expreward_lst = []

for iteration in range(0,50000): 
    loss_list = []
    
    action_set = []
    state_set = []
    reward_set = []
    policy_noise = 0.1
    for i in range(mb):
        init_state = env.reset().astype(np.float32)
        action_list, state_list, reward_list = real_chain(init_state, policy_noise, ns)
        action_set.append(action_list)
        state_set.append(state_list)
        reward_set.append(reward_list)

    action_set = np.array(action_set).transpose(1,0,2).reshape((ns,64,action_size))
    state_set = np.array(state_set).transpose(1,0,2,3).reshape((ns+1,64,state_size))
    reward_set = np.array(reward_set).transpose(1,0,2,3).reshape((ns,64,reward_size))

    #print 'action_list ', action_list
    #print 'state_list ', state_list
    #print 'reward_list ', reward_list
    for i in range(ns):
        
        loss,next_state_pred,next_reward_pred,state_grad,action_grad = train_envsim(state_set[i], action_set[i], state_set[i+1], reward_set[i])
        loss_list.append(loss)

        if False and iteration % 100 == 1:
            print "==========================="
            print "step", i
            print "last state", state_set[i][0]
            print "action", action_set[i][0]
            print "next state true", state_set[i+1][0]
            print "reward true", reward_set[i][0]
            print "next state pred", next_state_pred[0]
            print "next reward pred", next_reward_pred[0]
            print "state grad", state_grad.round(2)
            print "action grad", action_grad.round(2)
            

    loss_val = np.array(loss_list).mean()
    all_loss_lst.append(loss_val)

    if iteration % 10 == 1:
        print 'loss_list ', iteration, loss_val

    #need to get some real initial states

    if loss_val < 0.2: 
        initial_state_sim = state_set[0]

        simulated_reward, simulated_loss,reward_delta,policy_grad = simulation_function(initial_state_sim)
        all_expreward_lst.append(simulated_reward)
        if iteration % 10 == 1:
            print "simulated loss", simulated_loss
            print "policy grad", policy_grad
    
        if iteration % 5000 == 0:
            pass
            #plt.plot(all_loss_lst)
            #plt.title("Environment Simulator Loss over number of iterations")
            #plt.show()

            #plt.plot(all_expreward_lst)
            #plt.title("Expected Reward over number of policy gradient iterations")
            #plt.show()




