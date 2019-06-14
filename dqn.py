# -*- coding: utf-8 -*-
import numpy as np
import sys
import keras.models
import keras.backend as K
import random


    


class Agent:
    def __init__(self,network_model, q_values_func,memoryD,test_or_train,agent_model):
        self.q_network=network_model
        
        self.target_network=keras.models.clone_model(network_model)
        self.target_network.set_weights(network_model.get_weights())
        
        self.target_q_values_func=K.function([self.target_network.layers[0].input], [self.target_network.layers[3].output])

        
        self.q_values_func=q_values_func
        self.memoryD=memoryD
        
        self.history_frame=[None]*4
        self.num_step=0
        self.update_times=0
        
        self.test_or_train=test_or_train
        self.agent_model=agent_model
    
    def do_compile(self,optimizer,loss_func):
        self.q_network.compile(optimizer=optimizer, loss=loss_func)
        self.target_network.compile(optimizer=optimizer, loss=loss_func)

    def load_weights(self, weights_file_name):
        self.q_network.load_weights(weights_file_name)
        self.update_target_network()

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())
    
    def greedy_policy(self,q_values):
        return np.argmax(q_values)
    
    
    def greedy_epsilon_policy(self,q_values):
        epsilon=0.05
        rnd = random.random()
        if rnd<=epsilon:
            return random.randint(0, 4)
        
        return np.argmax(q_values)

    def linear_greedy_epsilon_policy(self,q_values):
        if self.num_step>949000:
            epsilon=0.05
        else:
            epsilon=0.999-0.000001*self.num_step
            
        rnd = random.random()
        if rnd<=epsilon:
            return random.randint(0, 4)
        
        return np.argmax(q_values)
    
    
    def select_action(self,state):
        state=np.expand_dims(np.asarray(state),0)
        q_values=self.q_values_func([state])[0]
        
        if self.test_or_train=='train':
            #return self.greedy_epsilon_policy(q_values)
            return self.linear_greedy_epsilon_policy(q_values)
        elif self.test_or_train=='test':
            return self.greedy_policy(q_values)
        else:
            print('In dqn.py, select_action function, wrong model!')
            sys.exit(0)
    
    def transform_actions(self,actions):
        one_hot_action = np.zeros((len(actions), 5), dtype='float32')
        one_hot_action[np.arange(len(actions), dtype='int'), actions] = 1

        return one_hot_action
    
    def process_new_frame(self,frame):
        if self.history_frame[0]==None:
            self.history_frame=[frame,frame,frame,frame]
        else:
            self.history_frame[0:3]=self.history_frame[1:]
            self.history_frame[-1]=frame
        return np.array(self.history_frame)
    
    
    
    
    def update(self):
        if self.agent_model=='dqn':
            self.update_dqn()
        elif self.agent_model=='ddqn':
            self.update_ddqn()
    
    
    def update_dqn(self):
        states, actions, rewards, new_states, is_terminals=self.memoryD.get_sample()
        # i.e.  action 3 -->  [0,0,0,1,0]
        actions=self.transform_actions(actions)
        
        q_values=self.target_q_values_func([new_states])[0]
        max_q_values = np.max(q_values, axis=1)
        

        max_q_values[is_terminals] = 0
        targets = rewards + 0.99* max_q_values
        targets = np.expand_dims(targets, axis=1)
        self.q_network.train_on_batch([states, actions], targets)
        
        if self.num_step%19000==0:
            self.update_times+=1
            self.update_target_network()
            if self.update_times>0 and self.update_times%30==0:
                self.q_network.save_weights('./model_weights_round_%d.h5' % (self.update_times))


    def update_ddqn(self):
        states, actions, rewards, new_states, is_terminals=self.memoryD.get_sample()
        # i.e.  action 3 -->  [0,0,0,1,0]
        actions=self.transform_actions(actions)
        

        q_values=self.q_values_func([new_states])[0]
        max_actions = np.argmax(q_values, axis=1)
        

        tmp=np.arange(0,len(max_actions))
        index=np.stack((tmp,max_actions),axis=0)
        

        target_q_values=self.target_q_values_func([new_states])[0]
        max_q_values=target_q_values[list(index)]
        
        

        max_q_values[is_terminals] = 0
        targets = rewards + 0.99* max_q_values
        targets = np.expand_dims(targets, axis=1)
        self.q_network.train_on_batch([states, actions], targets)
        
        if self.num_step%19000==0:
            self.update_times+=1
            self.update_target_network()
            if self.update_times>0 and self.update_times%30==0:
                self.q_network.save_weights('./model_weights_round_%d.h5' % (self.update_times))

        
        
                
