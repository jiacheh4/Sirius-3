# -*- coding: utf-8 -*-
import numpy as np
import sys
import keras.models
import keras.backend as K
from cnn import cnn_model
import random


    


class Agent:
    def __init__(self,network_model, q_values_func,memoryD):
        self.q_network_model=network_model
        
        self.target_network=keras.models.clone_model(network_model)
        self.target_network.set_weights(network_model.get_weights())
        self.target_q_values_func=K.function([self.target_network.layers[0].input], [self.target_network.layers[5].output])
        
        self.q_values_func=q_values_func
        self.memoryD=memoryD
        
        self.history_frame=[None]*4
        self.num_step=0
        self.update_times=0
    
    def do_compile(self,optimizer,loss_func):
        self.q_network_model.compile(optimizer=optimizer, loss=loss_func)
        self.target_network.compile(optimizer=optimizer, loss=loss_func)

    def load_weights(self, weights_path):
        self.q_network_model.load_weights(weights_path)
        self.update_target_network()

    def update_target_network(self):
        self.target_network.set_weights(self.q_network_model.get_weights())
           
    def greedy_epsilon_policy(self,q_values):
        epsilon=0.05
        rnd = random.random()
        if rnd<=epsilon:
            return random.randint(0, 4)
        
        return np.argmax(q_values)
        
    
    def select_action(self,state):
        state=np.expand_dims(np.asarray(state),0)
        q_values=self.q_values_func([state])[0]
        return self.greedy_epsilon_policy(q_values)
    
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
        states, actions, rewards, new_states, is_terminals=self.memoryD.get_sample()
        # i.e.  action 3 -->  [0,0,0,1,0]
        actions=self.transform_actions(actions)
        
        q_values=self.target_q_values_func([new_states])[0]
        max_q_values = np.max(q_values, axis=1)
        #把是terminal的q_value置为0
        max_q_values[is_terminals] = 0
        # gamma衰减因子  设置 0.99
        targets = rewards + 0.99* max_q_values
        targets = np.expand_dims(targets, axis=1)
        #在一个batch的数据上进行一次参数更新
        #因为我们是以一个batch,aka 32个经验数组去update的
        self.q_network_model.train_on_batch([states, actions], targets)
        
        if self.num_step%10000==0:
            self.update_times+=1
            self.update_target_network()
        
        
        
    '''
    def fit(self):
        print('Start main loop***')
        self.memoryD.reset()
        count=0
        
        while count<100000:
            count+=1
            t=0
            total_reward=0
            frame=env.get_initial_frame()
            while True:
                self.num_step+=1
                t+=1
                
                state=self.process_new_frame(frame)
                action=self.select_action(state)
                new_frame,reward,is_terminal=env.get_everthing(action)

                total_reward+=reward
                
                self.memoryD.append(frame,action,reward,is_terminal)
                
                # 设定跑了400帧之后再开始train
                if self.num_step>400:
                    self.update()
                
                if is_terminal or t> 10000:
                    break
                
                frame=new_frame
        '''

        
                