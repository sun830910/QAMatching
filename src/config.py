# -*- coding: utf-8 -*-

"""
Created on 2020-05-28 14:30
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com
"""

class Config(object):
    def __init__(self):
        self.config_dict={
            'data':{
                'training_set': '../data/BoP2017-DBQA.train.txt',
                'dev_set': '../data/BoP2017-DBQA.dev.txt',
                'training_vec_result': '../data/training_word_dict_result.txt',
        },

            'embedding':{
                'sgns_wiki':'../embedding/sgns.wiki.word'
            },

            'training_rule':{
                "max_len": 64,
                "embedding_dim": 300,
                "lstm_unit": 512,
                "conv_unit": 64,
                "conv_step": 5,
                "conv_layer_num": 3,
                "epoches": 1,
                "batch_size": 128,
                "model_path": '../model/training_result_model',
                'word_dict_path': '../data/training_word_dict_result.txt'

            },

            'testing_rule':{
                "max_len": 64,
                "batch_size": 128,
                "model_path": '../model/training_result_model',
                'word_dict_path': '../data/training_word_dict_result.txt'

            }
        }
    def get(self, section, name):
        return self.config_dict[section][name]