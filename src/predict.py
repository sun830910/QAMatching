# -*- coding: utf-8 -*-

"""
Created on 2020-07-09 13:53
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com
"""

import sys
sys.path.append("..")

import  tensorflow.keras as keras
import numpy
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import preprocess
from config import Config


class main(object):
    def __init__(self):
        self.config = Config()

    def predict(self, question, answer):
        max_len = self.config.get('testing_rule', 'max_len')
        model_path = self.config.get('result', 'model_path')
        dict_path = self.config.get('result', 'word_dict_path')

        # 加载训练后的词向量
        preprocess.load_word_dict(dict_path)

        model = keras.models.load_model(model_path)

        input_q = preprocess.sentence_to_words(question)
        input_a = preprocess.sentence_to_words(answer)

        preprocess.words_to_indexes(input_q)
        preprocess.words_to_indexes(input_a)

        input_q = pad_sequences(numpy.asarray([input_q]), max_len)
        input_a = pad_sequences(numpy.asarray([input_a]), max_len)

        result = model.predict([input_q, input_a, input_a])

        return result[1][0][0]

test = main()
print(test.predict('浙江大学人文学院有哪几个一级学科博士点？', '拥有中国语言文学、历史学和哲学3个一级学科博士点。'))
print(test.predict('浙江大学人文学院有哪几个一级学科博士点？', '有古籍研究所、韩国研究所、日本文化研究所等研究所。'))
