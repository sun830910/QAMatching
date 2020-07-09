# -*- coding: utf-8 -*-

"""
Created on 2020-05-28 17:48
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

class test(object):
    def __init__(self):
        self.config = Config()

    def main(self):
        max_len = self.config.get('testing_rule', 'max_len')
        batch_size = self.config.get('testing_rule', 'batch_size')
        dev_path = self.config.get('data', 'dev_set')
        model_path = self.config.get('result', 'model_path')
        dict_path = self.config.get('result', 'word_dict_path')

        # 加载训练后的词向量
        preprocess.load_word_dict(dict_path)
        # 获取开发集数据
        dev_q, dev_a, dev_tag = preprocess.get_dev_data(dev_path)

        dev_q = pad_sequences(numpy.array(dev_q), max_len)
        dev_a = pad_sequences(numpy.array(dev_a), max_len)
        dev_tag = numpy.array(dev_tag)

        all_tags = []
        index = []
        for k in range(len(dev_tag)):
            index += [k] * len(dev_tag[k])
            all_tags += dev_tag[k]

        model = keras.models.load_model(model_path)

        preds = model.predict([dev_q, dev_a, dev_a], batch_size=batch_size)
        mrr = preprocess.evaluate_mrr(dev_tag, preds)
        print("mrr = ", mrr)

        preds = [t[0] for t in preds[1]]

        pred_dev = open('pred_dev.txt', 'w')
        for i in range(len(preds)):
            pred_dev.write(str(int(index[i])) + ', ' + str(all_tags[i]) + ', ' + str(preds[i]) + '\n')

        pred_dev.close()

        scores = open('score.txt', 'w')

        idx = 0
        cnt = 0
        score = 0
        for tags in dev_tag:
            pred = preds[idx: idx + len(tags)]
            idx += len(tags)
            p = numpy.argsort(pred)
            orders = numpy.zeros(len(p))
            for k in range(len(p)):
                orders[p[k]] = len(p) - k
            for k in range(len(tags)):
                if tags[k] == 1:
                    cnt += 1
                    score += 1. / orders[k]
                    scores.write(str(int(orders[k])) + "\t" + str(score / cnt) + '\n')
        scores.close()

test = test()
test.main()