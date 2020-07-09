# -*- coding: utf-8 -*-

"""
Created on 2020-05-28 14:12
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com
"""

import sys
sys.path.append("..")
import numpy
from tensorflow.keras import layers
from tensorflow.keras.layers import \
     Embedding, Lambda, GlobalMaxPooling1D, Dot, BatchNormalization, Activation
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import preprocess
from config import Config
from tensorflow.keras.layers import Input, LSTM ,Conv1D,Bidirectional
from tensorflow.keras.models import *


class model(object):
    def __init__(self):
        self.config = Config()

    def get_training_embedding(self):
        """
        获得训练集embediing matrix
        :return:
        """
        # 加载
        preprocess.load_training_qa([self.config.get('data', 'training_set')])
        embedding_matrix = preprocess.get_embedding_matrix(
            self.config.get('embedding', 'sgns_wiki'), self.config.get('training_rule', 'embedding_dim')
        )
        return embedding_matrix

    def get_dev_embedding(self):
        dev_q, dev_a, dev_tag = preprocess.get_dev_data(self.config.get('data', 'dev_set'))
        max_len = self.config.get('training_rule', 'max_len')
        dev_q = pad_sequences(numpy.array(dev_q), max_len)
        dev_a = pad_sequences(numpy.array(dev_a), max_len)
        dev_tag = numpy.array(dev_tag)
        return dev_q, dev_a, dev_tag


    def biLSTM_CNN_model(self):
        max_len = self.config.get('training_rule', 'max_len')
        embedding_dim = self.config.get('training_rule', 'embedding_dim')
        embedding_matrix = self.get_training_embedding()
        lstm_unit = self.config.get('training_rule', 'lstm_unit')
        conv_layer_num = self.config.get('training_rule', 'conv_layer_num')
        conv_unit = self.config.get('training_rule', 'conv_unit')
        conv_step = self.config.get('training_rule', 'conv_step')

        # 输入层
        input_q = Input((max_len,))
        input_a1 = Input((max_len,))
        input_a2 = Input((max_len,))

        # 对输入层进行编码
        embedding_layer = Embedding(
            len(embedding_matrix), embedding_dim,
            weights=[embedding_matrix], input_length=max_len,
            trainable=False
        )

        # 编码层输出
        emb_q = embedding_layer(input_q)
        emb_a1 = embedding_layer(input_a1)
        emb_a2 = embedding_layer(input_a2)

        # 双向LSTM
        shared_lstm = Bidirectional(
            LSTM(lstm_unit, implementation=2, return_sequences=True)
        )

        # 双向LSTM输出
        encode_q = shared_lstm(emb_q)
        encode_a1 = shared_lstm(emb_a1)
        encode_a2 = shared_lstm(emb_a2)

        # 卷积层
        for k in range(conv_layer_num):
            conv = Conv1D(
                conv_unit, conv_step,
                padding='valid', strides=1
            )
            relu = Activation('relu')
            batchnorm = BatchNormalization()

            def bn_conv(x):
                return relu(batchnorm(conv(x)))

            # 做batchnorm 和 卷积
            encode_q = bn_conv(encode_q)
            encode_a1 = bn_conv(encode_a1)
            encode_a2 = bn_conv(encode_a2)

        # 池化
        vec_q = GlobalMaxPooling1D()(encode_q)
        vec_a1 = GlobalMaxPooling1D()(encode_a1)
        vec_a2 = GlobalMaxPooling1D()(encode_a2)

        # 计算相似度
        cosine_1 = Dot(axes=1, normalize=True)([vec_q, vec_a1])
        cosine_2 = Dot(axes=1, normalize=True)([vec_q, vec_a2])

        # 对一结果取反
        neg = Lambda(lambda x: -x, output_shape=lambda x: x)


        # 计算最终损失函数
        # sub对照训练时的输入为train_tag,若train_tag为1时，第1位的答案为正确答案，第2位的答案为错误答案
        sub = layers.add([cosine_1, neg(cosine_2)])

        # 模型接口
        model = Model(inputs=[input_q, input_a1, input_a2], outputs=[sub, cosine_1, cosine_2])
        # 损失为0.6*sub + 0.2*cosine_1 + 0.2*cosine_2
        model.compile(
            optimizer='adam', loss='mean_squared_error', loss_weights=[0.6, 0.2, 0.2]
        )
        print(model.summary())
        return model

    def train(self):
        model = self.biLSTM_CNN_model()
        max_len = self.config.get('training_rule', 'max_len')
        epoches = self.config.get('training_rule', 'epoches')
        batch_size = self.config.get('training_rule', 'batch_size')
        model_save_path = self.config.get('result', 'model_path')
        word_dict_save_path = self.config.get('result', 'word_dict_path')
        batch_per_epoch = len(preprocess.qa_list) // batch_size * 5
        for k in range(epoches):
            print('Epoch: %d' % k)
            # 取得训练集的问题、答案1、答案2与答案1和答案2的关系
            train_q, train_a1, train_a2, train_tag = \
            preprocess.get_train_samples(batch_size * batch_per_epoch, 0.25, 0.25)

            # 展平至等长
            train_q = pad_sequences(numpy.array(train_q), max_len)
            train_a1 = pad_sequences(numpy.array(train_a1), max_len)
            train_a2 = pad_sequences(numpy.array(train_a2), max_len)
            train_tag = numpy.array(train_tag)

            # 将train_tag转换成答案1与答案2的标签
            train_t1, train_t2 = preprocess.trans_tag(train_tag)

            model.fit(
                [train_q, train_a1, train_a2], [train_tag, train_t1, train_t2],
                batch_size=batch_size, epochs=1,
            )

            # 验证集计算模型MRR
            dev_q, dev_a, dev_tag = self.get_dev_embedding()
            # preds为[sub, cos1, cos2]
            preds = model.predict([dev_q, dev_a, dev_a], batch_size=batch_size)
            mrr = preprocess.evaluate_mrr(dev_tag, preds)
            print('\nmrr = ', mrr)

        model.save(model_save_path, overwrite=True)
        preprocess.save_word_dict(word_dict_save_path)

if __name__ == '__main__':
    test = model()
    test.train()

