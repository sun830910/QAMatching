# -*- coding: utf-8 -*-

"""
Created on 2020-05-28 09:44
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com
"""

import random
import numpy
import jieba
import re

# 标点符号禁用表
punc_list = [
    '、', '，', '。', '！', '？',
    '（', '）', '【', '】', '《', '》', '‘', '’', '“', '”',
    ',', '.', '!', '?', '(', ')', '[', ']', '{', '}', '\'', '\"'
]

# 词字典
word_dict = {}
# qa对
qa_list = []

def sentence_to_words(sentence):
    """
    对输入样本进行清洗后分词，并返回分词后结果（列表形式）
    :param sentence:输入样本，字符串形式
    :return: 句子经由分词后的数组形式
    """
    # words:返回结果
    words = []
    # 对读入sentence以""取开，成为数组形式
    items = sentence.split()
    # 遍历各个部分（问题、答案、标签）
    for item in items:
        # 对读入各个部分分词
        # item_words:分词后的数组形式
        item_words = list(jieba.cut(item))
        hint_word = 0
        # 对分词部分进行清洗与预处理
        for i in range(len(item_words)):
            # 去除[]内数字
            if (i + 2) < len(item_words) and item_words[i] == '[' and item_words[i + 2] == ']':
                # 判别是否为数字
                if item_words[i + 1].isdigit():
                    hint_word = 1
                    continue
            if hint_word != 0:
                hint_word = (hint_word + 1) % 3
                continue
            if item_words[i] in punc_list:
                continue
            words.append(item_words[i])
    return words

def words_to_indexes(items):
    """
    将数组形式的分词结果转换成索引形式的编码
    若items[i]这个词存在word_dict中，则将items[i]替换为word_dict中的索引，若没出现（OOV）则替换为0
    :param items:数组
    :return:
    """
    for i in range(len(items)):
        if items[i] in word_dict:
            items[i] = word_dict[items[i]]
        else:
            items[i] = 0


def update_tokens(tokens, words):
    """
    新增tokens
    :param tokens:tokens字典
    :param words:数组形式的分词结果
    :return:
    """
    for word in words:
        if word in tokens:
            tokens[word] += 1
        else:
            tokens.setdefault(word, 1)


def load_training_qa(filenames):
    """
    加载QA清单，并进行编码,用于加载训练集中的样本并用于训练时使用
    :param filenames:
    :return:
    """
    qa_table = {}
    # filenames：所有档案
    for filename in filenames:
        # 分个开启各个档案
        infile = open(filename, 'r', encoding='utf-8')
        # 分行读取各条数据
        for line in infile:
            # 以空格作为切片依据
            items = line.split('\t')
            # items[0]为label
            # items[1]为Question
            # items[2]为Answer

            # 若问题不在qa_table中
            if items[1] not in qa_table:
                # 将问题添加至qa_table，并新增一个包含两个列表的列表[[][]]
                qa_table.setdefault(items[1], [[], []])
            # 在qa_table中将label为1的部分新增答案
            qa_table[items[1]][items[0] != '0'].append(items[2])
            # qa_table中为[问题]
        infile.close()

    # 遍历qa_table
    for question_idx in qa_table:
        # 将问答对写如qa_list
        # question_idx:问题
        # qa_table[qu][0]:qa_table中标签为0的答案(假答案)
        # qa_table[qu][1]:qa_table中标签为1的答案(真答案)
        qa_list.append([question_idx, qa_table[question_idx][0], qa_table[question_idx][1]])
    # 此时qa_list为[[问题][假答案][真答案]]

    tokens = {}
    for question_idx in range(len(qa_list)):
        qa_list[question_idx][0] = sentence_to_words(qa_list[question_idx][0])
        update_tokens(tokens, qa_list[question_idx][0])
        for fake_answer_idx in range(len(qa_list[question_idx][1])):
            qa_list[question_idx][1][fake_answer_idx] = sentence_to_words(qa_list[question_idx][1][fake_answer_idx])
            update_tokens(tokens, qa_list[question_idx][1][fake_answer_idx])
        for real_answer_idx in range(len(qa_list[question_idx][2])):
            qa_list[question_idx][2][real_answer_idx] = sentence_to_words(qa_list[question_idx][2][real_answer_idx])
            update_tokens(tokens, qa_list[question_idx][2][real_answer_idx])

    # sorted_tokens:将tokens中出现的分词依照出现频次由高到低排序
    sorted_tokens = sorted(
        list(tokens.items()),
        key=lambda it: it[1], reverse=True
    )

    # word_dict:将sorted_tokens进行编码
    for i in range(len(sorted_tokens)):
        word_dict.setdefault(sorted_tokens[i][0], i + 1)

    # 将qa_list中的[问题][假答案][真答案]分别依序word_dict进行排序
    for question_idx in range(len(qa_list)):
        words_to_indexes(qa_list[question_idx][0])
        for fake_answer_idx in range(len(qa_list[question_idx][1])):
            words_to_indexes(qa_list[question_idx][1][fake_answer_idx])
        for real_answer_idx in range(len(qa_list[question_idx][2])):
            words_to_indexes(qa_list[question_idx][2][real_answer_idx])


def to_float_array(items):
    num_arr = []
    for item in items:
        if item != '':
            num_arr.append(float(item))
    return num_arr


def get_embedding_matrix(filename, embedding_dim, maxWords=1000000):
    """
    导入预训练向量
    :param filename: 预训练向量
    :param embedding_dim: 设定维度
    :param maxWords:
    :return:
    """
    embedding_dict = {}
    word = ''
    infile = open(filename, 'r', encoding='utf-8')
    firstLine = True
    cnt = 0
    for line in infile:
        if firstLine:
            firstLine = False
            continue
        cnt += 1
        if cnt > maxWords:
            break
        items = re.split(' ', line.strip())
        if len(items) == embedding_dim + 1:
            word = items[0]
            if word not in word_dict:
                num_words = len(word_dict)
                word_dict.setdefault(word, num_words + 1)
            embedding_dict.setdefault(word, to_float_array(items[1:]))
        elif len(items) != 0:
            print("ERROR! There are %d items" % len(items))
            print(items)
            print(cnt)
    infile.close()
    embedding_matrix = numpy.zeros(
        (len(word_dict) + 1, embedding_dim), dtype='float32'
    )
    for word, index in word_dict.items():
        embedding_vector = embedding_dict.get(word)
        if embedding_vector is None:
            embedding_vector = [random.uniform(-1, 1) for i in range(embedding_dim)]
            embedding_dict.setdefault(word, embedding_vector)
        if len(embedding_vector) != embedding_dim:
            print('Not padding embedding vector!')
        embedding_matrix[index] = numpy.asarray(
            embedding_vector, dtype='float32'
        )
    return embedding_matrix


def get_train_samples(num_samples, ratio_pn=0.33, ratio_np=0.33):
    """
    设定训练集中各个样本的情况并取得样本集

    :param num_samples:
    :param ratio_pn:
    :param ratio_np:
    :return:
    """
    if ratio_pn < 0 or ratio_np < 0 or ratio_pn + ratio_np > 1:
        print('illegal ratios!')
        return []
    train_samples = []
    # 第1位为正确答案，第2位为错误答案的情况的样本数
    num_pn_samples = int(num_samples * ratio_pn)
    # 第1位为错误答案，第2位为正确答案的情况的样本数
    num_np_samples = int(num_samples * ratio_np)
    # 第1位为错误答案，第2位为错误答案的情况的样本数
    num_0_samples = num_samples - num_pn_samples - num_np_samples
    for i in range(num_pn_samples):
        idx_q = random.randint(0, len(qa_list) - 1)
        # 若假答案池为空 或 真答案池为空就再决定一次问题
        while (len(qa_list[idx_q][1]) == 0 or len(qa_list[idx_q][2]) == 0):
            idx_q = random.randint(0, len(qa_list) - 1)

        # 第1位为正确答案，第2位为错误答案的情况时,tag为1
        idx_pa = random.randint(0, len(qa_list[idx_q][2]) - 1)
        idx_na = random.randint(0, len(qa_list[idx_q][1]) - 1)
        train_samples.append(
            (
                1, qa_list[idx_q][0],
                qa_list[idx_q][2][idx_pa], qa_list[idx_q][1][idx_na]
            )
        )

    for i in range(num_np_samples):
        idx_q = random.randint(0, len(qa_list) - 1)
        while (len(qa_list[idx_q][1]) == 0 or len(qa_list[idx_q][2]) == 0):
            idx_q = random.randint(0, len(qa_list) - 1)

        # 第1位为错误答案，第2位为正确答案的情况时,tag为-1
        idx_na = random.randint(0, len(qa_list[idx_q][1]) - 1)
        idx_pa = random.randint(0, len(qa_list[idx_q][2]) - 1)
        train_samples.append(
            (
                -1, qa_list[idx_q][0],
                qa_list[idx_q][1][idx_na], qa_list[idx_q][2][idx_pa]
            )
        )

    for i in range(num_0_samples):
        idx_q = random.randint(0, len(qa_list) - 1)
        # 若假答案池不足两个就再决定一次问题
        while (len(qa_list[idx_q][1]) < 2):
            idx_q = random.randint(0, len(qa_list) - 1)
        idx_na1 = random.randint(0, len(qa_list[idx_q][1]) - 1)
        idx_na2 = random.randint(0, len(qa_list[idx_q][1]) - 1)
        # 若两个假答案相同就再决定其中一个
        while (idx_na1 == idx_na2):
            idx_na2 = random.randint(0, len(qa_list[idx_q][1]) - 1)

        # 第1位为错误答案，第2位为错误答案的情况时,tag为0
        train_samples.append(
            (
                0, qa_list[idx_q][0],
                qa_list[idx_q][1][idx_na1], qa_list[idx_q][1][idx_na2]
            )
        )

    random.shuffle(train_samples)

    train_tag = [train_samples[i][0] for i in range(num_samples)]
    train_q = [train_samples[i][1] for i in range(num_samples)]
    train_a1 = [train_samples[i][2] for i in range(num_samples)]
    train_a2 = [train_samples[i][3] for i in range(num_samples)]

    return train_q, train_a1, train_a2, train_tag


def get_dev_data(filename):
    """
    取得验证集数据
    :param filename:
    :return:
    """
    # infile:读入data
    infile = open(filename, 'r', encoding='utf-8')
    dev_q, dev_a, dev_tag = [], [], []
    last_q, last_a, last_tag = '', [], []
    # 逐行读入
    for line in infile:
        # 以空格进行切分
        items = line.split('\t')
        # 若输入的问题不等于最后的问题
        if items[1] != last_q:
            # 对最后的问题进行分词
            q = sentence_to_words(last_q)
            # 对最后的问题带入索引
            words_to_indexes(q)
            # 遍历所有答案
            for a in last_a:
                # 新增问题
                dev_q.append(q)
                # 新增答案
                dev_a.append(sentence_to_words(a))
                words_to_indexes(dev_a[len(dev_a) - 1])
            # 若该答案为正确答案，tag中标注1
            if last_tag:
                dev_tag.append(last_tag)
            last_q, last_a, last_tag = items[1], [], []
        last_a.append(items[2])
        last_tag.append(int(items[0] != '0'))
    # 若为正确答案，last_tag为1
    if last_tag:
        # 对问题进行分词与取tokens
        q = sentence_to_words(last_q)
        words_to_indexes(q)
        #
        for a in last_a:
            dev_q.append(q)
            dev_a.append(sentence_to_words(a))
            words_to_indexes(dev_a[len(dev_a) - 1])
        dev_tag.append(last_tag)

    return dev_q, dev_a, dev_tag


def load_word_dict(filename):
    """
    加载词字典
    :param filename:
    :return:
    """
    infile = open(filename, 'r', encoding='utf-8')
    line = infile.read()
    infile.close()
    words = line.split('\t')
    word_num = 0
    word_dict.clear()
    for word in words:
        word_num += 1
        word_dict.setdefault(word, word_num)


def save_word_dict(filename):
    """
    保存词字典
    :param filename:
    :return:
    """
    sorted_tokens = sorted(
        list(word_dict.items()),
        key=lambda it: it[1], reverse=False
    )
    outfile = open(filename, 'w', encoding='utf-8')
    for token, index in sorted_tokens:
        outfile.write(token + '\t')
    outfile.close()


def get_test_data(filename):
    """
    取得测试数据
    :param filename:
    :return:
    """
    infile = open(filename, 'r', encoding='utf-8')
    test_q, test_a = [], []
    for line in infile:
        items = line.split('\t')
        test_q.append(sentence_to_words(items[0]))
        test_a.append(sentence_to_words(items[1]))
        words_to_indexes(test_q[len(test_q) - 1])
        words_to_indexes(test_a[len(test_a) - 1])
    return test_q, test_a


def trans_tag(tag):
    """
    用于训练时计算输入验证集的标签并转换成tag
    :param tag:
    :return:
    """
    t1 = numpy.zeros(len(tag))
    t2 = numpy.zeros(len(tag))
    for k in range(len(tag)):
        if tag[k] == 0:
            t1[k] = 0
            t2[k] = 0
        elif tag[k] == 1:
            t1[k] = 1
            t2[k] = 0
        else:
            t1[k] = 0
            t2[k] = 1
    return t1, t2


def evaluate_mrr(tag_lists, preds):
    """
    计算损失函数MRR
    :param tag_lists:
    :param preds:
    :return:
    """
    # idx:目前tag_lists中的指针
    idx = 0
    # cnt:样本计数器，取平均用
    cnt = 0
    # score:分数
    score = 0
    # 输入preds为[sub, cos1, cos2]
    # 此时preds为cos1，取与正确答案最相近的即可，与错误答案最不相同在问答匹配中没有意义
    preds = [t[0] for t in preds[1]]
    for tags in tag_lists:
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
    return score / cnt



if __name__ == '__main__':
    # print(sentence_to_words("台北市立建国高级中学附设高级进修补习学校什么时候停止开夜间部的？"))
    # print(sentence_to_words("1949年八月设补校。[3]"))
    print(get_dev_data('../data/BoP2017-DBQA.dev.txt'))