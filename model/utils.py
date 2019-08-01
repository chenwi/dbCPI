import gensim
import numpy as np
import nltk
import os
import pickle
import re
import math
from imblearn.over_sampling import SMOTE
from model.config import Config


class DataLoader(object):
    data_dir = Config.data_dir

    train_set_path = os.path.join(data_dir, r'zipdata/train_set.pkl')
    test_set_path = os.path.join(data_dir, r'zipdata/test_set.pkl')
    em1_path = os.path.join(data_dir, r'zipdata/embeddings1.npy')
    em2_path = os.path.join(data_dir, r'zipdata/embeddings2.npy')

    stop_words = nltk.corpus.stopwords.words('english')  # cause a drop in semeval 2010 corpus

    def __init__(self, max_length=160, position_max=35, is_train=True, test_path=None):
        self.max_length = max_length
        self.position_max = position_max
        self.artificial_class_index = -1
        self.numclass = 9

        ftrain = open(self.train_set_path, 'rb')
        ftest = open(self.test_set_path, 'rb')
        self.trainset = pickle.load(ftrain)
        self.testset = pickle.load(ftest)
        self.word_embeddings1 = np.load(self.em1_path)
        self.word_embeddings2 = np.load(self.em2_path)

        self.trainset_size = len(self.trainset[-1])
        self.testset_size = len(self.testset[-1])
        print(f'trainset_size: {self.trainset_size}')
        print(f'testset_size: {self.testset_size}')

        self.embeddings_dim = self.word_embeddings1.shape[1]
        print(self.word_embeddings1.shape)
        print(self.word_embeddings2.shape)

    def train_prepare(self):
        """
            train_words_list:
                [['a','good','boy'...],['a','nice','girl', ...] ,...]
            train_p1_list:
                 [[1, 2, 0...],[1, 2, 0, ...] ,...]
            train_p2_list:
                [[0, 2, 3...],[0, 2, 4, ...],...]
            train_y_list:
                [1,2, ...]
        """
        train_words_list, train_p1_list, train_p2_list, train_y_list = self.load_data('train')
        test_words_list, test_p1_list, test_p2_list, test_y_list = self.load_data('test')

        vocab = self.update_vocab(train_words_list + test_words_list)
        word_embeddings1, word2id_dic1 = self.load_embedding('emb', vocab, use_norm=False)
        word_embeddings2, word2id_dic2 = self.load_embedding('emb', vocab, use_norm=False)

        train_ids_list1, train_p1_list1, train_p2_list1 = self.mapping_padding(train_words_list,
                                                                               train_p1_list,
                                                                               train_p2_list,
                                                                               word2id_dic1)
        train_ids_list2, *_ = self.mapping_padding(train_words_list,  # train_p1_list2, train_p2_list2
                                                   train_p1_list,
                                                   train_p2_list,
                                                   word2id_dic2)

        test_ids_list1, test_p1_list1, test_p2_list1 = self.mapping_padding(test_words_list,
                                                                            test_p1_list,
                                                                            test_p2_list,
                                                                            word2id_dic1)
        test_ids_list2, *_ = self.mapping_padding(test_words_list,
                                                  test_p1_list,
                                                  test_p2_list,
                                                  word2id_dic2)

        trainset = (train_ids_list1, train_ids_list2, train_p1_list1, train_p2_list1, train_y_list)
        testset = (test_ids_list1, test_ids_list2, test_p1_list1, test_p2_list1, test_y_list)

        return trainset, testset, word_embeddings1, word_embeddings2

    def batch_iter(self, is_train, batch_size=20, num_epochs=1, oversample=False, shuffle=False):
        dataset = []
        if is_train:
            data = self.trainset
        else:
            data = self.testset
        label = data[-1]
        data = list(zip(*data))
        data = np.asarray(data)
        label = np.asarray(label)

        num_classes = self.numclass
        if oversample:
            sample_indice = np.expand_dims(np.arange(len(data)), axis=-1)
            # ros = RandomOverSampler(random_state=0)
            ros = SMOTE(random_state=0)
            random_indice, label = ros.fit_sample(sample_indice, label)
            random_indice = random_indice.astype(int)
            data = data[np.reshape(random_indice, [-1])]
        data_size = len(data)
        print('data_size: ', data_size)

        num_batches_per_epoch = math.ceil(data_size / batch_size)
        for epoch in range(num_epochs):
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
                shuffled_label = label[shuffle_indices]
            else:
                shuffled_data = data
                shuffled_label = label

            for batch in range(num_batches_per_epoch):
                start_index = batch * batch_size
                end_index = min((batch + 1) * batch_size, data_size)

                y_plus = [(i, y) if y != self.artificial_class_index else (i, 0) for i, y in
                          enumerate(shuffled_label[start_index: end_index])]
                c_minus = []
                for sample_index, y in y_plus:
                    c_minus_sample = [(sample_index, label_index) if label_index < y
                                      else (sample_index, label_index + 1)
                                      for label_index in range(num_classes - 1)]
                    c_minus.append(c_minus_sample)
                x1_batch, x2_batch, p1_batch, p2_batch, y_batch = zip(*shuffled_data[start_index: end_index])
                # yield shuffled_data[start_index: end_index], y_plus, c_minus, epoch
                dataset.append((x1_batch, x2_batch, p1_batch, p2_batch, y_batch, y_plus, c_minus))
                # yield num_batches_per_epoch,(x1_batch, x2_batch, p1_batch, p2_batch, y_batch, y_plus, c_minus)
        return dataset, data_size, num_batches_per_epoch

    def test_batch_iter(self, data, batch_size=20):
        dataset = []
        data = list(zip(*data))
        data = np.asarray(data)
        data_size = len(data)

        num_batches_per_epoch = math.ceil(data_size / batch_size)

        for batch in range(num_batches_per_epoch):
            start_index = batch * batch_size
            end_index = min((batch + 1) * batch_size, data_size)
            x1_batch, x2_batch, p1_batch, p2_batch, y_batch = zip(*data[start_index: end_index])
            dataset.append((x1_batch, x2_batch, p1_batch, p2_batch, y_batch))
            # yield (x1_batch, x2_batch, p1_batch, p2_batch, y_batch)
        return dataset, data_size

    def load_data(self, path, is_train_data=True):
        with open(path, 'r', encoding='utf8') as infile:
            data = infile.read()
        data_list = data.strip().split('\n\n')
        data_frame = [instance.split('\n') for instance in data_list]

        if is_train_data:
            words_list, p1_list, p2_list, y_list = [], [], [], []
            for instance in data_frame:
                if len(instance) < 3:
                    print(instance)
                    continue
                x, p1, p2, y = self.get_words_pos_y(instance, stopwords_removal=False, lowercase=False, lemmatize=False)
                words_list.append(x)  # train or test words [['a','boy'],['the','girl']...]
                p1_list.append(p1)  # train or test
                p2_list.append(p2)  # train or test
                y_list.append(y)  # train or test

            return words_list, p1_list, p2_list, y_list
        else:
            words_list, p1_list, p2_list = [], [], []
            for instance in data_frame:
                if len(instance) < 2:
                    print(instance)
                    continue
                x, p1, p2 = self.get_words_pos_y(instance, is_train_data=False)
                words_list.append(x)  # train or test words [['a','boy'],['the','girl']...]
                p1_list.append(p1)  # train or test
                p2_list.append(p2)  # train or test

            return words_list, p1_list, p2_list

    def remove_kh(self, s):
        s = " ".join(s)
        s = re.sub(r'[(](.*?)[)]|[[](.*?)[]]|[,](.*?)[,]', '', s)
        s = nltk.word_tokenize(s)
        return s

    def cut_long_words(self, words_before, e1, words_middle, e2, words_after):
        words = words_before + e1 + words_middle + e2 + words_after
        mid = e1 + words_middle + e2
        words_len, mid_len = len(words), len(mid)

        if (words_len > self.max_length) and (mid_len <= self.max_length):  # max_length=100
            res_len = self.max_length - mid_len
            if len(words_before) < len(words_after):  # qian
                words_before = words_before[-res_len:]
                words_after = words_after[:res_len - len(words_before)]
            else:  # hou
                words_before = words_before[-(res_len - len(words_after)):]
                words_after = words_after[:res_len]
            words = words_before + e1 + words_middle + e2 + words_after

        elif mid_len > self.max_length:
            words = e1 + words_middle[:self.max_length - len(e1) - len(e2)] + e2
        else:
            words = words_before + e1 + words_middle + e2 + words_after
        return words

    def get_words_pos_y(self, instance, is_train_data=True, stopwords_removal=False, lowercase=False, lemmatize=False):
        if is_train_data:
            pmid, sentence, relation = instance[0], instance[1], instance[2]
            y = relation
        else:
            pmid, sentence = instance[0], instance[1]
            y = None
        sentence = sentence.strip('"').strip('.')
        tokens = nltk.word_tokenize(sentence)

        e1_start, e1_end, e2_start, e2_end = None, None, None, None
        for i in range(len(tokens) - 2):
            if tokens[i] == '<':
                if tokens[i + 1] == 'e1':
                    e1_start = i
                elif tokens[i + 1] == '/e1':
                    e1_end = i
                elif tokens[i + 1] == 'e2':
                    e2_start = i
                elif tokens[i + 1] == '/e2':
                    e2_end = i
        if any(x is None for x in [e1_start, e1_end, e2_start, e2_end]):
            print(sentence)

        words_before = tokens[:e1_start]
        e1 = tokens[e1_start + 3: e1_end]
        words_middle = tokens[e1_end + 3: e2_start]
        e2 = tokens[e2_start + 3: e2_end]
        words_after = tokens[e2_end + 3:]

        if stopwords_removal:
            words_before = [word for word in words_before if word.lower() not in self.stop_words]
            words_middle = [word for word in words_middle if word.lower() not in self.stop_words]
            words_after = [word for word in words_after if word.lower() not in self.stop_words]

        if len(e1) > 20:
            e1 = ['Chemical']
        if len(e2) > 20:
            e2 = ['Protein']
        words = words_before + e1 + words_middle + e2 + words_after
        mid = e1 + words_middle + e2
        words_len, mid_len = len(words), len(mid)

        if (words_len > self.max_length) and (mid_len <= self.max_length):  # max_length=100
            res_len = (self.max_length - mid_len) // 2
            words_before = words_before[-res_len:] if res_len else []
            words_after = words_after[:res_len]
            words = words_before + e1 + words_middle + e2 + words_after

        elif mid_len > self.max_length:
            words_before = []
            words_after = []
            words_middle = words_middle[:self.max_length - len(e1) - len(e2)]
            words = e1 + words_middle + e2

        if lowercase:
            words = [word.lower() for word in words]

        p1 = list(range(-len(words_before), 0)) + [0] * len(e1) \
             + list(range(1, len(words_middle) + len(e2) + len(words_after) + 1))
        p2 = list(range(-(len(words_before) + len(e1) + len(words_middle)), 0)) \
             + [0] * len(e2) + list(range(1, len(words_after) + 1))

        if len(p1) != len(words) or len(p1) != len(p2):
            print(len(p1), len(p2), len(words))
        if is_train_data:
            return words, p1, p2, y
        return words, p1, p2

    def update_vocab(self, words_lists):  # words_lists: train and test list
        vocab = set()
        for words_list in words_lists:
            vocab.update(words_list)
        return vocab

    def load_embedding(self, path, vocab, use_norm=False):
        model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
        index, embeddings, word2id = 1, [], {}
        embeddings.append(np.zeros(model.vector_size))  # 0 pos init [0,0,...0]
        for word in vocab:
            # word = word.strip('-')  ####
            if word in model.vocab:
                wordvec = model[word]
                if use_norm:
                    wordvec /= np.sqrt((wordvec ** 2).sum(-1))
                embeddings.append(wordvec)
                word2id[word] = index
                index += 1
            else:
                # embeddings.append(np.random.rand(model0.vector_size))
                word2id[word] = 0
        del model
        return np.asarray(embeddings), word2id

    def mapping_padding(self, words_list, p1_list, p2_list, word2id_dict):
        ids_list, p1list, p2list = [], p1_list[:], p2_list[:]  # copy list
        for words in words_list:
            ids = [word2id_dict.get(word, 0) for word in words]
            ids_list.append(ids)

        for i, (ids, p1, p2) in enumerate(zip(ids_list, p1list, p2list)):
            assert len(ids) == len(p1)
            assert len(p1) == len(p2)
            ids.extend([0 for i in range(self.max_length - len(ids))])  # padding 0

            p1 = [-self.position_max if pos < -self.position_max else pos for pos in p1]
            p1 = [self.position_max if pos > self.position_max else pos for pos in p1]
            p1 = [i + self.position_max + 1 for i in p1]
            p1.extend([0 for i in range(self.max_length - len(p1))])
            p1list[i] = p1

            p2 = [-self.position_max if pos < -self.position_max else pos for pos in p2]
            p2 = [self.position_max if pos > self.position_max else pos for pos in p2]
            p2 = [i + self.position_max + 1 for i in p2]
            p2.extend([0 for i in range(self.max_length - len(p2))])
            p2list[i] = p2

        return ids_list, p1list, p2list

    def eval(self, y_pred):
        y_true = self.testset[-1]
        y_true = [i if i >= 0 else 9 for i in y_true]
        y_pred = [i if i >= 0 else 9 for i in y_pred]

        return self._calc_eval(y_true, y_pred)

    def _calc_eval(self, y_true, y_pred):
        assert len(y_true) == len(y_pred)
        matrix = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
        results2normal = list(zip(*(y_true, y_pred)))

        total = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for item in results2normal:
            rel, pre = item
            total[rel] += 1
            matrix[rel][pre] += 1

        prec, rec, f1 = [], [], []
        tp, fp, fn = [], [], []
        for i in range(len(matrix)):
            try:
                p = matrix[i][i] / sum([matrix[j][i] for j in range(len(matrix))])
                r = matrix[i][i] / sum([matrix[i][j] for j in range(len(matrix[0]))])
                tp.append(matrix[i][i])
                fp.append(sum([matrix[j][i] for j in range(len(matrix))]) - matrix[i][i])
                fn.append(sum([matrix[i][j] for j in range(len(matrix[0]))]) - matrix[i][i])
            except:
                p = 0
                r = 0
            prec.append(round(p, 4))
            rec.append(round(r, 4))

        for i in range(len(rec)):
            try:
                f = 2 * prec[i] * rec[i] / (prec[i] + rec[i])
            except:
                f = 0
            f1.append(round(f, 4))

        # all
        tp = [matrix[i][i] for i in range(len(matrix) - 1)]
        matrix_gs = []
        for i in range(len(matrix) - 1):
            m = matrix[i][0:len(matrix) - 1]
            matrix_gs.append(m)
        tp, ap, n = sum(tp), 0, sum(total)
        for i in range(len(matrix_gs)):  # col
            ap += sum(matrix_gs[i])

        matrix1 = np.asarray(matrix)
        print(matrix1)

        print()
        print('total:', total)
        print(f'prec:   {prec}')
        print(f'recall: {rec}')
        print(f'f1:     {f1}')
        print()

        # gs
        total_gs = total[2:6] + total[8:9]
        tp = [matrix[2][2], matrix[3][3], matrix[4][4], matrix[5][5], matrix[8][8]]
        print('tp: ', tp)
        matrix_gs = []
        for i in range(len(matrix)):
            m = matrix[i][2:6] + matrix[i][8:9]
            matrix_gs.append(m)
        tp, ap, n = sum(tp), 0, sum(total_gs)
        for i in range(len(matrix_gs)):  # col
            ap += sum(matrix_gs[i])
        try:
            PREC, REC = tp / ap, tp / n
            F1 = 2 * PREC * REC / (PREC + REC)
        except:
            PREC, REC, F1 = 0, 0, 0

        print('total_gs:', total_gs)
        print(f'PREC,REC,F {round(PREC,4), round(REC,4), round(F1,4)}')
        print()
        return F1


if __name__ == '__main__':
    d = DataLoader(is_train=True, test_path='')
    # d.load_data(d.train_in_path)
    # d.load_data(d.train_in_path)
    train_data_iter, train_size, train_num_batches = d.batch_iter(
        is_train=True, batch_size=2,
        num_epochs=1, oversample=True, shuffle=True)
    for step, data in enumerate(train_data_iter, 1):
        x_batch, x1_batch, p1_batch, p2_batch, y_batch, y_plus, c_minus = data
        print(x_batch)
        print(y_batch)
        print(y_plus)
        print(c_minus)
        break
