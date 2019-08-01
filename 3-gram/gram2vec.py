# encoding=utf-8
import multiprocessing

from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec, word2vec, KeyedVectors

sentences = word2vec.LineSentence('./token.txt')

# model = Word2Vec(sentences, size=10, window=5, min_count=1, workers=multiprocessing.cpu_count())

# path = get_tmpfile("./w2v_model.bin")
# model.save(path)
# for key in model.similar_by_word('ter', topn=10):
#     print(key)

#
# model = Word2Vec.load(path)
# model.save_word2vec_format("./w2v_vector.bin",binary=True)
#
# for key in model.similar_by_word('ter', topn=10):
#     print(key)
#
# model.wv.save_word2vec_format('./model.bin', binary=True)  # C binary format
m = KeyedVectors.load_word2vec_format('./model.bin', binary=True)
for key in m.similar_by_word('inh', topn=10):
    print(key)
# for v in m.vocab:
#     print(v,m[v])

# word_vectors = KeyedVectors.load_word2vec_format('./model.bin', binary=True)
# print(word_vectors)
