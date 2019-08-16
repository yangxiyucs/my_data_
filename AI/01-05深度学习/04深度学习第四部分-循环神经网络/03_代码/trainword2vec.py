import os.path
import sys
import multiprocessing

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

if __name__ == '__main__':

    if len(sys.argv) < 3:
      sys.exit(1)

    # inp表示语料库(分词)，outp：模型
    inp, outp = sys.argv[1:3]

    model = Word2Vec(LineSentence(inp), size=400, window=5, min_count=5, workers=multiprocessing.cpu_count())

    model.save(outp)