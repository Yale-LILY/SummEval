# pylint: disable=C0103
import os
from multiprocessing import Pool
from collections import Counter
import gin
from summ_eval.s3_utils import rouge_n_we, load_embeddings
from summ_eval.metric import Metric

dirname = os.path.dirname(__file__)

@gin.configurable
class RougeWeMetric(Metric):
    def __init__(self, emb_path=os.path.join(dirname, './embeddings/deps.words'), n_gram=3, \
                 n_workers=24, tokenize=True):
        """
        ROUGE-WE metric
        Taken from https://github.com/UKPLab/emnlp-ws-2017-s3/tree/b524407ada525c81ceacd2590076e20103213e3b

        Args:
                :param emb_path: path to dependency-based word embeddings found here:
                        https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/
                :param n_gram: n_gram length to be used for calculation; if n_gram=3,
                        only calculates ROUGE-WE for n=3; reset n_gram to calculate
                        for other n-gram lengths
                :param n_workers: number of processes to use if using multiprocessing
                :param tokenize: whether to apply stemming and basic tokenization to input;
                        otherwise assumes that user has done any necessary tokenization

        """
        self.word_embeddings = load_embeddings(emb_path)
        self.n_gram = n_gram
        self.n_workers = n_workers
        self.tokenize = tokenize

    def evaluate_example(self, summary, reference):
        if not isinstance(reference, list):
            reference = [reference]
        if not isinstance(summary, list):
            summary = [summary]
        score = rouge_n_we(summary, reference, self.word_embeddings, self.n_gram, \
                 return_all=True, tokenize=self.tokenize)
        score_dict = {f"rouge_we_{self.n_gram}_p": score[0], f"rouge_we_{self.n_gram}_r": score[1], \
                      f"rouge_we_{self.n_gram}_f": score[2]}
        return score_dict

    def evaluate_batch(self, summaries, references, aggregate=True):
        p = Pool(processes=self.n_workers)
        results = p.starmap(self.evaluate_example, zip(summaries, references))
        p.close()
        if aggregate:
            corpus_score_dict = Counter()
            for x in results:
                corpus_score_dict.update(x)
            for key in corpus_score_dict.keys():
                corpus_score_dict[key] /= float(len(summaries))
            return corpus_score_dict
        else:
            return results

    @property
    def supports_multi_ref(self):
        return True
