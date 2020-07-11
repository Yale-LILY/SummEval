# pylint: disable=C0103,C0301
import os
from collections import Counter
from multiprocessing import Pool
import gin
from summ_eval.metric import Metric
from summ_eval.s3_utils import S3, load_embeddings

dirname = os.path.dirname(__file__)


@gin.configurable
class S3Metric(Metric):
    def __init__(self, emb_path=os.path.join(dirname, './embeddings/deps.words'), \
        model_folder=os.path.join(dirname, './models/en/'), n_workers=24, tokenize=True):
        """
        S3 metric
        Taken from https://github.com/UKPLab/emnlp-ws-2017-s3/tree/b524407ada525c81ceacd2590076e20103213e3b

        Args:
                :param emb_path: path to dependency-based word embeddings found here:
                        https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/
                :param model_folder: path to S3 model folders found here:
                        https://github.com/UKPLab/emnlp-ws-2017-s3/tree/b524407ada525c81ceacd2590076e20103213e3b/models
                :param n_workers: number of processes to use if using multiprocessing
                :param tokenize: whether to apply stemming and basic tokenization to input; otherwise assumes that user has \
                        done any necessary tokenization

        """
        self.word_embeddings = load_embeddings(emb_path)
        self.model_folder = model_folder
        self.n_workers = n_workers
        self.tokenize = tokenize

    def evaluate_example(self, summary, reference):
        if not isinstance(reference, list):
            reference = [reference]
        if not isinstance(summary, list):
            summary = [summary]
        if len(reference) == 1 and isinstance(reference[0], str):
            reference = [reference]
        score = S3(reference, summary, self.word_embeddings, self.model_folder, self.tokenize)
        score_dict = {"s3_pyr": score[0], "s3_resp": score[1]}
        return score_dict

    def evaluate_batch(self, summaries, references, aggregate=False):
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
