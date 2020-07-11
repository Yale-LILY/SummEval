# pylint: disable=C0103,W0702
from multiprocessing import Pool
from itertools import repeat
import gin
#from allennlp.commands.elmo import ElmoEmbedder

from summ_eval.metric import Metric
from summ_eval.sentence_movers_utils import tokenize_texts, get_sim



@gin.configurable
class SentenceMoversMetric(Metric):
    def __init__(self, wordrep='glove', metric='sms', n_workers=24, tokenize=True):
        """
        Sentence Mover's Similarity metric
        Makes use of code here:
                https://github.com/eaclark07/sms/tree/master/wmd-relax-master

        Modified original code to use spacy for sentence tokenization, as this is
                faster than using nltk for sentence tokenization followed by applying
                spacy over individual sentences; we believe this keeps the integrity
                of the metric. We recommend using GloVe for the wordrep as it is much
                quicker and significant differences were not reported between it and
                ELMo in the original paper.

        Args:
                :param wordrep: GloVe or ELMo for embeddings
                :param metric: sms, wms, s+wms; please see original paper for variations
                :param n_workers: number of processes to use if using multiprocessing
                :param tokenize: whether to tokenize the input text; otherwise assumes
                        that your input is a spacy-processed Doc with .sents attributes
        """
        self.wordrep = wordrep
        self.metric = metric
        self.model = ElmoEmbedder() if wordrep == 'elmo' else None
        self.n_workers = n_workers
        self.tokenize = tokenize

    def evaluate_example(self, summary, reference):
        inLines = [(reference, summary)]
        token_doc_list, text_doc_list = tokenize_texts(inLines, self.wordrep, self.tokenize)
        score = get_sim(token_doc_list[0], text_doc_list[0], self.wordrep, self.model, self.metric)
        score_dict = {f"sentence_movers_{self.wordrep}_{self.metric}": score}
        return score_dict

    def evaluate_batch(self, summaries, references, aggregate=True):
        inLines = zip(references, summaries)
        token_doc_list, text_doc_list = tokenize_texts(inLines, self.wordrep, self.tokenize)
        p = Pool(processes=self.n_workers)
        results = p.starmap(get_sim, zip(token_doc_list, text_doc_list, \
                  repeat(self.wordrep), repeat(self.model), repeat(self.metric)))
        if aggregate:
            score_dict = {f"sentence_movers_{self.wordrep}_{self.metric}": sum(results)/len(results)}
        else:
            score_dict = [{f"sentence_movers_{self.wordrep}_{self.metric}": result} for result in results]
        return score_dict

    @property
    def supports_multi_ref(self):
        return False
