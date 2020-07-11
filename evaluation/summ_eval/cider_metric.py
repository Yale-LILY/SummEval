# pylint: disable=C0103
import gin
from nltk.tokenize import RegexpTokenizer
from summ_eval.cider_utils import CiderScorer
from summ_eval.metric import Metric

tokenizer = RegexpTokenizer(r'\w+')

@gin.configurable
class CiderMetric(Metric):
    def __init__(self, n_gram=4, sigma=6.0, tokenize=True):
        """
        CIDEr metric
        Makes use of https://github.com/Maluuba/nlg-eval/tree/master/nlgeval/pycocoevalcap/cider

        Args:
                :param n_gram: CIDEr calculation takes into account n_grams of up to this length
                :param sigma: sigma used in Gaussian length penalty, described in Section 8 of original paper
                :param tokenize: whether to apply basic tokenization to input; otherwise assumes that user has \
                        done any necessary tokenization

        """
        self.n_gram = n_gram
        self.sigma = sigma
        self.tokenize = tokenize

    def evaluate_example(self, summary, reference):
        if self.tokenize:
            if isinstance(reference, str):
                reference = " ".join(tokenizer.tokenize(reference))
            else:
                reference = [" ".join(tokenizer.tokenize(ref)) for ref in reference]
            summary = " ".join(tokenizer.tokenize(summary))
        cider_scorer = CiderScorer(n=self.n_gram, sigma=self.sigma)
        if not isinstance(reference, list):
            reference = [reference]
        cider_scorer += (summary, reference)
        (score, _) = cider_scorer.compute_score()
        score_dict = {"cider": score}
        return score_dict

    def evaluate_batch(self, summaries, references, aggregate=True):
        if self.tokenize:
            if isinstance(references[0], str):
                references = [" ".join(tokenizer.tokenize(reference)) \
                              for reference in references]
            else:
                references = [[" ".join(tokenizer.tokenize(ref)) \
                              for ref in reference] for reference in references]
            summaries = [" ".join(tokenizer.tokenize(summary)) for summary in summaries]
        cider_scorer = CiderScorer(n=self.n_gram, sigma=self.sigma)
        for summ, ref in zip(summaries, references):
            if not isinstance(ref, list):
                ref = [ref]
            cider_scorer += (summ, ref)
        (score, scores) = cider_scorer.compute_score()
        if not aggregate:
            scores_return = [{"cider": cur_score} for cur_score in scores]
            return scores_return
        score_dict = {"cider": score}
        return score_dict

    @property
    def supports_multi_ref(self):
        return True
