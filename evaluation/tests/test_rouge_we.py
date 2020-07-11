# pylint: disable=C0103,C0301
import unittest
from summ_eval.rouge_we_metric import RougeWeMetric
from summ_eval.test_util import EPS, CANDS_l2s, REFS_l2s


class TestScore(unittest.TestCase):
    def test_score_single(self):
        metric = RougeWeMetric(n_gram=1)
        score = metric.evaluate_example(CANDS_l2s[0], REFS_l2s[0])
        self.assertTrue((score['rouge_we_1_f'] - 0.22) < EPS)
        metric = RougeWeMetric(n_gram=2)
        score = metric.evaluate_example(CANDS_l2s[0], REFS_l2s[0])
        self.assertTrue((score['rouge_we_2_f'] - 0.09090909090909093) < EPS)


if __name__ == '__main__':
    unittest.main()
