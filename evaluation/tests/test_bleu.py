# pylint: disable=C0103
import unittest
from summ_eval.bleu_metric import BleuMetric
from summ_eval.test_util import EPS, CANDS, REFS


class Test2Score(unittest.TestCase):
    def test_score(self):
        metric = BleuMetric()
        score = metric.evaluate_batch(CANDS, REFS)
        self.assertTrue(score['bleu'] - 36.361227790893224 < EPS)

    def test_single(self):
        metric = BleuMetric()
        score = metric.evaluate_example(CANDS[0].lower(), REFS[0].lower())
        self.assertTrue(score['bleu'] - 65.80370064762461 < EPS)

if __name__ == '__main__':
    unittest.main()
