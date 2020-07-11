# pylint: disable=C0103
import unittest
from summ_eval.chrfpp_metric import ChrfppMetric
from summ_eval.test_util import EPS, CANDS_chrfpp, REFS_chrfpp

class TestScore(unittest.TestCase):
    def test_score(self):
        metric = ChrfppMetric()
        score = metric.evaluate_batch(CANDS_chrfpp, REFS_chrfpp)
        ref = {'chrf': 0.38735906038936213}
        self.assertTrue((score['chrf'] - ref['chrf']) < EPS)
        single_score = metric.evaluate_example(CANDS_chrfpp[0], REFS_chrfpp[0])
        ref_single = {'chrf': 0.6906099983606945}
        self.assertTrue((single_score['chrf'] - ref_single['chrf']) < EPS)

if __name__ == '__main__':
    unittest.main()
