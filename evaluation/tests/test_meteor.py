# pylint: disable=C0103
import unittest
from summ_eval.meteor_metric import MeteorMetric
from summ_eval.test_util import EPS, CANDS, REFS


class TestScore(unittest.TestCase):
    def test_score(self):
        metric = MeteorMetric()
        score = metric.evaluate_batch(CANDS, REFS)
        self.assertTrue((score['meteor'] - 0.4328158109487261) < EPS)

if __name__ == '__main__':
    unittest.main()
