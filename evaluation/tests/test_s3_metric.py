# pylint: disable=C0103
import os
import unittest
from summ_eval.s3_metric import S3Metric
from summ_eval.test_util import CAND_R, REF_R, CANDS, REFS, EPS


class TestScore(unittest.TestCase):
    def test_score(self):
        metric = S3Metric()
        score_dict = metric.evaluate_example(CAND_R, REF_R)
        self.assertTrue((score_dict["s3_pyr"] - 0.4402288438243088) < EPS)
        self.assertTrue((score_dict["s3_resp"] - 0.5103094504071222) < EPS)

    def test_score_batch(self):
        metric = S3Metric()
        score_dict = metric.evaluate_batch(CANDS, REFS)
        for predicted, (pyr, resp) in zip(score_dict, [
            (1.358148717958252, 1.5925579213409842),
            (1.1742432908208689, 1.4061338986807543),
            (0.6816419565604588, 0.7101254431464145),
        ]):
            self.assertTrue((predicted["s3_pyr"] - pyr) < EPS)
            self.assertTrue((predicted["s3_resp"] - resp) < EPS)

if __name__ == '__main__':
    unittest.main()
