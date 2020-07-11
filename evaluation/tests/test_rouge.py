# pylint: disable=C0103
import os
import unittest
from summ_eval.rouge_metric import RougeMetric
from summ_eval.test_util import CAND_R, REF_R, rouge_output, rouge_output_batch, CANDS, REFS, EPS

ROUGE_HOME = os.environ['ROUGE_HOME']

class TestScore(unittest.TestCase):
    def test_score(self):
        metric = RougeMetric(rouge_dir=ROUGE_HOME)
        score_dict = metric.evaluate_example(CAND_R, REF_R)["rouge"]
        for key, val in score_dict.items():
            self.assertTrue((val - rouge_output[key]) < EPS)

    def test_score_batch(self):
        metric = RougeMetric(rouge_dir=ROUGE_HOME)
        score_dict = metric.evaluate_batch(CANDS, REFS)["rouge"]
        for key, val in score_dict.items():
            self.assertTrue((val - rouge_output_batch[key]) < EPS)

if __name__ == '__main__':
    unittest.main()
