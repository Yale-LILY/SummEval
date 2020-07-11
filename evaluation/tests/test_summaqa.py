# pylint: disable=C0103,C0301
import unittest
from summ_eval.summa_qa_metric import SummaQAMetric
from summ_eval.test_util import EPS, ARTICLE_sqa, SUMM1, SUMM2

EPS = 1e-5


class TestScore(unittest.TestCase):
    def test_single(self):
        metric = SummaQAMetric(batch_size=8)
        score = metric.evaluate_example(SUMM1, ARTICLE_sqa)
        ans_dict = {'summaqa_avg_prob': 0.09602569331764244, 'summaqa_avg_fscore': 0.21143162393162393}
        self.assertTrue((score['summaqa_avg_prob'] - ans_dict['summaqa_avg_prob']) < EPS)
        self.assertTrue((score['summaqa_avg_fscore'] - ans_dict['summaqa_avg_fscore']) < EPS)

    def test_batch(self):
        metric = SummaQAMetric(batch_size=8)
        srcs = [ARTICLE_sqa, ARTICLE_sqa]
        gens = [SUMM1, SUMM2]
        score = metric.evaluate_batch(gens, srcs)
        ans_dict = {'summaqa_avg_prob': 0.058506956241520434, 'summaqa_avg_fscore': 0.10571581196581196}
        self.assertTrue((score['summaqa_avg_prob'] - ans_dict['summaqa_avg_prob']) < EPS)
        self.assertTrue((score['summaqa_avg_fscore'] - ans_dict['summaqa_avg_fscore']) < EPS)

if __name__ == '__main__':
    unittest.main()
