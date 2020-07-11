# pylint: disable=C0103
import unittest
from summ_eval.sentence_movers_metric import SentenceMoversMetric
from summ_eval.test_util import EPS, CANDS, REFS


class TestScore(unittest.TestCase):
    def test_glove_sms(self):
        metric = SentenceMoversMetric(wordrep='glove', metric='sms')
        score = metric.evaluate_batch(CANDS, REFS)
        avg = sum([0.998619953103503, 0.4511249300530058, 0.2903306055171392])/3

        self.assertTrue((score["sentence_movers_glove_sms"] - avg) < EPS)

    def test_glove_wms(self):
        metric = SentenceMoversMetric(wordrep='glove', metric='wms')
        score = metric.evaluate_batch(CANDS, REFS)
        avg = sum([1.0, 0.2417027898540903, 0.04358073486688484])/3

        self.assertTrue((score["sentence_movers_glove_wms"] - avg) < EPS)

    def test_glove_swms(self):
        metric = SentenceMoversMetric(wordrep='glove', metric='s+wms')
        score = metric.evaluate_batch(CANDS, REFS)
        avg = sum([0.9993097383211589, 0.330209238465829, 0.11248476630690257])/3

        self.assertTrue((score["sentence_movers_glove_s+wms"] - avg) < EPS)

    def test_glove_sms_single(self):
        metric = SentenceMoversMetric(wordrep='glove', metric='sms')
        score = metric.evaluate_example(CANDS[0], REFS[0])
        score0 = 0.998619953103503

        self.assertTrue((score["sentence_movers_glove_sms"] - score0) < EPS)

    def test_glove_wms_single(self):
        metric = SentenceMoversMetric(wordrep='glove', metric='wms')
        score = metric.evaluate_example(CANDS[0], REFS[0])
        score0 = 1.0

        self.assertTrue((score["sentence_movers_glove_wms"] - score0) < EPS)

    def test_glove_swms_single(self):
        metric = SentenceMoversMetric(wordrep='glove', metric='s+wms')
        score = metric.evaluate_example(CANDS[0], REFS[0])
        score0 = 0.9993097383211589

        self.assertTrue((score["sentence_movers_glove_s+wms"] - score0) < EPS)


if __name__ == '__main__':
    unittest.main()
