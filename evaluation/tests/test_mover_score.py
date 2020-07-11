# pylint: disable=C0103
import unittest
from summ_eval.mover_score_metric import MoverScoreMetric
from summ_eval.test_util import EPS, CANDS, REFS


class TestScore(unittest.TestCase):
    def test_scoreV1(self):
        metric = MoverScoreMetric(version=1, stop_wordsf=None, n_gram=1, remove_subwords=True)
        scores = metric.evaluate_batch(CANDS, REFS, aggregate=False)
        score0, score1, score2 = [0.8496908025040617, 0.6671326619849703, 0.5412637140628955]

        self.assertTrue((scores[0]["mover_score"] - score0) < EPS)
        self.assertTrue((scores[1]["mover_score"] - score1) < EPS)
        self.assertTrue((scores[2]["mover_score"] - score2) < EPS)

    def test_score_singleV1(self):
        metric = MoverScoreMetric(version=1, stop_wordsf=None, n_gram=1, remove_subwords=True)
        score = metric.evaluate_example(CANDS[0], REFS[0])
        ref_score = 0.9718250021258151

        self.assertTrue((score["mover_score"] - ref_score) < EPS)

    def test_scoreV2(self):
        metric = MoverScoreMetric(version=2, stop_wordsf=None, n_gram=1, remove_subwords=True)
        scores = metric.evaluate_batch(CANDS, REFS, aggregate=False)
        score0, score1, score2 = [0.7886297263998095, 0.6391288746985502, 0.4083365780562418]

        self.assertTrue((scores[0]["mover_score"] - score0) < EPS)
        self.assertTrue((scores[1]["mover_score"] - score1) < EPS)
        self.assertTrue((scores[2]["mover_score"] - score2) < EPS)

    def test_score_singleV2(self):
        metric = MoverScoreMetric(version=2, stop_wordsf=None, n_gram=1, remove_subwords=True)
        score = metric.evaluate_example(CANDS[0], REFS[0])
        ref_score = 0.8865908405747429

        self.assertTrue((score["mover_score"] - ref_score) < EPS)

if __name__ == '__main__':
    unittest.main()
