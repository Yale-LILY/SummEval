# pylint: disable=C0103,C0301
import unittest
from summ_eval.data_stats_metric import DataStatsMetric
from summ_eval.test_util import CANDS, ARTICLE


class TestScore(unittest.TestCase):
    def test_single(self):
        metric = DataStatsMetric()
        stats = metric.evaluate_example(CANDS[0], ARTICLE)
        ref_stats = {'coverage': 1.0, 'density': 3.0, 'compression': 31.9, \
            'extractive_fragments': ["28-year-old chef", "found dead in", "San Francisco", "mall"], \
            'summary_length': 10, 'percentage_novel_1-gram': 0.0, 'percentage_repeated_1-gram_in_summ': 0.0, \
            'percentage_novel_2-gram': 0.2222222222222222, 'percentage_repeated_2-gram_in_summ': 0.0, \
            'percentage_novel_3-gram': 0.5, 'percentage_repeated_3-gram_in_summ': 0.0}
        for key, val in stats.items():
            if key == "extractive_fragments":
                for count, ex in enumerate(val):
                    self.assertTrue(ex.text == ref_stats[key][count])
            else:
                self.assertTrue(val == ref_stats[key])

    def test_corpus(self):
        metric = DataStatsMetric()
        stats = metric.evaluate_batch(CANDS, [ARTICLE] * len(CANDS))
        ref_stats = {'coverage': 0.8631884057971014, 'density': 3.8892753623188407, \
            'compression': 19.50985507246377, 'summary_length': 19.333333333333332, \
            'percentage_novel_1-gram': 0.13681159420289857, 'percentage_repeated_1-gram_in_summ': 0.0, \
            'percentage_novel_2-gram': 0.35942760942760943, 'percentage_repeated_2-gram_in_summ': 0.0, \
            'percentage_novel_3-gram': 0.5400276052449965, 'percentage_repeated_3-gram_in_summ': 0.0}
        for key, val in stats.items():
            self.assertTrue(val == ref_stats[key])

if __name__ == '__main__':
    unittest.main()
