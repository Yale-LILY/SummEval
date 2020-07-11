# pylint: disable=C0103,R0201
import unittest
from summ_eval.syntactic_metric import SyntacticMetric
from summ_eval.test_util import CANDS, REFS


class TestScore(unittest.TestCase):
    def test_score(self):
        metric = SyntacticMetric()
        stats = metric.evaluate_example(CANDS[0], REFS[0])
        ref_stats = {'Words': 8, 'Sentences': 1, 'Verb Phrases': 1, \
            'Clauses': 1, 'T-units': 1, 'Dependent Clauses': 0, \
            'Complex T-units': 0, 'Coordinate Phrases': 0, 'Complex Nominals': 1, \
            'words/sentence': 8.0, 'words/t-unit': 8.0, 'words/clause': 8.0, \
            'clauses/sent': 1.0, 'verb phrases/t-unit': 1.0, 'clauses/t-unit': 1.0, \
            'dependent clauses/clause': 0, 'dependent clauses/t-unit': 0, \
            't-units/sentence': 1.0, 'complex t-units/t-units': 0,  \
            'coordinate phrases/t-unit': 0, 'coordinate-phrases/clauses': 0, \
            'complex nominals/t-unit': 1.0, 'complex nominals/clause': 1.0}
        for key, val in stats.items():
            self.assertTrue(val == ref_stats[key])

if __name__ == '__main__':
    unittest.main()
