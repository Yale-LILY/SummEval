# pylint: disable=C0103
import unittest
from summ_eval.bert_score_metric import BertScoreMetric
from summ_eval.test_util import EPS, CANDS, REFS


class TestScore(unittest.TestCase):
    def test_score(self):
        metric = BertScoreMetric(lang='en', model_type='roberta-large', num_layers=17, verbose=False, idf=False,\
                 batch_size=3, rescale_with_baseline=False)
        score_dict = metric.evaluate_batch(CANDS, REFS)

        avgP = sum([0.9843302369117737, 0.9832239747047424, 0.9120386242866516])/3
        avgR = sum([0.9823839068412781, 0.9732863903045654, 0.920428991317749])/3
        avgF = sum([0.9833561182022095, 0.9782299995422363, 0.916214644908905])/3
        self.assertTrue((score_dict['bert_score_precision'] - avgP) < EPS)
        self.assertTrue((score_dict['bert_score_recall'] - avgR) < EPS)
        self.assertTrue((score_dict['bert_score_f1'] - avgF) < EPS)

    def test_idf_score(self):
        metric = BertScoreMetric(lang='en', model_type='roberta-large', num_layers=17, verbose=False, idf=True,\
                 batch_size=3, rescale_with_baseline=False)
        score_dict = metric.evaluate_batch(CANDS, REFS)

        avgP = sum([0.9837872385978699, 0.9754738807678223, 0.8947395086288452])/3
        avgR = sum([0.9827190637588501, 0.9697767496109009, 0.9172918796539307])/3
        avgF = sum([0.9832529425621033, 0.972616970539093, 0.9058753848075867])/3
        self.assertTrue((score_dict['bert_score_precision'] - avgP) < EPS)
        self.assertTrue((score_dict['bert_score_recall'] - avgR) < EPS)
        self.assertTrue((score_dict['bert_score_f1'] - avgF) < EPS)

    def test_score_rescale(self):
        metric = BertScoreMetric(lang='en', model_type='roberta-large', num_layers=17, verbose=False, idf=False,\
                 batch_size=3, rescale_with_baseline=True)
        score_dict = metric.evaluate_batch(CANDS, REFS)

        avgP = sum([0.907000780105591, 0.900435566902161, 0.477955609560013])/3
        avgR = sum([0.895456790924072, 0.841467440128326, 0.527785062789917])/3
        avgF = sum([0.901383399963379, 0.871010780334473, 0.503565192222595])/3
        self.assertTrue((score_dict['bert_score_precision'] - avgP) < EPS)
        self.assertTrue((score_dict['bert_score_recall'] - avgR) < EPS)
        self.assertTrue((score_dict['bert_score_f1'] - avgF) < EPS)

    def test_idf_score_rescale(self):
        metric = BertScoreMetric(lang='en', model_type='roberta-large', num_layers=17, verbose=False, idf=True,\
                 batch_size=3, rescale_with_baseline=True)
        score_dict = metric.evaluate_batch(CANDS, REFS)

        avgP = sum([0.903778135776520, 0.854439020156860, 0.375287383794785])/3
        avgR = sum([0.897446095943451, 0.820639789104462, 0.509167850017548])/3
        avgF = sum([0.900772094726562, 0.837753534317017, 0.442304641008377])/3
        self.assertTrue((score_dict['bert_score_precision'] - avgP) < EPS)
        self.assertTrue((score_dict['bert_score_recall'] - avgR) < EPS)
        self.assertTrue((score_dict['bert_score_f1'] - avgF) < EPS)

    def test_multi_refs(self):
        cands = ['I like lemons.']
        refs = [['I am proud of you.', 'I love lemons.', 'Go go go.']]
        metric = BertScoreMetric(lang='en', batch_size=3, rescale_with_baseline=True)

        score_dict = metric.evaluate_batch(cands, refs)
        score_dict_best = metric.evaluate_batch(cands, [refs[0][1]])

        self.assertTrue((score_dict['bert_score_precision'] - score_dict_best['bert_score_precision']) < EPS)
        self.assertTrue((score_dict['bert_score_recall'] - score_dict_best['bert_score_recall']) < EPS)
        self.assertTrue((score_dict['bert_score_f1'] - score_dict_best['bert_score_f1']) < EPS)

if __name__ == '__main__':
    unittest.main()
