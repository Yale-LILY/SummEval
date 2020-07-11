import gin
import bert_score

from summ_eval.metric import Metric


@gin.configurable
class BertScoreMetric(Metric):
    def __init__(self, lang='en', model_type='bert-base-uncased', num_layers=8, verbose=False, idf=False,\
                 nthreads=4, batch_size=64, rescale_with_baseline=False):
        """
        BERT-Score metric

        Args (copied from https://github.com/Tiiiger/bert_score/blob/master/bert_score/score.py):
            :param model_type (str): bert specification, default using the suggested
                      model for the target langauge; has to specify at least one of
                      model_type or lang
            :param num_layers (int): the layer of representation to use.
                      default using the number of layer tuned on WMT16 correlation data
            :param verbose (bool): turn on intermediate status update
            :param idf (bool or dict): use idf weighting, can also be a precomputed idf_dict
            :param device (str): on which the contextual embedding model will be allocated on.
                      If this argument is None, the model lives on cuda:0 if cuda is available.
            param nthreads (int): number of threads
            param batch_size (int): bert score processing batch size
            param lang (str): language of the sentences; has to specify
                      at least one of model_type or lang. lang needs to be
                      specified when rescale_with_baseline is True.
            param rescale_with_baseline (bool): rescale bertscore with pre-computed baseline
        """
        self.lang = lang
        self.model_type = model_type
        self.num_layers = num_layers
        self.verbose = verbose
        self.idf = idf
        self.nthreads = nthreads
        self.batch_size = batch_size
        self.rescale_with_baseline = rescale_with_baseline

    def evaluate_example(self, summary, reference):
        assert not self.idf, "idf mode not supported for evaluating a single example"
        if isinstance(reference, str):
            reference = [reference]
        all_preds, hash_code = bert_score.score([summary], reference, model_type=self.model_type, \
                                                num_layers=self.num_layers,
                                                verbose=self.verbose, idf=self.idf, batch_size=self.batch_size,
                                                nthreads=self.nthreads, lang=self.lang, return_hash=True,
                                                rescale_with_baseline=self.rescale_with_baseline)
        print(f"hash_code: {hash_code}")
        score = [{"bert_score_precision": p.cpu().item(), "bert_score_recall": r.cpu().item(), "bert_score_f1": \
                 f.cpu().item()} for (p, r, f) in all_preds]
        return score

    def evaluate_batch(self, summaries, references, aggregate=True):
        all_preds, hash_code = bert_score.score(summaries, references, model_type=self.model_type, \
                                                num_layers=self.num_layers,
                                                verbose=self.verbose, idf=self.idf, batch_size=self.batch_size,
                                                nthreads=self.nthreads, lang=self.lang, return_hash=True,
                                                rescale_with_baseline=self.rescale_with_baseline)
        print(f"hash_code: {hash_code}")
        if aggregate:
            avg_scores = [s.mean(dim=0) for s in all_preds]
            p_val = avg_scores[0].cpu().item()
            r_val = avg_scores[1].cpu().item()
            f1_val = avg_scores[2].cpu().item()
            scores = {"bert_score_precision": p_val, "bert_score_recall": r_val, "bert_score_f1": f1_val}
            return scores
        else:
            cur_items = [{"bert_score_precision": p.cpu().item(), "bert_score_recall": r.cpu().item(), \
                         "bert_score_f1": f.cpu().item()} for (p, r, f) in list(zip(*all_preds))]
            return cur_items

    @property
    def supports_multi_ref(self):
        return True
