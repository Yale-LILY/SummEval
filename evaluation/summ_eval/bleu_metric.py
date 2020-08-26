# pylint: disable=C0103
from multiprocessing import Pool
import gin
import sacrebleu
from summ_eval.metric import Metric

@gin.configurable
class BleuMetric(Metric):
    def __init__(self, sent_smooth_method='exp', sent_smooth_value=None, sent_use_effective_order=True, \
       smooth_method='exp', smooth_value=None, force=False, lowercase=False, \
       use_effective_order=False, n_workers=24):
        """
        BLEU metric
        Wrapper around sacrebleu: https://github.com/mjpost/sacrebleu

        Args:
                :param smooth_value: For 'floor' smoothing, the floor value to use.
                :param use_effective_order: Account for references that are shorter than the largest n-gram.
                :param force: Ignore data that looks already tokenized
                :param lowercase: Lowercase the data
                :param n_workers: number of processes to use if using multiprocessing
                sent* parameters are the same but specify what is used for evaluate_example

        """
        self.sent_smooth_method = sent_smooth_method
        self.sent_smooth_value = sent_smooth_value
        self.sent_use_effective_order = sent_use_effective_order
        self.smooth_method = smooth_method
        self.smooth_value = smooth_value
        self.force = force
        self.lowercase = lowercase
        self.use_effective_order = use_effective_order
        self.n_workers = n_workers

    def evaluate_example(self, summary, reference):
        #print("BLEU is intended as a corpus-level metric. Be careful!")
        if isinstance(reference, str):
            reference = [reference]
        score = sacrebleu.sentence_bleu(summary, reference, smooth_method=self.sent_smooth_method, \
             smooth_value=self.sent_smooth_value, use_effective_order=self.sent_use_effective_order)
        score_dict = {"bleu" : score.score}
        return score_dict

    def evaluate_batch(self, summaries, references, aggregate=True):
        if aggregate:
            if isinstance(references[0], str):
                references = [references]
            score = sacrebleu.corpus_bleu(summaries, references, smooth_method=self.smooth_method, \
               smooth_value=self.smooth_value, force=self.force, lowercase=self.lowercase, \
               use_effective_order=self.use_effective_order)
            score_dict = {"bleu": score.score}
        else:
            p = Pool(processes=self.n_workers)
            score_dict = p.starmap(self.evaluate_example, zip(summaries, references))
            p.close()
        return score_dict

    @property
    def supports_multi_ref(self):
        return True
