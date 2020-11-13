# pylint: disable=W0102,C0301,W1401,C0303,C0103,W0221,C0200,W0106
# Uses code from https://github.com/PrimerAI/blanc

from collections import Counter
import gin
from blanc import BlancHelp, BlancTune

from summ_eval.metric import Metric

@gin.configurable
class BlancMetric(Metric):
    def __init__(self, device='cuda', inference_batch_size=128, finetune_batch_size=24, use_tune=True):
        self.device = device
        self.inference_batch_size = inference_batch_size
        self.finetune_batch_size = finetune_batch_size
        self.use_tune = use_tune

    def evaluate_example(self, summary, input_text):
        if self.use_tune:
            blanc_mod = BlancTune(device=self.device)
        else:
            blanc_mod = BlancHelp(device=self.device)
        score = blanc_mod.eval_once(input_text, summary)
        return {"blanc": score}

    def evaluate_batch(self, summaries, input_texts, aggregate=True):
        corpus_score_dict = Counter()
        if self.use_tune:
            blanc_mod = BlancTune(device='cuda', inference_batch_size=self.inference_batch_size, finetune_batch_size=self.finetune_batch_size)
        else:
            blanc_mod = BlancTune(device=self.device, inference_batch_size=self.inference_batch_size)
            
        results = blanc_mod.eval_pairs(input_texts, summaries)
        results = [{"blanc": score} for score in results]
        if aggregate:
            [corpus_score_dict.update(x) for x in results]
            for key in corpus_score_dict.keys():
                corpus_score_dict[key] /= float(len(input_texts))
            return corpus_score_dict
        else:
            return results

    @property
    def supports_multi_ref(self):
        return False
