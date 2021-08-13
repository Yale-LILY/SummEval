# pylint: disable=C0103,W0632,E1101
from collections import Counter
import gin
import os
import requests
import zipfile, io
from stanza.server import CoreNLPClient
from summ_eval.metric import Metric
from summ_eval.syntactic_utils import get_stats

dirname = os.path.dirname(__file__)

if not os.path.exists(os.path.join(dirname, "stanford-corenlp-full-2018-10-05")):
    url = 'http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip'
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(os.path.join(dirname, "./"))
    print(f'Please run the following command and add it to your startup script: \n export CORENLP_HOME={os.path.join(dirname, "./stanford-corenlp-full-2018-10-05/")}')

@gin.configurable
class SyntacticMetric(Metric):
    def __init__(self):
        """
        Syntactic metric

        This is the L2 Syntactic Complexity Analyzer from:
                http://www.personal.psu.edu/xxl13/downloads/l2sca.html

        NOTE: this metric only uses the summary but we keep the default
                arguments for consistency.

        """
        pass

    def evaluate_example(self, summary, reference):
        with CoreNLPClient(annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'parse'], \
                           timeout=30000, memory='16G') as client:
            answer = get_stats(client, summary)
            return answer

    def evaluate_batch(self, summaries, references, aggregate=True):
        corpus_score_dict = Counter()
        with CoreNLPClient(annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'parse'], \
                           timeout=30000, memory='16G') as client:
            if aggregate:
                corpus_score_dict = Counter()
            else:
                corpus_score_dict = []
            for count, summ in enumerate(summaries):
                print(count)
                stats = get_stats(client, summ)
                if aggregate:
                    corpus_score_dict.update(stats)
                else:
                    corpus_score_dict.append(stats)
            if aggregate:
                for key in corpus_score_dict.keys():
                    corpus_score_dict[key] /= float(len(summaries))
            return corpus_score_dict

    @property
    def supports_multi_ref(self):
        return False
