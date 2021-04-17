# pylint: disable=W0221,C0103
import spacy
import gin
from summ_eval.summa_qa_utils import QA_Metric, QG_masked, evaluate_corpus
from summ_eval.metric import Metric

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print('Downloading the spacy en_core_web_sm model\n'
        "(don't worry, this will only happen once)", file=stderr)
    from spacy.cli import download
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

@gin.configurable
class SummaQAMetric(Metric):
    def __init__(self, batch_size=8, max_seq_len=384, use_gpu=True, tokenize=True):
        """
        SummaQA metric
        Makes use of code here:
                https://github.com/recitalAI/summa-qa
        Added batching, GPU usage for speed; makes use of more recent version of
                transformers library, which may result in slightly different numbers
                but should keep integrity of the metric

        NOTE: the metric as used in the paper "Answers Unite! Unsupervised Metrics
                for Reinforced Summarization Models" is calculated with respect to
                the source text (e.g. news article) as opposed to the reference.
                Question answering as a metric on reference text has been used in the
                paper "Question answering as an automatic evaluation metric for news
                article summarization" and you may also try this metric with the
                reference text.

        Args:
                :param batch_size: batch size to use for QA model
                :param max_seq_len: maximum sequence length for input to QA model;
                        truncates question, text pairs to this length
                :param use_gpu: whether to use GPU for QA model
                :param tokenize: whether to tokenize the input text; otherwise assumes
                        that your input is a spacy-processed Doc with .sents and .ents attributes
        """
        self.qa_metric = QA_Metric(batch_size=batch_size, max_seq_len=max_seq_len, use_gpu=use_gpu)
        self.question_generator = QG_masked()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.use_gpu = use_gpu
        self.tokenize = tokenize

    def evaluate_example(self, summary, input_text):
        if self.tokenize:
            input_text = nlp(input_text, disable=["tagger", "textcat"])
        masked_questions, answer_spans = self.question_generator.get_questions(input_text)
        score_dict = self.qa_metric.compute(masked_questions, answer_spans, summary)
        return score_dict

    def evaluate_batch(self, summaries, input_texts, aggregate=True):
        if self.tokenize:
            input_texts = [nlp(text, disable=["tagger", "textcat"]) for text in input_texts]
        scores = evaluate_corpus(input_texts, summaries, batch_size=self.batch_size, \
                  max_seq_len=self.max_seq_len, aggregate=aggregate)
        return scores

    @property
    def supports_multi_ref(self):
        return False
