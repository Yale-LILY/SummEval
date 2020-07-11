# pylint: disable=C0301,C0103,E1102,R0201
from collections import Counter
import re
import string
import torch
from transformers import BertTokenizer, BertForQuestionAnswering

 # code from huggingface see https://huggingface.co/transformers/model_doc/bert.html#bertforquestionanswering

class QA_Bert():
    def __init__(self):

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        self.SEP_id = self.tokenizer.encode('[SEP]')[0]

    def predict(self, input_ids, token_type_ids, attention_mask):

        start_scores, end_scores = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        start_scores = torch.functional.F.softmax(start_scores, -1) * token_type_ids.float()
        end_scores = torch.functional.F.softmax(end_scores, -1) * token_type_ids.float()

        start_values, start_indices = start_scores.topk(1)
        end_values, end_indices = end_scores.topk(1)

        probs = []
        asws = []
        for idx, (input_id, start_index, end_index) in enumerate(zip(input_ids, start_indices, end_indices)):
            cur_toks = self.tokenizer.convert_ids_to_tokens(input_id)
            asw = ' '.join(cur_toks[start_index[0] : end_index[0]+1])
            prob = start_values[idx][0] * end_values[idx][0]
            asws.append(asw)
            probs.append(prob.item())
        return asws, probs


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))



def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


class QG_masked:
    """
    Cloze style Question Generator based on spacy named entity recognition
    """

    def __init__(self):
        pass

    def get_questions(self, text_input):
        """
        Generate a list of questions on a text
        Args:
          text_input: a string
        Returns:
          a list of question
        """
        masked_questions = []
        asws = []

        for sent in text_input.sents:
            for ent in sent.ents:
                id_start = ent.start_char - sent.start_char
                id_end = ent.start_char - sent.start_char + len(ent.text)
                masked_question = sent.text[:id_start] + \
                    "MASKED" + sent.text[id_end:]
                masked_questions.append(masked_question)
                asws.append(ent.text)

        return masked_questions, asws


class QA_Metric:
    """
    Question Answering based metric
    """

    def __init__(self, model=None, batch_size=8, max_seq_len=384, use_gpu=True):

        if model is None:
            model = QA_Bert()
        self.model = model
        if torch.cuda.is_available() and use_gpu:
            self.gpu = True
            self.model.model.to("cuda")
        else:
            self.gpu = False
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

    def compute(self, questions, true_asws, evaluated_text):
        """
        Calculate the QA scores for a given text we want to evaluate and a list of questions and their answers.
        Args:
          questions: a list of string
          true_asws: a list of string
          evaluated_text: a string
        Returns:
          a dict containing the probability score and the f-score
        """
        if not questions:
            return {"summaqa_avg_prob": 0, "summaqa_avg_fscore": 0}

        score_prob, score_f = 0, 0
        probs = []
        asws = []
        slines = []
        for count, (question, true_asw) in enumerate(zip(questions, true_asws)):
            if count % self.batch_size == 0 and count != 0:
                input_ids = torch.tensor([ex['input_ids'] for ex in slines])
                token_type_ids = torch.tensor([ex['token_type_ids'] for ex in slines])
                attention_mask = torch.tensor([ex['attention_mask'] for ex in slines])
                if self.gpu:
                    input_ids = input_ids.to("cuda")
                    token_type_ids = token_type_ids.to("cuda")
                    attention_mask = attention_mask.to("cuda")
                asw_pred, prob = self.model.predict(input_ids, token_type_ids, attention_mask)
                asws.extend(asw_pred)
                probs.extend(prob)
                slines = []
            cur_dict = self.model.tokenizer.encode_plus(question, evaluated_text, max_length=self.max_seq_len, pad_to_max_length=True, return_token_type_ids=True)
            slines.append(cur_dict)
        if slines != []:
            input_ids = torch.tensor([ex['input_ids'] for ex in slines])
            token_type_ids = torch.tensor([ex['token_type_ids'] for ex in slines])
            attention_mask = torch.tensor([ex['attention_mask'] for ex in slines])
            if self.gpu:
                input_ids = input_ids.to("cuda")
                token_type_ids = token_type_ids.to("cuda")
                attention_mask = attention_mask.to("cuda")
            asw_pred, prob = self.model.predict(input_ids, token_type_ids, attention_mask)
            asws.extend(asw_pred)
            probs.extend(prob)
        for asw, true_asw in zip(asws, true_asws):
            score_f += f1_score(asw, true_asw)
        score_prob = sum(probs)

        return {"summaqa_avg_prob": score_prob/len(questions), "summaqa_avg_fscore": score_f/len(questions)}


def evaluate_corpus(srcs, gens, model=None, questionss=None, aswss=None, batch_size=8, max_seq_len=384, aggregate=True):
    """
    Calculate the QA scores for an entire corpus.
    Args:
      srcs: a list of string (one string per document)
      gens: a list of string (one string per summary)
      model: [optional]: any model that fits the function predict in qa_models; by default BERT_QA
      questionss: [optional]: a list of list with the questions already generated for each src. If None, it will generate it.
      aswss: [optional]: a list of list with the ground truth asws for the questions (questionss). If None, it will generate it as well.
    Returns:
      a dict containing the probability score and f-score, averaged for the corpus
    """
    assert any([questionss, aswss]) == all([questionss, aswss]
                                           ), "questionss/aswss should be None if the other is None"

    # if questionss is None initialize a question generator
    if not questionss:
        question_generator = QG_masked()
    # initialize the metric with a QA model
    qa_metric = QA_Metric(model, batch_size=batch_size, max_seq_len=max_seq_len)

    if aggregate:
        global_score = {"summaqa_avg_prob": 0, "summaqa_avg_fscore": 0}
    else:
        global_score = []

    for i, (src, gen) in enumerate(zip(srcs, gens)):
        # if questionss is None, generate the questions and answers else get the corrisponding ones.
        if not questionss:
            masked_questions, masked_question_asws = question_generator.get_questions(src)
        else:
            masked_questions, masked_question_asws = questionss[i], aswss[i]

        # compute the metric
        gen_score = qa_metric.compute(
                masked_questions, masked_question_asws, gen)
        if aggregate:
            global_score['summaqa_avg_prob'] += gen_score['summaqa_avg_prob']
            global_score['summaqa_avg_fscore'] += gen_score['summaqa_avg_fscore']
        else:
            global_score.append(gen_score)

    if aggregate:
        # average it
        global_score['summaqa_avg_prob'] = global_score['summaqa_avg_prob'] / len(srcs)
        global_score['summaqa_avg_fscore'] = global_score['summaqa_avg_fscore'] / len(srcs)

    return global_score
