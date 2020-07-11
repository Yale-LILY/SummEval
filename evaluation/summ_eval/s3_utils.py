# pylint: disable=C0103,W0621

import math
import collections
import os
import pickle as pkl
import six

from nltk.util import ngrams
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

import numpy as np
from scipy import spatial


tokenizer = RegexpTokenizer(r'\w+')
stopset = frozenset(stopwords.words('english'))
stemmer = SnowballStemmer("english")

###################################################
###                Pre-Processing
###################################################

def is_ngram_content(ngram):
    for gram in ngram:
        if not gram in stopset:
            return True
    return False

def get_all_content_words(sentences, N, tokenize):
    all_words = []
    if tokenize:
        for s in sentences:
            all_words.extend([stemmer.stem(r) for r in tokenizer.tokenize(s)])
    else:
        if isinstance(sentences, list):
            all_words = sentences[0].split()
        else:
            all_words = sentences.split()

    if N == 1:
        content_words = [w for w in all_words if w not in stopset]
    else:
        content_words = all_words

    normalized_content_words = map(normalize_word, content_words)
    if N > 1:
        return [gram for gram in ngrams(normalized_content_words, N) if is_ngram_content(gram)]
    return normalized_content_words

def compute_word_freq(words):
    word_freq = {}
    for w in words:
        word_freq[w] = word_freq.get(w, 0) + 1
    return word_freq

def compute_tf(sentences, N=1, tokenize=False):
    content_words = get_all_content_words(sentences, N, tokenize) ## stemmed
    content_words = list(content_words)
    content_words_count = len(content_words)
    content_words_freq = compute_word_freq(content_words)

    content_word_tf = dict((w, f / float(content_words_count)) for w, f in content_words_freq.items())
    return content_word_tf

def pre_process_summary(summary, ngrams, tokenize):
    return compute_tf(summary, ngrams, tokenize)

###################################################
###                Metrics
###################################################

def KL_Divergence(summary_freq, doc_freq):
    sum_val = 0
    for w, f in summary_freq.items():
        if w in doc_freq:
            sum_val += f * math.log(f / float(doc_freq[w]))

    if np.isnan(sum_val):
        raise Exception("KL_Divergence returns NaN")

    return sum_val

def compute_average_freq(l_freq_1, l_freq_2):
    average_freq = {}
    keys = set(l_freq_1.keys()) | set(l_freq_2.keys())

    for k in keys:
        s_1 = l_freq_1.get(k, 0)
        s_2 = l_freq_2.get(k, 0)
        average_freq[k] = (s_1 + s_2) / 2.

    return average_freq

def JS_Divergence(doc_freq, summary_freq):
    average_freq = compute_average_freq(summary_freq, doc_freq)
    js = (KL_Divergence(summary_freq, average_freq) + KL_Divergence(doc_freq, average_freq)) / 2.

    if np.isnan(js):
        raise Exception("JS_Divergence returns NaN")

    return js

def JS_eval(summary, references, n, tokenize):
    sum_rep = pre_process_summary(summary, n, tokenize)
    refs_reps = [pre_process_summary(ref, n, tokenize) for ref in references]

    avg = 0.
    for ref_rep in refs_reps:
        avg += JS_Divergence(ref_rep, sum_rep)

    return avg / float(len(references))


###################################################
###             Pre-Processing
###################################################

def get_all_content_words_stem(sentences, stem, tokenize=False):
    all_words = []
    if tokenize:
        for s in sentences:
            if stem:
                all_words.extend([stemmer.stem(r) for r in tokenizer.tokenize(s)])
            else:
                all_words.extend(tokenizer.tokenize(s))
    else:
        if isinstance(sentences, list):
            all_words = sentences[0].split()
        else:
            all_words = sentences.split()

    normalized_content_words = list(map(normalize_word, all_words))
    return normalized_content_words

def pre_process_summary_stem(summary, stem=True, tokenize=True):
    summary_ngrams = get_all_content_words_stem(summary, stem, tokenize=tokenize)
    return summary_ngrams

def _ngrams(words, n):
    queue = collections.deque(maxlen=n)
    for w in words:
        queue.append(w)
        if len(queue) == n:
            yield tuple(queue)

def _ngram_counts(words, n):
    return collections.Counter(_ngrams(words, n))

def _ngram_count(words, n):
    return max(len(words) - n + 1, 0)

def _counter_overlap(counter1, counter2):
    result = 0
    for k, v in six.iteritems(counter1):
        result += min(v, counter2[k])
    return result

def _safe_divide(numerator, denominator):
    if denominator > 0:
        return numerator / denominator
    else:
        return 0

def _safe_f1(matches, recall_total, precision_total, alpha, return_all=False):
    recall_score = _safe_divide(matches, recall_total)
    precision_score = _safe_divide(matches, precision_total)
    denom = (1.0 - alpha) * precision_score + alpha * recall_score
    if denom > 0.0:
        if return_all:
            return precision_score, recall_score, (precision_score * recall_score) / denom
        else:
            return (precision_score * recall_score) / denom
    else:
        if return_all:
            return precision_score, recall_score, 0.0
        else:
            return 0.0

def rouge_n(peer, models, n, alpha, tokenize):
    """
    Compute the ROUGE-N score of a peer with respect to one or more models, for
    a given value of `n`.
    """

    if len(models) == 1 and isinstance(models[0], str):
        models = [models]
    peer = pre_process_summary_stem(peer, True, tokenize)
    models = [pre_process_summary_stem(model, True, tokenize) for model in models]

    matches = 0
    recall_total = 0
    peer_counter = _ngram_counts(peer, n)
    for model in models:
        model_counter = _ngram_counts(model, n)
        matches += _counter_overlap(peer_counter, model_counter)
        recall_total += _ngram_count(model, n)
    precision_total = len(models) * _ngram_count(peer, n)
    return _safe_f1(matches, recall_total, precision_total, alpha)

def _has_embedding(ngram, embs):
    for w in ngram:
        if not w in embs:
            return False
    return True

def _get_embedding(ngram, embs):
    res = []
    for w in ngram:
        res.append(embs[w])
    return np.sum(np.array(res), 0)

def _find_closest(ngram, counter, embs):
    ## If there is nothin to match, nothing is matched
    if len(counter) == 0:
        return "", 0, 0

    ## If we do not have embedding for it, we try lexical matching
    if not _has_embedding(ngram, embs):
        if ngram in counter:
            return ngram, counter[ngram], 1
        else:
            return "", 0, 0

    ranking_list = []
    ngram_emb = _get_embedding(ngram, embs)
    for k, v in six.iteritems(counter):
        ## First check if there is an exact match
        if k == ngram:
            ranking_list.append((k, v, 1.))
            continue

        ## if no exact match and no embeddings: no match
        if not _has_embedding(k, embs):
            ranking_list.append((k, v, 0.))
            continue

        ## soft matching based on embeddings similarity
        k_emb = _get_embedding(k, embs)
        ranking_list.append((k, v, 1 - spatial.distance.cosine(k_emb, ngram_emb)))

    ## Sort ranking list according to sim
    ranked_list = sorted(ranking_list, key=lambda tup: tup[2], reverse=True)

    ## extract top item
    return ranked_list[0]


def _soft_overlap(peer_counter, model_counter, embs):
    THRESHOLD = 0.8
    result = 0
    for k, v in six.iteritems(peer_counter):
        closest, count, sim = _find_closest(k, model_counter, embs)
        if sim < THRESHOLD:
            continue
        if count <= v:
            del model_counter[closest]
            result += count
        else:
            model_counter[closest] -= v
            result += v

    return result

def rouge_n_we(peer, models, embs, n, alpha=0.5, return_all=False, tokenize=False):
    """
    Compute the ROUGE-N-WE score of a peer with respect to one or more models, for
    a given value of `n`.
    """

    if len(models) == 1 and isinstance(models[0], str):
        models = [models]
    peer = pre_process_summary_stem(peer, False, tokenize)
    models = [pre_process_summary_stem(model, False, tokenize) for model in models]

    matches = 0
    recall_total = 0
    peer_counter = _ngram_counts(peer, n)
    for model in models:
        model_counter = _ngram_counts(model, n)
        matches += _soft_overlap(peer_counter, model_counter, embs)
        recall_total += _ngram_count(model, n)
    precision_total = len(models) * _ngram_count(peer, n)
    return _safe_f1(matches, recall_total, precision_total, alpha, return_all)

# convert to unicode and convert to lower case
def normalize_word(word):
    return word.lower()

def _convert_to_numpy(vector):
    return np.array([float(x) for x in vector])

def load_embeddings(filepath):
    dict_embedding = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.rstrip().split(" ")
            key = line[0]
            vector = line[1::]
            dict_embedding[key.lower()] = _convert_to_numpy(vector)
    return dict_embedding

def load_model(model_folder, tgt):
    model_file = [f for f in os.listdir(model_folder) if tgt in f]
    assert len(model_file) == 1, "Unable to find the correct model: " + str(tgt)
    model_file = os.path.join(model_folder, model_file[0])
    model = pkl.load(open(model_file, "rb"), encoding='latin1')
    return model

def S3(references, system_summary, word_embs, model_folder, tokenize):
    ### Extract features
    instance = extract_feature(references, system_summary, word_embs, tokenize)
    features = sorted(instance.keys())

    feature_vector = []
    for feat in features:
        feature_vector.append(instance[feat])

    ### Load models
    model_pyr = load_model(model_folder, 'pyr')
    model_resp = load_model(model_folder, 'resp')

    ### Apply models
    X = np.array([feature_vector])
    score_pyr = model_pyr.predict(X)[0]
    score_resp = model_resp.predict(X)[0]

    return (score_pyr, score_resp)

def extract_feature(references, summary_text, word_embs, tokenize):
    features = {}

    ## Get ROUGE_1, ROUGE-2 Recall
    features["ROUGE_1_R"] = rouge_n(summary_text, references, 1, 0., tokenize)
    features["ROUGE_2_R"] = rouge_n(summary_text, references, 2, 0., tokenize)

    ### Get JS
    features["JS_eval_1"] = JS_eval(summary_text, references, 1, tokenize)
    features["JS_eval_2"] = JS_eval(summary_text, references, 2, tokenize)

    features["ROUGE_1_R_WE"] = rouge_n_we(summary_text, references, word_embs, 1, 0., tokenize=tokenize)
    features["ROUGE_2_R_WE"] = rouge_n_we(summary_text, references, word_embs, 2, 0., tokenize=tokenize)

    return features
