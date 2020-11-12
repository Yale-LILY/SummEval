# pylint: disable=W0102,C0301,W1401,C0303,C0103,W0221,C0200,W0106,W0622,C0321,R1718,R0911,R1721,W0212
# Uses code from https://raw.githubusercontent.com/yg211/acl20-ref-free-eval/master/ref_free_metrics/

import copy
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AffinityPropagation
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

import networkx as nx


def normaliseList(ll,max_value=10.):
    minv = min(ll)
    maxv = max(ll)
    gap = maxv-minv

    new_ll = [(x-minv)*max_value/gap for x in ll]

    return new_ll



def get_human_score(topic, summ_name, human):
    block = summ_name.split('-')[1].split('.')[0]
    id = summ_name.split('.')[-1]
    key = 'topic{}-{}_sum{}'.format(topic.split('.')[0],block,id)
    if key not in human: return None
    else: return human[key]

def get_idf_weights(ref_vecs):
    sim_matrix = cosine_similarity(ref_vecs,ref_vecs)
    #dfs = [np.sort(sim_matrix[i])[-2] for i in range(len(ref_vecs))]
    dfs = [np.sum(sim_matrix[i])-1. for i in range(len(ref_vecs))]
    dfs = [1.*d/(len(ref_vecs)-1) for d in dfs]
    dfs = [(d+1.)/2. for d in dfs]
    idf = [-1.*np.log(df) for df in dfs]
    return idf


def get_ref_vecs(doc_sent_vecs, doc_sent_weights, info_dic):
    ref_dic = {}
    docs = set([info_dic[k]['doc'] for k in info_dic])
    for dd in docs:
        ref_dic[dd] = [i for i in range(len(doc_sent_weights)) if doc_sent_weights[i]>=0.1 and info_dic[i]['doc']==dd]
    vecs = []
    for dd in ref_dic:
        allv = np.array(doc_sent_vecs)[ref_dic[dd]]
        meanv = np.mean(allv,axis=0)
        vecs.append(meanv)
    return vecs


def get_sim_metric(summ_vec_list, doc_sent_vecs, doc_sent_weights, method='cos'):
    #print('weights', doc_sent_weights)
    # get the avg doc vec, then cosine
    if method == 'cos':
        summ_vec = np.mean(np.array(summ_vec_list),axis=0)
        dvec = np.matmul(np.array(doc_sent_weights).reshape(1,-1),  np.array(doc_sent_vecs))
        return cosine_similarity(dvec,summ_vec.reshape(1,-1))[0][0]
        # below: good performance with true_ref, poorer performance with other pseduo-refs
        #ref_vecs = get_ref_vecs(doc_sent_vecs, doc_sent_weights, info_dic)
        #sims = cosine_similarity(np.array(ref_vecs), np.array(summ_vec).reshape(1,-1))
        #return np.mean(sims)
    # bert-score, quicker to run and gives similar performance to mover-bert-score
    else:
        ref_vecs = [doc_sent_vecs[i] for i in range(len(doc_sent_weights)) if doc_sent_weights[i]>0.1]
        #ref_vecs = get_ref_vecs(doc_sent_vecs, doc_sent_weights, info_dic)
        weights = [doc_sent_weights[i] for i in range(len(doc_sent_weights)) if doc_sent_weights[i]>0.1]
        idf_weights = get_idf_weights(ref_vecs)
        sim_matrix = cosine_similarity(np.array(ref_vecs),np.array(summ_vec_list))
        recall = np.mean(np.max(sim_matrix,axis=1))
        idf_recall = np.dot(np.max(sim_matrix,axis=1),idf_weights)/np.sum(idf_weights)
        precision = np.mean(np.max(sim_matrix,axis=0))
        if recall+precision == 0:
            f1 = None
        else:
            f1 = 2.*recall*precision/(recall+precision)
            idf_f1 = 2.*idf_recall*precision/(idf_recall+precision)
        if method.lower().startswith('f'): return f1
        elif method.lower().startswith('r'): return recall
        elif method.lower().startswith('p'): return precision
        elif method.lower().startswith('idf'):
            if 'recall' in method: return idf_recall
            elif 'f1' in method: return idf_f1
            else: return None
        elif method.lower().startswith('w'): return np.dot(np.array(np.max(sim_matrix,axis=1)),np.array(weights))/np.sum(weights)
        else: return None


def parse_docs(docs,bert_model):
    all_sents = []
    sent_index = {}
    cnt = 0
    for dd in docs:
        dname = dd[0].split('/')[-1]
        doc_len = len(dd[1])
        for i, sent in enumerate(dd[1]):
            sent_index[cnt] = {'doc': dname, 'text': sent, 'inside_doc_idx': i, 'doc_len': doc_len,
                               'inside_doc_position_ration': i * 1. / doc_len}
            cnt += 1
            all_sents.append(sent)
    all_vecs = None
    if bert_model is not None:
        all_vecs = bert_model.encode(all_sents)
    return sent_index, all_vecs #, all_sents


def parse_refs(refs,model):
    all_sents = []
    sent_index = {}
    cnt = 0
    for i,rr in enumerate(refs):
        if len(rr[1]) == 1: # TAC09, one piece of text
            ref_sents = sent_tokenize(rr[1][0])
        else: # TAC08, a list of sentences
            ref_sents = rr[1]
        ref_name = 'ref{}'.format(i)
        for j, sent in enumerate(ref_sents):
            sent_index[cnt] = {'doc': ref_name, 'text': sent, 'inside_doc_idx': j, 'doc_len': len(ref_sents),
                               'inside_doc_position_ration': j * 1. / len(ref_sents)}
            cnt += 1
            all_sents.append(sent)
    all_vecs = None
    if model is not None:
        all_vecs = model.encode(all_sents)
    return sent_index, all_vecs



# build pseudo references; the selected sentences will have non-zero weights
def get_weights(sent_info_dic:dict, sent_vecs:list, metric:str):
    # use full source docs as the pseudo ref
    if metric == 'full_doc':
        weights = [1.]*len(sent_info_dic)
    # randomly extract N sentences as the pseudo ref
    elif metric.startswith('random'):
        if '_' in metric:
            ref_length = int(metric.split('_')[1])
        else:
            ref_length = 10 # by default we randomly select 10 sents from each doc as the pseudo-ref
        ridx = np.arange(len(sent_info_dic))
        np.random.shuffle(ridx)
        weights = [1. if i in ridx[:ref_length] else 0. for i in range(len(ridx))]
    # extract top N sentences as the pseudo ref
    elif metric.startswith('top'):
        if '_' in metric:
            topn = int(metric.split('_')[0][3:])
            thres = float(metric.split('_')[1])
        else:
            topn = int(metric[3:])
            thres = -1
        weights = get_top_weights(sent_info_dic, topn)
        if thres > 0:
            get_other_weights(sent_vecs, sent_info_dic, weights, thres)
    # SBert based LexRank, SLR in the paper
    elif metric.startswith('indep_graph') or metric.startswith('global_graph'):
        eles = metric.split('_')
        num = int(eles[2][3:])
        if 'extra' in metric:
            assert len(eles) == 5
            top_n = int(eles[3][5:])
            extra_amp = float(eles[-1])
        else:
            extra_amp = None
            top_n = None
        if 'indep' in metric:
            weights = get_indep_graph_weights(sent_info_dic, sent_vecs, num, top_n, extra_amp)
        else:
            weights = get_global_graph_weights(sent_info_dic, sent_vecs, num, top_n, extra_amp)
    # SBert-based cluster, global version (use all sents from all source docs to build a graph); SC_{G} in the paper
    elif metric.startswith('global_cluster'):
        weights = get_global_cluster_weights(sent_vecs)
    # SBert-based cluster, independent version (use sents from each source doc to build a graph); SC_{I} in the paper
    elif metric.startswith('indep_cluster'):
        weights = get_indep_cluster_weights(sent_info_dic, sent_vecs)
    elif metric.startswith('simmax'):
        simmax = float(metric.split('_')[1])
        weights = get_top_sim_weights(sent_info_dic, sent_vecs,simmax)
    return weights

def parse_documents(docs, bert_model, ref_metric, debug=False):
    if ref_metric == 'true_ref': # use golden ref as pseudo ref; the upper bound case
        sent_info_dic, sent_vecs = parse_refs(docs,bert_model)
        sents_weights = [1.] * len(sent_info_dic)
    else: # use strategy specified by 'ref_metric' to construct pseudo refs
        sent_info_dic, sent_vecs = parse_docs(docs,bert_model)
        sents_weights = get_weights(sent_info_dic, sent_vecs, ref_metric)
    if debug:
        pseudo_ref = [sent_info_dic[k]['text'] for k in sent_info_dic if sents_weights[k]>0.1]
        print('=====pseudo ref=====')
        print('\n'.join(pseudo_ref))
    return sent_info_dic, sent_vecs, sents_weights



def get_doc_simtop(sim_matrix, max_sim_value):
    nn = sim_matrix.shape[0]
    for i in range(1,sim_matrix.shape[0]):
        if np.max(sim_matrix[i][:i])>max_sim_value: 
            nn = i
            break
    return nn


def get_top_sim_weights(sent_info, full_vec_list, max_sim_value):
    doc_names = set([sent_info[k]['doc'] for k in sent_info])
    weights = [0.]*len(sent_info)
    for dn in doc_names:
        doc_idx = [k for k in sent_info if sent_info[k]['doc']==dn]
        sim_matrix = cosine_similarity(np.array(full_vec_list)[doc_idx], np.array(full_vec_list)[doc_idx])
        nn = get_doc_simtop(sim_matrix, max_sim_value)
        for i in range(np.min(doc_idx),np.min(doc_idx)+nn): weights[i] = 1.
    return weights


def get_top_weights(sent_index, topn):
    weights = []
    for i in range(len(sent_index)):
        if sent_index[i]['inside_doc_idx'] < topn:
            weights.append(1.)
        else:
            weights.append(0.)
    return weights


def get_subgraph(sim_matrix, threshold):
    gg = nx.Graph()
    for i in range(0,sim_matrix.shape[0]-1):
        for j in range(i+1,sim_matrix.shape[0]):
            if sim_matrix[i][j] >= threshold:
                gg.add_node(i)
                gg.add_node(j)
                gg.add_edge(i,j)
    subgraph = [gg.subgraph(c) for c in nx.connected_components(gg)]
    subgraph_nodes = [list(sg._node.keys()) for sg in subgraph]
    return list(subgraph_nodes)


def get_other_weights(full_vec_list, sent_index, weights, thres):
    similarity_matrix = cosine_similarity(full_vec_list, full_vec_list)
    subgraphs = get_subgraph(similarity_matrix, thres)
    for sg in subgraphs:
        if any(weights[n]>=0.9 for n in sg): continue #ignore the subgraph similar to a top sentence
        if len(set([sent_index[n]['doc'] for n in sg])) < 2: continue #must appear in multiple documents
        for n in sg: weights[n]=1./len(sg)
        #print(sg,'added to weights')


def graph_centrality_weight(similarity_matrix):
    weights_list = [np.sum(similarity_matrix[i])-1. for i in range(similarity_matrix.shape[0])]
    return weights_list


def graph_weights(full_vec_list):
    similarity_matrix = cosine_similarity(full_vec_list, full_vec_list)
    weights_list = graph_centrality_weight(similarity_matrix)
    return weights_list


def get_indep_graph_weights(sent_info_dic, sent_vecs, num, top_n, extra_ratio):
    doc_names = set([sent_info_dic[key]['doc'] for key in sent_info_dic])
    wanted_id = []
    for dname in doc_names:
        ids = np.array([key for key in sent_info_dic if sent_info_dic[key]['doc']==dname])
        doc_weights = np.array(graph_weights(np.array(sent_vecs)[ids]))
        if top_n is not None: 
            for j in range(top_n): 
                if j>=len(doc_weights): break
                doc_weights[j] *= extra_ratio
        wanted_id.extend(list(ids[doc_weights.argsort()[-num:]]))
    weights = [0.]*len(sent_vecs)
    for ii in wanted_id: weights[ii] = 1.
    return weights


def get_global_graph_weights(sent_info_dic, sent_vecs, num, top_n, extra_ratio):
    raw_weights = graph_weights(sent_vecs)
    if top_n is not None:
        top_ids = [i for i in sent_info_dic if sent_info_dic[i]['inside_doc_idx']<top_n]
        adjusted_weights = [w*extra_ratio if j in top_ids else w for j,w in enumerate(raw_weights) ]
    else:
        adjusted_weights = raw_weights
    wanted_id = np.array(adjusted_weights).argsort()[-num:]
    weights = [0.] * len(sent_vecs)
    for ii in wanted_id: weights[ii] = 1.
    return weights


def get_indep_cluster_weights(sent_info_dic, sent_vecs):
    doc_names = set([sent_info_dic[key]['doc'] for key in sent_info_dic])
    sums = [np.sum(sv) for sv in sent_vecs]
    wanted_ids = []
    for dname in doc_names:
        sids = np.array([key for key in sent_info_dic if sent_info_dic[key]['doc']==dname])
        clustering = AffinityPropagation().fit(np.array(sent_vecs)[sids])
        centers = clustering.cluster_centers_
        for cc in centers: wanted_ids.append(sums.index(np.sum(cc)))
    print('indep cluster, pseudo-ref sent num', len(wanted_ids))
    weights = [1. if i in wanted_ids else 0. for i in range(len(sent_vecs))]
    return weights


def get_global_cluster_weights(sent_vecs):
    clustering = AffinityPropagation().fit(sent_vecs)
    centers = clustering.cluster_centers_
    print('global cluster, pseudo-ref sent num', len(centers))
    sums = [np.sum(sv) for sv in sent_vecs]
    ids = []
    for cc in centers: ids.append(sums.index(np.sum(cc)))
    assert len(ids) == len(centers)
    weights = [1. if i in ids else 0. for i in range(len(sent_vecs))]
    return weights


def get_all_token_vecs(model, sent_info_dict):
    all_sents = [sent_info_dict[i]['text'] for i in sent_info_dict]
    all_vecs, all_tokens = model.encode(all_sents, token_vecs=True)
    assert len(all_vecs) == len(all_tokens)
    for i in range(len(all_vecs)):
        assert len(all_vecs[i]) == len(all_tokens[i])
    return all_vecs, all_tokens

def build_pseudo_ref(sent_info_dic, sents_weights, all_tokens, all_token_vecs):
    ref_dic = {k:sent_info_dic[k] for k in sent_info_dic if sents_weights[k]>=0.1}
    # get sents in the pseudo ref
    ref_sources = set(ref_dic[k]['doc'] for k in ref_dic)
    ref_idxs = []
    if len(ref_dic) >= 15: #all(['ref' in rs for rs in ref_sources]):
        # group sentences from the same doc into one pseudo ref
        for rs in ref_sources:
            ref_idxs.append([k for k in ref_dic if ref_dic[k]['doc']==rs])
    else:
        ref_idxs.append([k for k in ref_dic])
    # get vecs and tokens of the pseudo reference
    ref_vecs = []
    ref_tokens = []
    for ref in ref_idxs:
        vv, tt = kill_stopwords(ref, all_token_vecs, all_tokens)
        ref_vecs.append(vv)
        ref_tokens.append(tt)
    return ref_vecs, ref_tokens

def get_sbert_score(ref_token_vecs, summ_token_vecs, sim_metric):
    recall_list = []
    precision_list = []
    f1_list = []
    empty_summs_ids = []

    for i,rvecs in enumerate(ref_token_vecs):
        r_recall_list = []
        r_precision_list = []
        r_f1_list = []
        for j,svecs in enumerate(summ_token_vecs):
            if svecs is None:
                empty_summs_ids.append(j)
                r_recall_list.append(None)
                r_precision_list.append(None)
                r_f1_list.append(None)
                continue
            sim_matrix = cosine_similarity(rvecs,svecs)
            recall = np.mean(np.max(sim_matrix, axis=1))
            precision = np.mean(np.max(sim_matrix, axis=0))
            f1 = 2. * recall * precision / (recall + precision)
            r_recall_list.append(recall)
            r_precision_list.append(precision)
            r_f1_list.append(f1)
        recall_list.append(r_recall_list)
        precision_list.append(r_precision_list)
        f1_list.append(r_f1_list)
    empty_summs_ids = list(set(empty_summs_ids))
    recall_list = np.array(recall_list)
    precision_list = np.array(precision_list)
    f1_list = np.array(f1_list)
    if 'recall' in sim_metric:
        scores = []
        for i in range(len(summ_token_vecs)):
            if i in empty_summs_ids: scores.append(None)
            else: scores.append(np.mean(recall_list[:,i]))
        return scores
        #return np.mean(np.array(recall_list), axis=0)
    elif 'precision' in sim_metric:
        scores = []
        for i in range(len(summ_token_vecs)):
            if i in empty_summs_ids: scores.append(None)
            else: scores.append(np.mean(precision_list[:,i]))
        return scores
        #return np.mean(np.array(precision_list), axis=0)
    else:
        assert 'f1' in sim_metric
        scores = []
        for i in range(len(summ_token_vecs)):
            if i in empty_summs_ids: scores.append(None)
            else: scores.append(np.mean(f1_list[:,i]))
        return scores
        #return np.mean(np.mean(f1_list),axis=0)

def get_token_vecs(model, sents, remove_stopwords=True):
    if len(sents) == 0: return None, None
    vecs, tokens = model.encode(sents, token_vecs=True)
    for i, rtv in enumerate(vecs):
        if i==0:
            full_vec = rtv
            full_token = tokens[i]
        else:
            full_vec = np.row_stack((full_vec, rtv))
            full_token.extend(tokens[i])
    if remove_stopwords:
        mystopwords = list(set(stopwords.words()))
        mystopwords.extend(['[cls]','[sep]'])
        wanted_idx = [j for j,tk in enumerate(full_token) if tk.lower() not in mystopwords]
    else:
        wanted_idx = [k for k in range(len(full_token))]
    return full_vec[wanted_idx], np.array(full_token)[wanted_idx]

def kill_stopwords(sent_idx, all_token_vecs, all_tokens):
    for i,si in enumerate(sent_idx):
        assert len(all_token_vecs[si]) == len(all_tokens[si])
        if i == 0:
            full_vec = copy.deepcopy(all_token_vecs[si])
            full_token = copy.deepcopy(all_tokens[si])
        else:
            full_vec = np.row_stack((full_vec, all_token_vecs[si]))
            full_token.extend(all_tokens[si])
    # For now we're hard-coding English for CNNDM
    mystopwords = list(set(stopwords.words()))
    mystopwords.extend(['[cls]','[sep]'])
    wanted_idx = [j for j,tk in enumerate(full_token) if tk.lower() not in mystopwords]
    return full_vec[wanted_idx], np.array(full_token)[wanted_idx]
