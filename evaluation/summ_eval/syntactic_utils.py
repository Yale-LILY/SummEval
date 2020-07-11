# pylint: disable=C0103,W0632,E1101,C0301

def get_stats(client, summary):
    patterncount = []
    ann = client.annotate(summary)
    w = 0
    for sent in ann.sentence:
        w += len(sent.token)
    for pattern in patternlist:
        count = 0
        matches = client.tregrex(summary, pattern.replace("'", ""))
        for sent in matches['sentences']:
            count += len(sent.keys())
        patterncount.append(int(count))
    patterncount[7] = patterncount[-4] + patterncount[-5] + patterncount[-6]
    patterncount[2] = patterncount[2] + patterncount[-3]
    patterncount[3] = patterncount[3] + patterncount[-2]
    patterncount[1] = patterncount[1] + patterncount[-1]

    output = []
    output.append(w)
    for count in patterncount[:8]:
        output.append(count)

    #list of frequencies of structures other than words
    [s, vp, c, t, dc, ct, cp, cn] = patterncount[:8]

    #compute the 14 syntactic complexity indices
    mls = division(w, s)
    mlt = division(w, t)
    mlc = division(w, c)
    c_s = division(c, s)
    vp_t = division(vp, t)
    c_t = division(c, t)
    dc_c = division(dc, c)
    dc_t = division(dc, t)
    t_s = division(t, s)
    ct_t = division(ct, t)
    cp_t = division(cp, t)
    cp_c = division(cp, c)
    cn_t = division(cn, t)
    cn_c = division(cn, c)

    #add syntactic complexity indices to output
    for ratio in [mls, mlt, mlc, c_s, vp_t, c_t, dc_c, dc_t, t_s, ct_t, cp_t, cp_c, cn_t, cn_c]:
        output.append(ratio)

    output_dict = {}
    for field, score in zip(fields, output):
        output_dict[field] = score
    return output_dict



def division(x, y):
    if float(x) == 0 or float(y) == 0:
        return 0
    return float(x)/float(y)

#sentence (S)
s_p = "'ROOT'"
#verb phrase (VP)
vp_p = "'VP > S|SINV|SQ'"
vp_q_p = "'MD|VBZ|VBP|VBD > (SQ !< VP)'"

#clause (C)
c_p = "'S|SINV|SQ [> ROOT <, (VP <# VB) | <# MD|VBZ|VBP|VBD | < (VP [<# MD|VBP|VBZ|VBD | < CC < (VP <# MD|VBP|VBZ|VBD)])]'"

#T-unit (T)
t_p = "'S|SBARQ|SINV|SQ > ROOT | [$-- S|SBARQ|SINV|SQ !>> SBAR|VP]'"

#dependent clause (DC)
dc_p = "'SBAR < (S|SINV|SQ [> ROOT <, (VP <# VB) | <# MD|VBZ|VBP|VBD | < (VP [<# MD|VBP|VBZ|VBD | < CC < (VP <# MD|VBP|VBZ|VBD)])])'"

#complex T-unit (CT)
ct_p = "'S|SBARQ|SINV|SQ [> ROOT | [$-- S|SBARQ|SINV|SQ !>> SBAR|VP]] << (SBAR < (S|SINV|SQ [> ROOT <, (VP <# VB) | <# MD|VBZ|VBP|VBD | < (VP [<# MD|VBP|VBZ|VBD | < CC < (VP <# MD|VBP|VBZ|VBD)])]))'"
#coordinate phrase (CP)
cp_p = "'ADJP|ADVP|NP|VP < CC'"
#complex nominal (CN)
cn1_p = "'NP !> NP [<< JJ|POS|PP|S|VBG | << (NP $++ NP !$+ CC)]'"
cn2_p = "SBAR [<# WHNP | <# (IN < That|that|For|for) | <, S]  [$+ VP | > VP]"
cn3_p = "'S < (VP <# VBG|TO) $+ VP'"

#fragment clause
fc_p = "'FRAG > ROOT !<< (S|SINV|SQ [> ROOT <, (VP <# VB) | <# MD|VBZ|VBP|VBD | < (VP [<# MD|VBP|VBZ|VBD | < CC < (VP <# MD|VBP|VBZ|VBD)])])'"

#fragment T-unit
ft_p = "'FRAG > ROOT !<< (S|SBARQ|SINV|SQ > ROOT | [$-- S|SBARQ|SINV|SQ !>> SBAR|VP])'"
patternlist = [s_p, vp_p, c_p, t_p, dc_p, ct_p, cp_p, cn1_p, cn2_p, cn3_p, fc_p, ft_p, vp_q_p]

fields = "Words,Sentences,Verb Phrases,Clauses,T-units,Dependent Clauses,Complex T-units,Coordinate Phrases,"\
                "Complex Nominals,words/sentence,words/t-unit,words/clause,clauses/sent,"\
                "verb phrases/t-unit,clauses/t-unit,dependent clauses/clause,dependent clauses/t-unit,"\
                "t-units/sentence,complex t-units/t-units,coordinate phrases/t-unit,coordinate-phrases/clauses,"\
                "complex nominals/t-unit,complex nominals/clause"
fields = fields.split(",")
