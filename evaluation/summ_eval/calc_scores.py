# pylint: disable=C0103,C0301,W0702,W0703
import sys
import json
import argparse
from collections import defaultdict
import gin
#from summ_eval.test_util import metrics_description
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
import spacy

def cli_main():
    #parser = argparse.ArgumentParser(description=metrics_description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser = argparse.ArgumentParser(description="predictor")
    parser.add_argument('--config-file', type=str, help='config file with metric parameters')
    parser.add_argument('--metrics', type=str, help='comma-separated string of metrics')
    parser.add_argument('--aggregate', type=bool, help='whether to aggregate scores')
    parser.add_argument('--jsonl-file', type=str, help='input jsonl file to score')
    parser.add_argument('--article-file', type=str, help='input article file')
    parser.add_argument('--summ-file', type=str, help='input summary file')
    parser.add_argument('--ref-file', type=str, help='input reference file')
    parser.add_argument('--output-file', type=str, help='output file')
    parser.add_argument('--eos', type=str, help='EOS for ROUGE (if reference not supplied as list)')
    args = parser.parse_args()

    # =====================================
    # INITIALIZE METRICS
    gin.parse_config_file(args.config_file)
    toks_needed = set()
    metrics = [x.strip() for x in args.metrics.split(",")]
    metrics_dict = {}
    if "rouge" in metrics:
        from summ_eval.rouge_metric import RougeMetric
        metrics_dict["rouge"] = RougeMetric()
        toks_needed.add("line_delimited")

    if "bert_score" in metrics:
        from summ_eval.bert_score_metric import BertScoreMetric
        bert_score_metric = BertScoreMetric()
        metrics_dict["bert_score"] = bert_score_metric
        toks_needed.add("space")
    if "mover_score" in metrics:
        from summ_eval.mover_score_metric import MoverScoreMetric
        mover_score_metric = MoverScoreMetric()
        metrics_dict["mover_score"] = mover_score_metric
        toks_needed.add("space")
    if "chrf" in metrics:
        from summ_eval.chrfpp_metric import ChrfppMetric
        metrics_dict["chrf"] = ChrfppMetric()
        toks_needed.add("space")
    if "meteor" in metrics:
        from summ_eval.meteor_metric import MeteorMetric
        metrics_dict["meteor"] = MeteorMetric()
        toks_needed.add("space")
    if "bleu" in metrics:
        from summ_eval.bleu_metric import BleuMetric
        metrics_dict["bleu"] = BleuMetric()
        toks_needed.add("space")
    if "cider" in metrics:
        from summ_eval.cider_metric import CiderMetric
        metrics_dict["cider"] = CiderMetric()
        toks_needed.add("stem")

    if "s3" in metrics:
        from summ_eval.s3_metric import S3Metric
        metrics_dict["s3"] = S3Metric()
        toks_needed.add("stem")
    if "rouge_we" in metrics:
        from summ_eval.rouge_we_metric import RougeWeMetric
        metrics_dict["rouge_we"] = RougeWeMetric()
        toks_needed.add("stem")

    if "stats" in metrics:
        from summ_eval.data_stats_metric import DataStatsMetric
        metrics_dict['stats'] = DataStatsMetric()
        toks_needed.add("spacy")
    if "sms" in metrics:
        from summ_eval.sentence_movers_metric import SentenceMoversMetric
        metrics_dict['sms'] = SentenceMoversMetric()
        toks_needed.add("spacy")
    if "summaqa" in metrics:
        from summ_eval.summa_qa_metric import SummaQAMetric
        metrics_dict['summaqa'] = SummaQAMetric()
        toks_needed.add("spacy")
        toks_needed.add("space")
    if "syntactic" in metrics:
        from summ_eval.syntactic_metric import SyntacticMetric
        metrics_dict["syntactic"] = SyntacticMetric()
        toks_needed.add("space")
    if "supert" in metrics:
        from summ_eval.supert_metric import SupertMetric
        metrics_dict['supert'] = SupertMetric()
        toks_needed.add("space")
    if "blanc" in metrics:
        from summ_eval.blanc_metric import BlancMetric
        metrics_dict['blanc'] = BlancMetric()
        toks_needed.add("space")
    # =====================================





    # =====================================
    # READ INPUT
    print("Reading the input")
    ids = []
    articles = []
    references = []
    summaries = []
    bad_lines = 0
    if args.jsonl_file is not None:
        try:
            with open(args.jsonl_file) as inputf:
                for count, line in enumerate(inputf):
                    try:
                        data = json.loads(line)
                        try:
                            ids.append(data['id'])
                        except:
                            pass
                        if len(data['decoded']) == 0:
                            bad_lines += 1
                            continue
                        summaries.append(data['decoded'])
                        references.append(data['reference'])
                        if "summaqa" in metrics or "stats" in metrics or "supert" in metrics or "blanc" in metrics:
                            try:
                                articles.append(data['text'])
                            except:
                                raise ValueError("You specified summaqa and stats, which" \
                                    "require input articles, but we could not parse the file!")
                    except:
                        bad_lines += 1
        except Exception as e:
            print("Input did not match required format")
            print(e)
            sys.exit()
        print(f"This many bad lines encountered during loading: {bad_lines}")

    if args.summ_file is not None:
        with open(args.summ_file) as inputf:
            summaries = inputf.read().splitlines()
    if args.ref_file is not None:
        with open(args.ref_file) as inputf:
            references = inputf.read().splitlines()
    if "summaqa" in metrics or "stats" in metrics or "supert" in metrics or "blanc" in metrics:
        if args.article_file is None and len(articles) == 0:
            raise ValueError("You specified summaqa and stats, which" \
                 "require input articles, but we could not parse the file!")
        if len(articles) > 0:
            pass
        else:
            with open(args.article_file) as inputf:
                articles = inputf.read().splitlines()
    if len(ids) == 0:
        ids = list(range(0, len(summaries)))
    # =====================================




    # =====================================
    # TOKENIZATION
    print("Preparing the input")
    references_delimited = None
    summaries_delimited = None
    if len(references) > 0:
        if isinstance(references[0], list):
            if "line_delimited" in toks_needed:
                references_delimited = ["\n".join(ref) for ref in references]
            if "space" in toks_needed:
                references_space = [" ".join(ref) for ref in references]
        elif args.eos is not None:
            if "line_delimited" not in toks_needed:
                raise ValueError('You provided a delimiter but are not using a metric which requires one.')
            if args.eos == "\n":
                references_delimited = [ref.split(args.eos) for ref in references]
            else:
                references_delimited = [f"{args.eos}\n".join(ref.split(args.eos)) for ref in references]
        elif "line_delimited" in toks_needed:
            references_delimited = references
        if "space" in toks_needed:
            references_space = references

    if isinstance(summaries[0], list):
        if "line_delimited" in toks_needed:
            summaries_delimited = ["\n".join(summ) for summ in summaries]
        if "space" in toks_needed:
            summaries_space = [" ".join(summ) for summ in summaries]
    elif args.eos is not None:
        if "line_delimited" not in toks_needed:
            raise ValueError('You provided a delimiter but are not using a metric which requires one.')
        if args.eos == "\n":
            summaries_delimited = [ref.split(args.eos) for ref in summaries]
        else:
            summaries_delimited = [f"{args.eos}\n".join(ref.split(args.eos)) for ref in summaries]
    elif "line_delimited" in toks_needed:
        summaries_delimited = summaries
    if "space" in toks_needed:
        summaries_space = summaries

    if "stem" in toks_needed:
        tokenizer = RegexpTokenizer(r'\w+')
        stemmer = SnowballStemmer("english")
        if isinstance(summaries[0], list):
            summaries_stemmed = [[stemmer.stem(word) for word in tokenizer.tokenize(" ".join(summ))] for summ in summaries]
            references_stemmed = [[stemmer.stem(word) for word in tokenizer.tokenize(" ".join(ref))] for ref in references]
        else:
            summaries_stemmed = [[stemmer.stem(word) for word in tokenizer.tokenize(summ)] for summ in summaries]
            references_stemmed = [[stemmer.stem(word) for word in tokenizer.tokenize(ref)] for ref in references]
        summaries_stemmed = [" ".join(summ) for summ in summaries_stemmed]
        references_stemmed = [" ".join(ref) for ref in references_stemmed]

    if "spacy" in toks_needed:
        try:
            nlp = spacy.load('en_core_web_md')
        except OSError:
            print('Downloading the spacy en_core_web_md model\n'
                "(don't worry, this will only happen once)", file=stderr)
            from spacy.cli import download
            download('en_core_web_md')
            nlp = spacy.load('en_core_web_md')
        disable = ["tagger", "textcat"]
        if "summaqa" not in metrics:
            disable.append("ner")
        if isinstance(summaries[0], list):
            summaries_spacy = [nlp(" ".join(text), disable=disable) for text in summaries]
        else:
            summaries_spacy = [nlp(text, disable=disable) for text in summaries]
        if "stats" in metrics:
            summaries_spacy_stats = [[tok.text for tok in summary] for summary in summaries_spacy]
        if "sms" in metrics:
            if isinstance(references[0], list):
                references_spacy = [nlp(" ".join(text), disable=disable) for text in references]
            else:
                references_spacy = [nlp(text, disable=disable) for text in references]
        if "summaqa" in metrics or "stats" in metrics:
            if isinstance(articles[0], list):
                input_spacy = [nlp(" ".join(text), disable=disable) for text in articles]
            else:
                input_spacy = [nlp(text, disable=disable) for text in articles]
            if "stats" in metrics:
                input_spacy_stats = [[tok.text for tok in article] for article in input_spacy]
    if "supert" in metrics or "blanc" in metrics:
        inputs_space = articles
    # =====================================



    # =====================================
    # GET SCORES
    if args.aggregate:
        final_output = dict()
    else:
        final_output = defaultdict(lambda: defaultdict(int))
    #import pdb;pdb.set_trace()
    for metric, metric_cls in metrics_dict.items():
        print(f"Calculating scores for the {metric} metric.")
        try:
            if metric == "rouge":
                output = metric_cls.evaluate_batch(summaries_delimited, references_delimited, aggregate=args.aggregate)
                # only rouge uses this input so we can delete it
                del references_delimited
                del summaries_delimited
            elif metric in ('bert_score', 'mover_score', 'chrf', 'meteor', 'bleu'):
                output = metric_cls.evaluate_batch(summaries_space, references_space, aggregate=args.aggregate)
            elif metric in ('s3', 'rouge_we', 'cider'):
                output = metric_cls.evaluate_batch(summaries_stemmed, references_stemmed, aggregate=args.aggregate)
            elif metric == "sms":
                output = metric_cls.evaluate_batch(summaries_spacy, references_spacy, aggregate=args.aggregate)
            elif metric in ('summaqa', 'stats', 'supert', 'blanc'):
                if metric == "summaqa":
                    output = metric_cls.evaluate_batch(summaries_space, input_spacy, aggregate=args.aggregate)
                elif metric == "stats":
                    output = metric_cls.evaluate_batch(summaries_spacy_stats, input_spacy_stats, aggregate=args.aggregate)
                elif metric in ('supert', 'blanc'):
                    output = metric_cls.evaluate_batch(summaries_space, inputs_space, aggregate=args.aggregate)
            if args.aggregate:
                final_output.update(output)
            else:
                ids = list(range(0, len(ids)))
                for cur_id, cur_output in zip(ids, output):
                    final_output[cur_id].update(cur_output)
        except Exception as e:
            print(e)
            print(f"An error was encountered with the {metric} metric.")
    # =====================================






    # =====================================
    # OUTPUT SCORES
    metrics_str = "_".join(metrics)
    #json_file_end = args.jsonl_file.split("/")[-1]
    json_file_end = args.jsonl_file.replace("/", "_")
    with open(f"outputs/{args.output_file}_{json_file_end}_{metrics_str}.jsonl", "w") as outputf:
        if args.aggregate:
            json.dump(final_output, outputf)
        else:
            for key, value in final_output.items():
                value["id"] = key
                json.dump(value, outputf)
                outputf.write("\n")
    # =====================================

if __name__ == '__main__':
    cli_main()

