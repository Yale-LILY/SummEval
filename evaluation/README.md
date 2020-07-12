# summ_eval
Toolkit for summarization evaluation to unify metrics and promote robust comparison of summarization systems. Includes popular and recent metrics for summarization as well as several machine translation metrics. If you have a metric you want to add, pull requests are welcomed!

# Metrics #
Below are the metrics included in the tookit, followed by the associated paper and code used within the toolkit: 
- ROUGE: [PAPER](https://www.aclweb.org/anthology/W04-1013.pdf) [CODE](https://github.com/bheinzerling/pyrouge/tree/master/pyrouge)
- ROUGE-we: [PAPER](https://www.aclweb.org/anthology/D15-1222.pdf) [CODE](https://github.com/UKPLab/emnlp-ws-2017-s3/tree/master/S3)
- MoverScore: [PAPER](https://www.aclweb.org/anthology/D19-1053.pdf) [CODE](https://github.com/AIPHES/emnlp19-moverscore/)
- BertScore: [PAPER](https://arxiv.org/pdf/1904.09675.pdf) [CODE](https://github.com/Tiiiger/bert_score)
- Sentence Mover's Similarity: [PAPER](https://www.aclweb.org/anthology/P19-1264.pdf) [CODE](https://github.com/eaclark07/sms)
- SummaQA: [PAPER](https://www.aclweb.org/anthology/D19-1320.pdf) [CODE](https://github.com/recitalAI/summa-qa)
- METEOR: [PAPER](https://www.aclweb.org/anthology/W05-0909.pdf) [CODE](https://github.com/Maluuba/nlg-eval/tree/master/nlgeval/pycocoevalcap/meteor)
- Learning to Score System Summaries for Better Content Selection Evaluation: [PAPER](https://www.aclweb.org/anthology/W17-4510/) [CODE](https://github.com/UKPLab/emnlp-ws-2017-s3)
- Measures of extractiveness, misc statistics: (Novel n-grams, Repetition, Length) [PAPER](https://www.aclweb.org/anthology/N18-1065/) [CODE](https://github.com/lil-lab/newsroom)
- Syntactic evaluation: [PAPER](https://www.benjamins.com/catalog/ijcl.15.4.02lu) [CODE](http://www.personal.psu.edu/xxl13/downloads/L2SCA-2016-06-30.tgz)
- Cider: [PAPER](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vedantam_CIDEr_Consensus-Based_Image_2015_CVPR_paper.pdf) [CODE](https://github.com/Maluuba/nlg-eval/tree/master/nlgeval/pycocoevalcap/cider)
- CHRF: [PAPER](https://www.statmt.org/wmt17/pdf/WMT70.pdf) [CODE](https://github.com/m-popovic/chrF)
- BLEU: [PAPER](https://www.aclweb.org/anthology/P02-1040.pdf) [CODE](https://github.com/mjpost/sacreBLEU)




# SETUP # 

First install the summ_eval toolkit:
```bash
git clone https://github.com/Yale-LILY/summarization-repository.git
cd evaluation
pip install -e .
```

To finish the setup, please run and follow the prompts in this script:

```
python setup_finalize.py
```

You can test your installation and get familiar with the library through `tests/`

```
python -m unittest discover
```

# USAGE # 

## Command-line interface
We provide a command-line interface `calc-scores` which makes use of [gin config](https://github.com/google/gin-config) files to set metric parameters. 

### Examples
Run ROUGE on given source and target files and write to `rouge.jsonl`, analogous to [files2rouge](https://github.com/pltrdy/files2rouge). 
```
calc-scores --config-file=examples/basic.config --metrics "rouge" --summ-file summ_eval/1.summ --ref-file summ_eval/1.ref --output-file rouge.jsonl --eos " . " --aggregate True
```

**NOTE**: if you're seeing slow-ish startup time, try commenting out the metrics you're not using in the config; otherwise this will load all modules. 


Run ROUGE and BertScore on a jsonl file which contains "reference" and "summary" keys and write to `output.jsonl`. 
```
calc-scores --config-file=examples/basic.config --metrics "rouge, bert_score" --jsonl-file data.jsonl --output-file rouge_bertscore.jsonl
```

For a full list of options, please run:
```
calc-scores --help
```


## For use in scripts
If you want to use the evaluation metrics as part of other scripts, we have you covered!

```
from summ_eval.rouge_metric import RougeMetric
rouge = RougeMetric()
```

### Evaluate on a batch
```
summaries = ["This is one summary", "This is another summary"]
references = ["This is one reference", "This is another"]

rouge_dict = rouge.evaluate_batch(summaries, references)
```

### Evaluate on a single example
```
rouge_dict = rouge.evaluate_example(summaries[0], references[0])
```


### Evaluate with multiple references
Currently the command-line tool does not use multiple references for simplicity. Each metric has a `supports_multi_ref` property to tell you if it supports multiple references. 

```
print(rouge.supports_multi_ref) # True
multi_references = [["This is ref 1 for summ 1", "This is ref 2 for summ 1"], ["This is ref 1 for summ 2", "This is ref 2 for summ 2"]]
rouge_dict = rouge.evaluate_batch(summaries, multi_references)
```
