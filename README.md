# Summarization Repository 
This repo was initially released with the [SummEval](#citation) paper. It
contains the python package [`summ-eval`](https://pypi.org/project/summ-eval/),
a toolkit for summarization evaluation, as well as the [dataset](#dataset) used
in the original publication.


## Table of Contents

1. [Evaluation Toolkit](#evaluation-toolkit)
2. [Dataset](#dataset)
3. [Contribute](#contribute)
4. [Updates](#updates)
5. [Citation](#citation)


## `summ-eval`: An Evaluation Toolkit

This toolkit for summarization evaluation is provided to unify metrics and promote robust comparison of summarization
systems. The toolkit contains popular and recent metrics for summarization as well as several machine translation metrics.
We invite contributions of additional metrics, please [contribute](#contribute).

The metrics currently included in the tookit, with the associated paper and code used within the toolkit, and where they
can be found within the package.
|Metric|Paper|Code|
|-|-|-|
|ROUGE|[ROUGE: A Package for Automatic Evaluation of Summaries](https://www.aclweb.org/anthology/W04-1013.pdf)|[Link](https://github.com/bheinzerling/pyrouge/tree/master/pyrouge)|
|ROUGE-we|[Better Summarization Evaluation with Word Embeddings for ROUGE](https://www.aclweb.org/anthology/D15-1222.pdf)|[Link](https://github.com/UKPLab/emnlp-ws-2017-s3/blob/master/S3/ROUGE.py#L152)|
|MoverScore|[MoverScore: Text Generation Evaluating with Contextualized Embeddings and Earth Mover Distance](https://www.aclweb.org/anthology/D19-1053.pdf)|[Link](https://github.com/AIPHES/emnlp19-moverscore/)|
|BertScore|[BertScore: Evaluating Text Generation with BERT](https://arxiv.org/pdf/1904.09675.pdf)|[Link](https://github.com/Tiiiger/bert_score)|
|Sentence Mover's Similarity|[Sentence Mover’s Similarity: Automatic Evaluation for Multi-Sentence Texts](https://www.aclweb.org/anthology/P19-1264.pdf)|[Link](https://github.com/eaclark07/sms)|
|SummaQA|[Answers Unite! Unsupervised Metrics for Reinforced Summarization Models](https://www.aclweb.org/anthology/D19-1320.pdf)|[Link](https://github.com/recitalAI/summa-qa)|
|BLANC|[Fill in the BLANC: Human-free quality estimation of document summaries](https://arxiv.org/pdf/2002.09836.pdf)|[Link](https://github.com/PrimerAI/blanc)|
|SUPERT|[SUPERT: Towards New Frontiers in Unsupervised Evaluation Metrics for Multi-Document Summarization](https://www.aclweb.org/anthology/2020.acl-main.124.pdf)|[Link](https://github.com/yg211/acl20-ref-free-eval)|
|METEOR|[METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments ](https://www.aclweb.org/anthology/W05-0909.pdf)|[Link](https://github.com/Maluuba/nlg-eval/tree/master/nlgeval/pycocoevalcap/meteor)|
|S<sup>3</sup>|[Learning to Score System Summaries for Better Content Selection Evaluation](https://www.aclweb.org/anthology/W17-4510/)|[Link](https://github.com/UKPLab/emnlp-ws-2017-s3)|
|Misc. statistics<br/>(extractiveness, novel n-grams, repetition, length)|[Newsroom: A Dataset of 1.3 Million Summaries with Diverse Extractive Strategies](https://www.aclweb.org/anthology/N18-1065/)| [Link](https://github.com/lil-lab/newsroom)|
|Syntactic Evaluation|[Automatic Analysis of Syntactic Complexity in Second Language writing](https://www.benjamins.com/catalog/ijcl.15.4.02lu)|[Link](http://www.personal.psu.edu/xxl13/downloads/L2SCA-2016-06-30.tgz)|
|CIDer|[CIDEr: Consensus-based Image Description Evaluation](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vedantam_CIDEr_Consensus-Based_Image_2015_CVPR_paper.pdf)|[Link](https://github.com/Maluuba/nlg-eval/tree/master/nlgeval/pycocoevalcap/cider)|
|CHRF|[CHRF++: words helping character n-grams](https://www.statmt.org/wmt17/pdf/WMT70.pdf)|[Link](https://github.com/m-popovic/chrF)|
|BLEU|[BLEU: a Method for Automatic Evaluation of Machine Translation](https://www.aclweb.org/anthology/P02-1040.pdf)|[Link](https://github.com/mjpost/sacreBLEU)|


## Setup

You can install the base package `summ_eval` via pip:
```bash
pip install summ-eval
```

You can also install summ_eval from source:

```
git clone https://github.com/Yale-LILY/SummEval.git
cd evaluation
pip install -e .
```

Additional setup is required for some metrics. Many of these should run when first
imported. If you run into issues, please create an issue on this repository.

### Rouge 

The original Perl rouge implementation is used to ensure consistency with previous work.
This requires the perl scripts to be downloaded from this repository and the location
of the download to be set as `ROUGE_HOME`. These steps should happen when you
first import the `rouge_metric` sub-module.

Additionally, the XML:Parser Perl package is required. This can be installed with
```bash
apt-get install libxml-parser-perl
```
in ubuntu systems, at least.

Finally, the `pyrouge` python package is required. This can be installed manually
or with the `summ-eval[rouge]` extra.


### S3; Sentence Mover's Similarity and Supert
These metrics all require NLTK's `stopwords` corpus. This can be installed with
```bash
python -m nltk.downloader stopwords
```

### Supert
Supert requires NLTK's `punkt` tokenizer. This can be installed with
```bash
python -m nltk.downloader tokenizer
```

Additionally, this metric the `sentence_transformers` folder to be findable by importlib.
To do this, `./evaluation/summ_eval/` must be added to the python path.
This can be done either manually with `export PYTHONPATH=$PYTHONPATH:./evaluation/summ_eval/` or
will be done automatically when `summ_eval.supert_metric` is imported.

### Meteor
This implementation is in java so requires the java run time environment.
This can be installed with
```bash
apt-get install default-jre
```

### Blanc; Mover score
As currently implemented, these metrics require CUDA acceleration.
This will require cuda to be installed, which can be confirmed by successfully running `nvidia-smi` and having
the appropriate GPU reported.

Installation instructions are provided by [NVidia](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html).

## Usage

### As a package
Each metric can be imported and instantiated as a class.

```python
from summ_eval.selected_metric import SelectedMetric
metric = SelectedMetric()
```

With the metric instantiated, you can evaluate a batch of summaries and references, or a single example.
```python
batch_summaries = ["This is one summary", "This is another summary"]
batch_references = ["This is one reference", "This is another"]

batch_result_dict = metric.evaluate_batch(batch_summaries, batch_references)
single_result_dict = metric.evaluate_example(summaries[0], references[0])
```


Each metric has a `supports_multi_ref` property to indicate whether it supports multiple references.
If it does, then a list of references can be passed to `evaluate` and `evaluate_batch` for each
summary.

```python
assert metric.supports_multi_ref
multi_references = [
    ["This is ref 1 for summ 1", "This is ref 2 for summ 1"],
    ["This is ref 1 for summ 2", "This is ref 2 for summ 2"]
]
batch_result_dict = metric.evaluate_batch(batch_summaries, multi_references)
single_result_dict = metric.evaluate_example(summaries[0], multi_references[0])
```

### As a CLI
If you want to use the evaluation metrics from a shell, we have you covered!
We provide a command-line interface, `calc-scores`. Use
```bash
calc-scores --help
```
for a full list of options.

TLDR: you will need to specify
* `--summ-file`: a file containing summaries, one per line
* `--ref-file`: a file containing references, one per line
* `--output-file`: a file to write the results to
* `--metrics`: a comma-separated list of metrics to use

For example, to run the ROUGE metric and write the results to `rouge.jsonl`, analogous to [files2rouge](https://github.com/pltrdy/files2rouge). 
```bash
calc-scores --config-file=examples/basic.config --metrics "rouge" --summ-file summ_eval/1.summ --ref-file summ_eval/1.ref --output-file rouge.jsonl --eos " . " --aggregate True
```

Other options exist for file I/O convenience as well as configuration.

To run ROUGE and BertScore on a `.jsonl` file which contains `reference` and `decoded` keys (i.e., system output) and write to `output.jsonl`.
```bash
calc-scores --config-file=examples/basic.config --metrics "rouge, bert_score" --jsonl-file data.jsonl --output-file rouge_bertscore.jsonl
```

The CLI makes use of [gin config](https://github.com/google/gin-config) files to more conveniently set metric parameters.

**NOTE**: if you're seeing slow-ish startup time, try commenting out the metrics you're not using in the config; otherwise this will load all modules. 

**NOTE** currently the command-line tool does not support multiple references for simplicity of the interface. This will
have to be completed with a python script.


### Metric naming
The metrics can be imported from
|Metric|Import|CLI string
|-|-|-|
|ROUGE|`from summ_eval.rouge_metric import RougeMetric`|`rouge`|
|ROUGE-we|`from summ_eval.rouge_metric import RougeWeMetric`|`rouge_we`|
|MoverScore|`from summ_eval.mover_score import MoverScoreMetric`|`mover_score`|
|BertScore|`from summ_eval.bert_score import BertScoreMetric`|`bert_score`|
|Sentence Mover's Similarity|`from summ_eval.sms_metric import SmsMetric`|`sms`|
|SummaQA|`from summ_eval.summa_qa_metric import SummaQaMetric`|`summa_qa`|
|BLANC|`from summ_eval.blanc_metric import BlancMetric`|`blanc`|
|SUPERT|`from summ_eval.supert_metric import SupertMetric`|`supert`|
|METEOR|`from summ_eval.meteor_metric import MeteorMetric`|`meteor`|
|S<sup>3</sup>|`from summ_eval.s3_metric import S3Metric`|`s3`|
|Misc. statistics<br/>(extractiveness, novel n-grams, repetition, length)|`from summ_eval.misc_stats import MiscStatsMetric`|`misc_stats`|
|Syntactic Evaluation|`from summ_eval.syntactic_metric import SyntacticMetric`|`syntactic`|
|CIDer|`from summ_eval.cider_metric import CiderMetric`|`cider`|
|CHRF|`from summ_eval.chrf_metric import ChrfMetric`|`chrf`|
|BLEU|`from summ_eval.bleu_metric import BleuMetric`|`bleu`|
## SummEval: a Dataset
As part of this release, we share summaries generated by recent summarization model trained on the CNN/DailyMail dataset [here](#model-outputs).</br>
We also share human annotations, collected from both crowdsource workers and experts [here](#human-annotations).

Both datasets are shared WITHOUT the source articles that were used to generate the summaries. <br/>
To recreate the full dataset please follow the instructions listed [here](#data-preparation). 

### Model Outputs

|Model|Paper|Outputs|Type|
|-|-|-|-|
|M0|_Lead-3 Baseline_|[Link](https://storage.googleapis.com/sfr-summarization-repo-research/M0.tar.gz)|Extractive|
|M1|[Neural Document Summarization by Jointly Learning to Score and Select Sentences](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16838/16118)|[Link](https://storage.googleapis.com/sfr-summarization-repo-research/M1.tar.gz)|Extractive|
|M2|[BANDITSUM: Extractive Summarization as a Contextual Bandit](http://aclweb.org/anthology/P18-1061)|[Link](https://storage.googleapis.com/sfr-summarization-repo-research/M2.tar.gz)|Extractive|
|M3|[Neural Latent Extractive Document Summarization](http://aclweb.org/anthology/D18-1088)|[Link](https://storage.googleapis.com/sfr-summarization-repo-research/M3.tar.gz)|Extractive|
|M4|[Ranking Sentences for Extractive Summarization with Reinforcement Learning](https://www.aclweb.org/anthology/N18-1158/)|[Link](https://storage.googleapis.com/sfr-summarization-repo-research/M4.tar.gz)|Extractive|
|M5|[Learning to Extract Coherent Summary via Deep Reinforcement Learning](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16838/16118)|[Link](https://storage.googleapis.com/sfr-summarization-repo-research/M5.tar.gz)|Extractive|
|M6|[Neural Extractive Text Summarization with Syntactic Compression](https://www.aclweb.org/anthology/D19-1324/)|[Link](https://storage.googleapis.com/sfr-summarization-repo-research/M6.tar.gz)|Extractive|
|M7|[STRASS: A Light and Effective Method for Extractive Summarization Based on Sentence Embeddings](https://www.aclweb.org/anthology/P19-2034/)|[Link](https://storage.googleapis.com/sfr-summarization-repo-research/M7.tar.gz)|Extractive|
|M8|[Get To The Point: Summarization with Pointer-Generator Networks](http://aclweb.org/anthology/P17-1099)|[Link](https://storage.googleapis.com/sfr-summarization-repo-research/M8.tar.gz)|Abstractive|
|M9|[Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting](https://www.aclweb.org/anthology/P18-1063)|[Link](https://storage.googleapis.com/sfr-summarization-repo-research/M9.tar.gz)|Abstractive|
|M10|[Bottom-Up Abstractive Summarization](https://www.aclweb.org/anthology/D18-1443/)|[Link](https://storage.googleapis.com/sfr-summarization-repo-research/M10.tar.gz)|Abstractive|
|M11|[Improving Abstraction in Text Summarization](http://aclweb.org/anthology/D18-1207)|[Link](https://storage.googleapis.com/sfr-summarization-repo-research/M11.tar.gz)|Abstractive|
|M12|[A Unified Model for Extractive and Abstractive Summarization using Inconsistency Loss](http://aclweb.org/anthology/P18-1013)|[Link](https://storage.googleapis.com/sfr-summarization-repo-research/M12.tar.gz)|Abstractive|
|M13|[Multi-Reward Reinforced Summarization with Saliency and Entailment](http://aclweb.org/anthology/N18-2102)|[Link](https://storage.googleapis.com/sfr-summarization-repo-research/M13.tar.gz)|Abstractive|
|M14|[Soft Layer-Specific Multi-Task Summarization with Entailment and Question Generation](http://aclweb.org/anthology/P18-1064)|[Link](https://storage.googleapis.com/sfr-summarization-repo-research/M14.tar.gz)|Abstractive|
|M15|[Closed-Book Training to Improve Summarization Encoder Memory](http://aclweb.org/anthology/D18-1440)|[Link](https://storage.googleapis.com/sfr-summarization-repo-research/M15.tar.gz)|Abstractive|
|M16|[An Entity-Driven Framework for Abstractive Summarization](https://www.aclweb.org/anthology/D19-1323/)|[Link](https://storage.googleapis.com/sfr-summarization-repo-research/M16.tar.gz)|Abstractive|
|M17|[Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)|[Link](https://storage.googleapis.com/sfr-summarization-repo-research/M17.tar.gz)|Abstractive|
|M18|[Better Rewards Yield Better Summaries: Learning to Summarise Without References](https://www.aclweb.org/anthology/D19-1307)|[Link](https://storage.googleapis.com/sfr-summarization-repo-research/M18.tar.gz)|Abstractive|
|M19|[Text Summarization with Pretrained Encoders](https://www.aclweb.org/anthology/D19-1387)|[Link](https://storage.googleapis.com/sfr-summarization-repo-research/M19.tar.gz)|Abstractive|
|M20|[Fine-Tuning GPT-2 from Human Preferences](https://openai.com/blog/fine-tuning-gpt-2/)|[Link](https://storage.googleapis.com/sfr-summarization-repo-research/M20.tar.gz)|Abstractive|
|M21|[Unified Language Model Pre-training for Natural Language Understanding and Generation](https://arxiv.org/abs/1905.03197)|[Link](https://storage.googleapis.com/sfr-summarization-repo-research/M21.tar.gz)|Abstractive|
|M22|[BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://www.aclweb.org/anthology/2020.acl-main.703)|[Link](https://storage.googleapis.com/sfr-summarization-repo-research/M22.tar.gz)|Abstractive|
|M23|[PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization](https://arxiv.org/pdf/1912.08777.pdf)|[Link](https://storage.googleapis.com/sfr-summarization-repo-research/M23.tar.gz)|Abstractive|

**IMPORTANT:** 

All model outputs were obtained from the original authors of the models and shared with their consent.<br/>
When using any of the model outputs, please also _cite the original paper_.


### Human annotations

Human annotations of model generated summaries can be found [here](https://storage.googleapis.com/sfr-summarization-repo-research/model_annotations.aligned.jsonl).

The annotations include summaries generated by 16 models from 100 source news articles (1600 examples in total). <br/>
Each of the summaries was annotated by 5 indepedent crowdsource workers and 3 independent experts (8 annotations in total). <br/>
Summaries were evaluated across 4 dimensions: _coherence_, _consistency_, _fluency_, _relevance_. <br/>
Each source news article comes with the original reference from the CNN/DailyMail dataset and 10 additional crowdsources reference summaries.

### Data preparation

Both model generated outputs and human annotated data require pairing with the original CNN/DailyMail articles.

To recreate the datasets follow the instructions:
1. Download CNN Stories and Daily Mail Stories from https://cs.nyu.edu/~kcho/DMQA/
2. Create a `cnndm` directory and unpack downloaded files into the directory
3. Download and unpack model outputs or human annotations.
4. Run the `pair_data.py` script to pair the data with original articles

Example call for _model outputs_:

`python3 data_processing/pair_data.py --model_outputs <file-with-data-annotations> --story_files <dir-with-stories>`

Example call for _human annotations_:

`python3 data_processing/pair_data.py --data_annotations <file-with-data-annotations> --story_files <dir-with-stories>`



## Contribute

Please create a GitHub issue if you have any questions, suggestions, requests or bug-reports. 
We welcome PRs!


### Setup
The project uses [poetry](https://python-poetry.org/) for dependency management so
after installing poetry, installing the project dependencies is as simple as
```bash
poetry install --all-extras
```

### Testing

You can test your installation (assuming you're in the `./summ_eval` folder) and get familiar with the library through `tests/`

```
poetry run python -m unittest discover
```

## Updates
_04/19/2020_ - Updated the [human annotation file](https://drive.google.com/file/d/1d2Iaz3jNraURP1i7CfTqPIj8REZMJ3tS/view?usp=sharing) to include all models from paper and metric scores.<br/>
_04/19/2020_ - SummEval is now pip-installable! Check out the [pypi page](https://pypi.org/project/summ-eval/).<br/>
_04/09/2020_ - Please see [this comment](https://github.com/Yale-LILY/SummEval/issues/13#issuecomment-812918298) with code for computing system-level metric correlations!  <br/>
_11/12/2020_ - Added the reference-less BLANC and SUPERT metrics! <br/>
_7/16/2020_ - Initial commit! :) 


## Citation
[SummEval: Re-evaluating Summarization Evaluation](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00373/100686)

Authors: [Alex Fabbri*](http://alex-fabbri.github.io/), [Wojciech Kryściński*](https://twitter.com/iam_wkr), [Bryan McCann](https://bmccann.github.io/), [Caiming Xiong](http://cmxiong.com/), [Richard Socher](https://www.socher.org/), and [Dragomir Radev](http://www.cs.yale.edu/homes/radev/)<br/>
<sub><sup>\* - Equal contributions from authors</sup></sub>

The SummEval project is a collaboration between [Yale LILY Lab](https://yale-lily.github.io/) and [Salesforce Research](https://einstein.ai/). <br/><br/>
<p align="center">
<img src="https://raw.githubusercontent.com/Yale-LILY/SummEval/master/assets/logo-lily.png" height="100" alt="LILY Logo" style="padding-right:160">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://raw.githubusercontent.com/Yale-LILY/SummEval/master/assets/logo-salesforce.svg" height="100" alt="Salesforce Logo"> 
</p>

```
@article{fabbri2020summeval,
  title={SummEval: Re-evaluating Summarization Evaluation},
  author={Fabbri, Alexander R and Kry{\'s}ci{\'n}ski, Wojciech and McCann, Bryan and Xiong, Caiming and Socher, Richard and Radev, Dragomir},
  journal={arXiv preprint arXiv:2007.12626},
  year={2020}
}
```
