# Summarization Repository 

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

### Training and Evaluation Datasets
Generated training data can be found [here](TODO).

Manually annotated validation and test data can be found [here](TODO).

Both generated and manually annotated datasets require pairing with the original CNN/DailyMail articles.

To recreate the datasets follow the instructions:
1. Download CNN Stories and Daily Mail Stories from https://cs.nyu.edu/~kcho/DMQA/
2. Create a `cnndm` directory and unpack downloaded files into the directory
3. Download and unpack FactCC data _(do not rename directory)_
4. Run the `pair_data.py` script to pair the data with original articles

Example call:

`python3 data_pairing/pair_data.py <dir-with-factcc-data> <dir-with-stories>`

**IMPORTANT:** 

All model outputs were obtained from the original authors of the models and shared with their consent.
When using any of the model outputs, please cite the original paper.

### Citation

```
@article{XXXX:2020,
  author    = {XXX, YYYY, ZZZZ},
  title     = {XXXXXXXXXXXXXXX},
  journal   = {XXXX},
  year      = {2020},
}
```


### Get Involved

Please create a GitHub issue if you have any questions, suggestions, requests or bug-reports. 
We welcome PRs!
