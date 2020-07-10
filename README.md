# Summarization Repository 

### Model Outputs

|Model|Paper|Outputs|Type|
|-|-|-|-|
|M0|_Lead-3 Baseline_|[Link](TODO)|Extractive|
|M1|[Neural Document Summarization by Jointly Learning to Score and Select Sentences](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16838/16118)|[Link](TODO)|Extractive|
|M2|[BANDITSUM: Extractive Summarization as a Contextual Bandit](http://aclweb.org/anthology/P18-1061)|[Link](TODO)|Extractive|
|M3|[Neural Latent Extractive Document Summarization](http://aclweb.org/anthology/D18-1088)|[Link](TODO)|Extractive|
|M4|[Ranking Sentences for Extractive Summarization with Reinforcement Learning](https://arxiv.org/pdf/1802.08636.pdf)|[Link](TODO)|Extractive|
|M5|[Learning to Extract Coherent Summary via Deep Reinforcement Learning](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16838/16118)|[Link](TODO)|Extractive|
|M6|[Neural Extractive Text Summarization with Syntactic Compression](https://arxiv.org/abs/1902.00863)|[Link](TODO)|Extractive|
|M7|[STRASS: A Light and Effective Method for Extractive Summarization Based on Sentence Embeddings](https://arxiv.org/abs/1907.07323)|[Link](TODO)|Extractive|
|M8|[Get To The Point: Summarization with Pointer-Generator Networks](http://aclweb.org/anthology/P17-1099)|[Link](TODO)|Abstractive|
|M9|[Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting](https://arxiv.org/pdf/1805.11080.pdf)|[Link](TODO)|Abstractive|
|M10|[Bottom-Up Abstractive Summarization](https://arxiv.org/abs/1808.10792)|[Link](TODO)|Abstractive|
|M11|[Improving Abstraction in Text Summarization](http://aclweb.org/anthology/D18-1207)|[Link](TODO)|Abstractive|
|M12|[A Unified Model for Extractive and Abstractive Summarization using Inconsistency Loss](http://aclweb.org/anthology/P18-1013)|[Link](TODO)|Abstractive|
|M13|[Multi-Reward Reinforced Summarization with Saliency and Entailment](http://aclweb.org/anthology/N18-2102)|[Link](TODO)|Abstractive|
|M14|[Soft Layer-Specific Multi-Task Summarization with Entailment and Question Generation](http://aclweb.org/anthology/P18-1064)|[Link](TODO)|Abstractive|
|M15|[Closed-Book Training to Improve Summarization Encoder Memory](http://aclweb.org/anthology/D18-1440)|[Link](TODO)|Abstractive|
|M16|[An Entity-Driven Framework for Abstractive Summarization](https://arxiv.org/abs/1909.02059)|[Link](TODO)|Abstractive|
|M17|[Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)|[Link](TODO)|Abstractive|
|M18|[Better Rewards Yield Better Summaries: Learning to Summarise Without References](http://arxiv.org/abs/1909.01214)|[Link](TODO)|Abstractive|
|M19|[Text Summarization with Pretrained Encoders](https://arxiv.org/abs/1908.08345)|[Link](TODO)|Abstractive|
|M20|[Fine-Tuning GPT-2 from Human Preferences](https://openai.com/blog/fine-tuning-gpt-2/)|[Link](TODO)|Abstractive|
|M21|[Unified Language Model Pre-training for Natural Language Understanding and Generation](https://arxiv.org/abs/1905.03197)|[Link](TODO)|Abstractive|
|M22|[BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461)|[Link](TODO)|Abstractive|
|M23|[PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization](https://arxiv.org/pdf/1912.08777.pdf)|[Link](TODO)|Abstractive|

### Training and Evaluation Datasets
Generated training data can be found [here](https://storage.googleapis.com/sfr-factcc-data-research/unpaired_generated_data.tar.gz).

Manually annotated validation and test data can be found [here](https://storage.googleapis.com/sfr-factcc-data-research/unpaired_annotated_data.tar.gz).

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
