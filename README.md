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
