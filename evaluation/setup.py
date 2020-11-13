
from setuptools import setup
from setuptools import find_packages


setup(name='summ_eval',
      version='0.1',
      description='Toolkit for summarization evaluation', 
      url='https://github.com/Alex-Fabbri/summ_eval.git', 
      author='Alex Fabbri, Wojciech Kryściński', 
      author_email='alexander.fabbri@yale.edu, wojciech.kryscinski@salesforce.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'pyrouge @ git+git://github.com/bheinzerling/pyrouge.git',
          'bert-score', 
          'moverscore', 
          'gin-config', 
          'pytorch_pretrained_bert', 
          'en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz',
          'en_core_web_md @ https://github.com/explosion/spacy-models/releases/download/en_core_web_md-2.2.0/en_core_web_md-2.2.0.tar.gz',
          'psutil',
          'six', 
          'wmd',
          'stanza',
          'transformers>=2.2.0',
          'spacy==2.2.0',
          'sacrebleu',
          'pyemd==0.5.1',
          'click', 
          'nltk', 
          'scipy',
          'blanc',
      ],
      entry_points={
          'console_scripts': [
            'calc-scores = summ_eval.calc_scores:cli_main',
          ],
      }
)
