
from os import path
from setuptools import setup
from setuptools import find_packages
import urllib.request
import pip
pip.main(['install', 'cython'])


f = urllib.request.urlopen("https://raw.githubusercontent.com/Yale-LILY/SummEval/master/README.md")
long_description = f.read()

setup(name='summ_eval',
      version='0.80',
      description='Toolkit for summarization evaluation', 
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/Yale-LILY/SummEval', 
      author='Alex Fabbri, Wojciech Kryściński', 
      author_email='alexander.fabbri@yale.edu, wojciech.kryscinski@salesforce.com',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=[
          'bert-score', 
          'moverscore @ git+https://github.com/AIPHES/emnlp19-moverscore.git', 
          'gin-config', 
          'pytorch_pretrained_bert', 
          'psutil',
          'six', 
          'wmd',
          'stanza',
          'sacremoses',
          'transformers>=2.2.0',
          'spacy>=2.2.0',
          'sacrebleu',
          'pyemd==0.5.1',
          'click', 
          'nltk', 
          'scipy',
          'scikit-learn==0.21.3',
          'networkx',
          'blanc',
      ],
      entry_points={
          'console_scripts': [
            'calc-scores = summ_eval.calc_scores:cli_main',
          ],
      }
)
