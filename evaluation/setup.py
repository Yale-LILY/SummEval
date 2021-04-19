
from os import path
from setuptools import setup
from setuptools import find_packages



# read the contents of your README file
# https://packaging.python.org/guides/making-a-pypi-friendly-readme/
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, '..', 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='summ_eval',
      version='0.20',
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
          'pyrouge',
          'bert-score', 
          'moverscore', 
          'gin-config', 
          'pytorch_pretrained_bert', 
          'psutil',
          'six', 
          'wmd',
          'stanza',
          'sacremoses',
          'transformers>=2.2.0',
          'spacy==2.2.0',
          'sacrebleu',
          'pyemd==0.5.1',
          'click', 
          'nltk', 
          'scipy',
          'sklearn',
          'networkx',
          'blanc',
      ],
      entry_points={
          'console_scripts': [
            'calc-scores = summ_eval.calc_scores:cli_main',
          ],
      }
)
