import os
import click
import requests
import bz2
import zipfile, io

dirname, _ = os.path.split(os.path.abspath(__file__))


if click.confirm("Setup ROUGE?", default=True):
    print(f'Please run the following command and add it to your startup script: \n export ROUGE_HOME={os.path.join(dirname, "summ_eval/ROUGE-1.5.5/")}')

if click.confirm("Setup SuPERT?", default=True):
    print(f'Please run the following command and add it to your startup script: \n export PYTHONPATH=$PYTHONPATH:{os.path.join(dirname, "summ_eval/")}')

if click.confirm("Download METEOR jar?", default=True):
    if not os.path.exists(os.path.join(dirname, "summ_eval/meteor-1.5.jar")):
        url = 'https://github.com/Maluuba/nlg-eval/blob/master/nlgeval/pycocoevalcap/meteor/meteor-1.5.jar?raw=true'
        r = requests.get(url)
        with open(os.path.join(dirname, "summ_eval/meteor-1.5.jar"), "wb") as outputf:
            outputf.write(r.content)
    else:
        print("METEOR jar already downloaded!")

if click.confirm("Download embeddings for S3 and ROUGE-WE metrics?", default=True):
    if not os.path.exists(os.path.join(dirname, "summ_eval/embeddings")):
        os.mkdir(os.path.join(dirname, "summ_eval/embeddings"))
    if not os.path.exists(os.path.join(dirname, "summ_eval/embeddings/deps.words")):
        print("Downloading the embeddings; this may take a while")
        url = "http://u.cs.biu.ac.il/~yogo/data/syntemb/deps.words.bz2"
        r = requests.get(url)
        d = bz2.decompress(r.content)
        with open(os.path.join(dirname, "summ_eval/embeddings/deps.words"), "wb") as outputf:
            outputf.write(d)
    else:
        print("Embeddings already downloaded!")

if click.confirm("Download CoreNLP for Syntactic metric?", default=True):
    if not os.path.exists(os.path.join(dirname, "summ_eval/stanford-corenlp-full-2018-10-05")):
        url = 'http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip'
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(os.path.join(dirname, "summ_eval/"))
        print(f'Please run the following command and add it to your startup script: \n export CORENLP_HOME={os.path.join(dirname, "summ_eval/stanford-corenlp-full-2018-10-05/")}')
    else:
        print("CORENLP already downloaded!")
