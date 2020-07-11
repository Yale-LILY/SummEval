# pylint: disable=C0103,W0702
# Python wrapper for METEOR implementation, by Xinlei Chen --
# https://github.com/Maluuba/nlg-eval/blob/master/nlgeval/pycocoevalcap/meteor/meteor.py
# Acknowledge Michael Denkowski for the generous discussion and help
from __future__ import division

import atexit
import logging
import os
import re
import subprocess
import sys
import threading
import psutil
from summ_eval.metric import Metric

dirname = os.path.dirname(__file__)

def enc(s):
    return s.encode('utf-8')

def dec(s):
    return s.decode('utf-8')

class MeteorMetric(Metric):
    def __init__(self, METEOR_JAR=os.path.join(dirname, 'meteor-1.5.jar')):
        """
        METEOR metric
            Taken from nlg-eval:
                # Python wrapper for METEOR implementation, by Xinlei Chen --
                # https://github.com/Maluuba/nlg-eval/blob/master/nlgeval/pycocoevalcap/meteor/meteor.py
                # Acknowledge Michael Denkowski for the generous discussion and help

            NOTE: assumes the presence of data/paraphrase-en.gz
            :param METEOR_JAR: location of METEOR jar
        """
        self.METEOR_JAR = METEOR_JAR
        # Used to guarantee thread safety
        self.lock = threading.Lock()
        mem = '2G'
        mem_available_G = psutil.virtual_memory().available / 1E9
        if mem_available_G < 2:
            logging.warning("There is less than 2GB of available memory.\n"
                            "Will try with limiting Meteor to 1GB of memory but this might cause issues.\n"
                            "If you have problems using Meteor, "
                            "then you can try to lower the `mem` variable in meteor.py")
            mem = '1G'

        meteor_cmd = ['java', '-jar', '-Xmx{}'.format(mem), self.METEOR_JAR,
                      '-', '-', '-stdio', '-l', 'en', '-norm']
        env = os.environ.copy()
        env['LC_ALL'] = "C"
        self.meteor_p = subprocess.Popen(meteor_cmd,
                                         cwd=os.path.dirname(os.path.abspath(__file__)),
                                         env=env,
                                         stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE)

        atexit.register(self.close)

    def close(self):
        with self.lock:
            if self.meteor_p:
                self.meteor_p.kill()
                self.meteor_p.wait()
                self.meteor_p = None
        # if the user calls close() manually, remove the
        # reference from atexit so the object can be garbage-collected.
        if atexit is not None and atexit.unregister is not None:
            atexit.unregister(self.close)

    def _stat(self, hypothesis_str, reference_list):
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||', '')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        score_line = re.sub(r'\s+', ' ', score_line)
        self.meteor_p.stdin.write(enc(score_line))
        self.meteor_p.stdin.write(enc('\n'))
        self.meteor_p.stdin.flush()
        return dec(self.meteor_p.stdout.readline()).strip()

    def evaluate_example(self, summary, reference):
        scores = []
        eval_line = 'EVAL'
        with self.lock:
            if not isinstance(reference, list):
                reference = [reference]
            stat = self._stat(summary, reference)
            eval_line += ' ||| {}'.format(stat)
            self.meteor_p.stdin.write(enc('{}\n'.format(eval_line)))
            self.meteor_p.stdin.flush()
            v = self.meteor_p.stdout.readline()
            try:
                scores.append(float(dec(v.strip())))
            except:
                sys.stderr.write("Error handling value: {}\n".format(v))
                sys.stderr.write("Decoded value: {}\n".format(dec(v.strip())))
                sys.stderr.write("eval_line: {}\n".format(eval_line))
                raise
            score = float(dec(self.meteor_p.stdout.readline()).strip())
        score_dict = {"meteor" : score}
        return score_dict

    def evaluate_batch(self, summaries, references, aggregate=True):
        scores = []
        eval_line = 'EVAL'
        with self.lock:
            for ref, summ in zip(references, summaries):
                if not isinstance(ref, list):
                    ref = [ref]
                stat = self._stat(summ, ref)
                eval_line += ' ||| {}'.format(stat)
            self.meteor_p.stdin.write(enc('{}\n'.format(eval_line)))
            self.meteor_p.stdin.flush()
            for _ in range(len(summaries)):
                v = self.meteor_p.stdout.readline()
                try:
                    scores.append(float(dec(v.strip())))
                except:
                    sys.stderr.write("Error handling value: {}\n".format(v))
                    sys.stderr.write("Decoded value: {}\n".format(dec(v.strip())))
                    sys.stderr.write("eval_line: {}\n".format(eval_line))
            score = float(dec(self.meteor_p.stdout.readline()).strip())
        if aggregate:
            score_dict = {"meteor" : score}
        else:
            score_dict = [{"meteor" : cur_score} for cur_score in scores]
        return score_dict

    def __del__(self):
        self.close()

    @property
    def supports_multi_ref(self):
        return True
