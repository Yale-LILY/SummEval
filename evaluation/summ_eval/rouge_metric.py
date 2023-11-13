# pylint: disable=W0102,C0301,W1401,C0303,C0103
import os
from pathlib import Path
import tempfile
import shutil
import gin
import logging
import subprocess

from summ_eval import logger
from summ_eval.metric import Metric
from summ_eval.test_util import rouge_empty


logger = logger.getChild(__name__)

ROUGE_HOME = os.environ['ROUGE_HOME'] or Path(__file__).parent / "ROUGE-1.5.5"
if "ROUGE_HOME" not in os.environ:
    logger.info("ROUGE_HOME not set, using default location %s", ROUGE_HOME)

if not os.path.exists(ROUGE_HOME):
    logger.info("ROUGE_HOME=%s not a directory.", ROUGE_HOME)
    try:
        logger.info("Installing rouge Perl script to {ROUGE_HOME} - this will take a few seconds")
        subprocess.run(["curl", "-L", "https://github.com/Yale-LILY/SummEval/tarball/7e4330d", "-o", "project.tar.gz", "-s"])
        subprocess.run(["tar", "-xzf", "project.tar.gz"])
        subprocess.run(["mv", "Yale-LILY-SummEval-7e4330d/evaluation/summ_eval/ROUGE-1.5.5/", ROUGE_HOME])
        subprocess.run(["rm", "project.tar.gz"])
        subprocess.run(["rm", "-rf", "Yale-LILY-SummEval-7e4330d/"])
    except subprocess.CalledProcessError as err:
        logger.error("Failed to install the rouge Perl script; please install manually and set the ROUGE_HOME environment variable.")
        raise err

try:
    from pyrouge import Rouge155
except ImportError as err:
    logger.error("pyrouge not installed; please install with `pip install -U  git+https://github.com/bheinzerling/pyrouge.git`.")
    raise err


@gin.configurable
class RougeMetric(Metric):
    def __init__(self, rouge_dir=ROUGE_HOME, rouge_args=None, verbose=False):
        """
        ROUGE metric
        Makes use of pyrouge: https://github.com/bheinzerling/pyrouge

        Args:
                :param rouge_dir: directory of ROUGE-1.5.5/, by default uses environment's ROUGE_HOME variable
                :param rouge_args: arguments for ROUGE calculation; if None, defaults to "-c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -a -m"; a string of parameters. Please see ROUGE-1.5.5 README (e.g. https://github.com/andersjo/pyrouge/tree/master/tools/ROUGE-1.5.5) for a list of possible parameters
                :param verbose: whether to log data preparation or just output

        """
        # 
        log_level = logging.ERROR if not verbose else None
        try:
            self.r = Rouge155(rouge_dir=rouge_dir, rouge_args=rouge_args, log_level=log_level)
        except:
            logger.error(f'Please run this command: \n pip install -U  git+https://github.com/bheinzerling/pyrouge.git')
            exit()
        self.rouge_args = rouge_args

    def evaluate_example(self, summary, reference):
        if not isinstance(reference, list):
            reference = [reference]
        if len(summary) == 0:
            return rouge_empty
        self.r.system_dir = tempfile.mkdtemp()
        self.r.model_dir = tempfile.mkdtemp()
        self.r.system_filename_pattern = 'system.(\d+).txt'
        self.r.model_filename_pattern = 'model.[A-Z].#ID#.txt'
        with open(os.path.join(self.r.system_dir, "system.0.txt"), "w") as outputf:
            outputf.write(summary)
        for ref_idx, ref in enumerate(reference):
            with open(os.path.join(self.r.model_dir, f"model.{chr(ord('A') + ref_idx)}.0.txt"), "w") as outputf:
                outputf.write(ref)
        if self.rouge_args is not None:
            output = self.r.convert_and_evaluate(rouge_args=f"-e {self.r.data_dir} " + self.r.args)
        else:
            output = self.r.convert_and_evaluate()
        output_dict = self.r.output_to_dict(output)
        shutil.rmtree(self.r.system_dir)
        shutil.rmtree(self.r.model_dir)
        return {"rouge": output_dict}

    def evaluate_batch(self, summaries, references, aggregate=True):
        if not aggregate:
            results = [self.evaluate_example(summ, ref) for ref, summ in zip(references, summaries)]
            return results
        self.r.system_dir = tempfile.mkdtemp()
        self.r.model_dir = tempfile.mkdtemp()
        self.r.system_filename_pattern = 'system.(\d+).txt'
        self.r.model_filename_pattern = 'model.[A-Z].#ID#.txt'
        for idx, (refs, summ) in enumerate(zip(references, summaries)):
            with open(os.path.join(self.r.system_dir, f"system.{idx}.txt"), "w") as outputf:
                outputf.write(summ)
            if not isinstance(refs, list):
                refs = [refs]
            for ref_idx, ref in enumerate(refs):
                with open(os.path.join(self.r.model_dir, f"model.{chr(ord('A') + ref_idx)}.{idx}.txt"), "w") as outputf:
                    outputf.write(ref)
        if self.rouge_args is not None:
            output = self.r.convert_and_evaluate(rouge_args=f"-e {self.r.data_dir} " + self.r.args)
        else:
            output = self.r.convert_and_evaluate()
        output_dict = self.r.output_to_dict(output)
        shutil.rmtree(self.r.system_dir)
        shutil.rmtree(self.r.model_dir)
        return {"rouge": output_dict}

    @property
    def supports_multi_ref(self):
        return True
