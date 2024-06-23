FROM python:3.9.19

# Use a directory for the project
WORKDIR /app

# First, must install the old package from pip, so it will correctly install the dependencies
RUN pip install summ_eval

# Install (and replace) the package with its latest version
COPY evaluation/ /app/SummEval/evaluation
COPY data_processing/ /app/SummEval/data_processing
WORKDIR /app/SummEval/evaluation
RUN pip install -e .

# Set the default ROUGE_HOME
ENV ROUGE_HOME=/app/SummEval/evaluation/summ_eval/ROUGE-1.5.5

# Try to import the module, but it will fail
RUN python -c 'from summ_eval.rouge_metric import RougeMetric' || true
RUN pip install -U git+https://github.com/bheinzerling/pyrouge.git

# Install perl XML::Parser, required by ROUGE metric
RUN apt-get update && apt-get install libxml-parser-perl -y
