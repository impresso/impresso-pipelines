# Python Package: [impresso-pipelines]

## Overview
This repository contains a Python package designed for efficient and modular processing. Currently, it includes the following subpackages:

- **Language Identification Pipeline**: Detects the language of input text and provides a corresponding probability score.
- **OCR QA Pipeline**: Evaluates the quality of OCR-processed text by calculating a score (0-1) representing the proportion of recognized words in the input text using a language-specific, efficient Bloom filter database.
- **LDA Topic Modeling Pipeline**: Uses topic modelling to assign the most relevant topics to the input text. 



## Installation
To install the package with all subpackages, use:
```bash
pip install "impresso_pipelines[all]"
```

To install individual subpackages without any additional dependencies, use:
```bash
pip install "impresso_pipelines[langident]"         # Language Identification
pip install "impresso_pipelines[ocrqa]"             # OCR QA
pip install "impresso_pipelines[ldatopics]"         # LDA Topics
pip install "impresso_pipelines[newsagencies]"      # News Agencies
pip install "impresso_pipelines[solrnormalization]" # Solrnomalization
```

## Usage
Import and use the subpackages as follows:
```python
from impresso_pipelines.langident import LangIdentPipeline
from impresso_pipelines.ocrqa import OCRQAPipeline
from impresso_pipelines.ldatopics import LDATopicsPipeline
```

## Running the Pipeline examples
For usage examples, refer to the individual README files:

 - [Langident Pipeline](README_langident.md)
 - [OCR QA Pipeline](README_ocrqa.md)
 - [LDA Topics Pipeline](README_ldatopics.md)



Additional examples are available in the documentation notebooks:
 - [langident_pipeline_demo.ipynb](https://github.com/impresso/impresso-datalab-notebooks/tree/main/annotate/langident_pipeline_demo.ipynb)
 - [ocrqa_pipeline_demo.ipynb](https://github.com/impresso/impresso-datalab-notebooks/tree/main/annotate/ocrqa_pipeline_demo.ipynb).

## Future Plans
More Impresso functionality (newsagency detection, named entity recognition and linking) will be added to enhance functionality and broaden use cases.


## About Impresso

### Impresso project

[Impresso - Media Monitoring of the Past](https://impresso-project.ch) is an interdisciplinary research project that aims to develop and consolidate tools for processing and exploring large collections of media archives across modalities, time, languages and national borders. The first project (2017-2021) was funded by the Swiss National Science Foundation under grant No. [CRSII5_173719](http://p3.snf.ch/project-173719) and the second project (2023-2027) by the SNSF under grant No. [CRSII5_213585](https://data.snf.ch/grants/grant/213585) and the Luxembourg National Research Fund under grant No. 17498891.

### Copyright

Copyright (C) 2025 The Impresso team.

### License

This program is provided as open source under the [GNU Affero General Public License](https://github.com/impresso/impresso-pyindexation/blob/master/LICENSE) v3 or later.

---

<p align="center">
  <img src="https://github.com/impresso/impresso.github.io/blob/master/assets/images/3x1--Yellow-Impresso-Black-on-White--transparent.png?raw=true" width="350" alt="Impresso Project Logo"/>
</p>


