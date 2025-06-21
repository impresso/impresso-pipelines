# Python Package `impresso-pipelines`

## Overview
This repository contains a Python package designed for modular and efficient text processing workflows. Currently, it includes the following subpackages:

- **Language Identification Pipeline**: Identifies the language of input text and returns a probability score.
- **OCR QA Pipeline**: Assesses the quality of OCR text by estimating the proportion of recognized words (0–1), using efficient language-specific Bloom filters.
- **LDA Topic Modeling Pipeline**: Soft clustering of input texts using LDA-based topic modeling. 
- **News Agencies Pipeline**: Extracts and ranks news agency entities from text, providing relevance scores and optional links to Wikidata.
- **Lucene/Solr ormalization Pipeline**: Replicates Solr’s language-specific text normalization to clarify how input text is tokenized and indexed in impresso.


## Installation
To install the full package with all submodules:
```bash
pip install "impresso-pipelines[all]"
```
The `[all]` extra installs all dependencies required for each component.


To install individual modules without unnecessary dependencies, use:
```bash
pip install "impresso-pipelines[langident]"         # Language Identification
pip install "impresso-pipelines[ocrqa]"             # OCR QA
pip install "impresso-pipelines[ldatopics]"         # LDA Topics
pip install "impresso-pipelines[newsagencies]"      # News Agencies
pip install "impresso-pipelines[solrnormalization]" # Solr text normalization
```

## Usage
Each pipeline is instantiated from a corresponding class. 
```python
from impresso_pipelines.langident import LangIdentPipeline
from impresso_pipelines.ocrqa import OCRQAPipeline
from impresso_pipelines.ldatopics import LDATopicsPipeline
from impresso_pipelines.newsagencies import NewsAgenciesPipeline
from impresso_pipelines.solrnormalization import SolrNormalizationPipeline
```

## Pipeline Examples
For usage examples, refer to the individual README files:

 - [Langident Pipeline](README_langident.md)
 - [OCR QA Pipeline](README_ocrqa.md)
 - [LDA Topics Pipeline](README_ldatopics.md)
 - [News Agencies Pipeline](README_newsagencies.md)
 - [Solr normalization Pipeline](README_solrnormalization.md)



See also the interactive notebooks for further examples:
 - [langident_pipeline_demo.ipynb](https://github.com/impresso/impresso-datalab-notebooks/tree/main/annotate/langident_pipeline_demo.ipynb)
 - [ocrqa_pipeline_demo.ipynb](https://github.com/impresso/impresso-datalab-notebooks/tree/main/annotate/ocrqa_pipeline_demo.ipynb)
 - [ldatopics_pipeline_demo.ipynb](https://github.com/impresso/impresso-datalab-notebooks/tree/main/annotate/ldatopics_pipeline_demo.ipynb)
 - [newsagencies_pipeline_demo.ipynb](https://github.com/impresso/impresso-datalab-notebooks/tree/main/annotate/newsagencies_pipeline_demo.ipynb)
 - [solrnormalization_pipeline_demo.ipynb](https://github.com/impresso/impresso-datalab-notebooks/tree/main/annotate/solrnormalization_pipeline_demo.ipynb).

## Future Plans
Additional functionality will be added to extend use cases and support further processing tasks.


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


