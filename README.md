# Thesis-Project

Project for the Master Thesis on Computer Engineering with University of Bologna

# Project Structure: 

The project starts from the Research Agent developed by [Jinheon Baek, Sujay Kumar Jauhar, Silviu Cucerzan, and Sung Ju Hwang](https://github.com/JinheonBaek/ResearchAgent/tree/main)

- code/
	- main.py — entrypoint to run the end-to-end pipeline
	- knowledge/
		- store.py — lightweight knowledge store and entity retrieval
	- models/
		- CustomAgent.py — Ollama server wrapper with support for a variety of open weights model
	- pipelines/
		- research_pipeline.py — orchestration of generate and validate iterations
    - review_pipeline.py — framework for analyzing existing papers
		- agents/
			- base.py — shared prompt-formatting helpers
			- problem_identifier.py — generates/refines problems
			- problem_validator.py — reviews problems across 5 metrics in parallel
            - ...
	- utils/
		- s2.py — Semantic Scholar API helpers (papers, references, embeddings)
		- data_io.py — JSONL loading and ID utilities
		- formatting.py — small text utilities
    - jutil.py — utilities to convert .pdf files to .json files while maintaining the original structure of the file. Developed to work with scientific papers

# Citation

```bibtex
@inproceedings{Baek2025ResearchAgent,
  title={ResearchAgent: Iterative Research Idea Generation over Scientific Literature with Large Language Models},
  author={Jinheon Baek and Sujay Kumar Jauhar and Silviu Cucerzan and Sung Ju Hwang},
  booktitle={NAACL},
  year={2025},
  url={https://api.semanticscholar.org/CorpusID:269042844}
}
```
