#!/usr/bin/env python3
"""Generate silver gold records for synthetic RAES public questions.

The generated records are intentionally labeled as synthetic silver gold. They
are useful for cheap filtering, pilot scoring, and data production triage, but
they are not a replacement for expert-verified RAES gold.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


VERIFIED_AT = "2026-06-02"


SOURCE_CATALOG: dict[str, list[tuple[str, str]]] = {
    # Academic / research methods
    "Search-R1": [("Search-R1 arXiv record", "https://arxiv.org/abs/2503.09516"), ("Search-R1 repository", "https://github.com/PeterGriffinJin/Search-R1")],
    "WebGPT": [("WebGPT arXiv record", "https://arxiv.org/abs/2112.09332"), ("OpenAI WebGPT blog", "https://openai.com/research/webgpt")],
    "Self-RAG": [("Self-RAG arXiv record", "https://arxiv.org/abs/2310.11511"), ("Self-RAG repository", "https://github.com/AkariAsai/self-rag")],
    "ReAct": [("ReAct arXiv record", "https://arxiv.org/abs/2210.03629"), ("ReAct project page", "https://react-lm.github.io/")],
    "RAGAS faithfulness evaluation": [("RAGAS documentation", "https://docs.ragas.io/"), ("RAGAS repository", "https://github.com/explodinggradients/ragas")],
    "FEVER evidence annotation": [("FEVER arXiv record", "https://arxiv.org/abs/1803.05355"), ("FEVER project site", "https://fever.ai/")],
    "AIS attribution evaluation": [("AIS arXiv search", "https://arxiv.org/search/?query=attribution+information+seeking&searchtype=all"), ("ALCE arXiv record", "https://arxiv.org/abs/2305.14627")],
    "BEIR retrieval benchmarking": [("BEIR arXiv record", "https://arxiv.org/abs/2104.08663"), ("BEIR repository", "https://github.com/beir-cellar/beir")],
    "BrowseComp": [("BrowseComp arXiv record", "https://arxiv.org/abs/2504.12516"), ("BrowseComp benchmark page", "https://openai.com/index/browsecomp/")],
    "GAIA": [("GAIA arXiv record", "https://arxiv.org/abs/2311.12983"), ("GAIA dataset page", "https://huggingface.co/datasets/gaia-benchmark/GAIA")],
    "WebArena": [("WebArena arXiv record", "https://arxiv.org/abs/2307.13854"), ("WebArena repository", "https://github.com/web-arena-x/webarena")],
    "SWE-bench": [("SWE-bench arXiv record", "https://arxiv.org/abs/2310.06770"), ("SWE-bench website", "https://www.swebench.com/")],
    "STORM": [("STORM arXiv record", "https://arxiv.org/abs/2402.14207"), ("STORM repository", "https://github.com/stanford-oval/storm")],
    "DSPy teleprompters": [("DSPy documentation", "https://dspy.ai/"), ("DSPy repository", "https://github.com/stanfordnlp/dspy")],
    "Tree of Thoughts": [("Tree of Thoughts arXiv record", "https://arxiv.org/abs/2305.10601"), ("Tree of Thoughts repository", "https://github.com/princeton-nlp/tree-of-thought-llm")],
    "Reflexion": [("Reflexion arXiv record", "https://arxiv.org/abs/2303.11366"), ("Reflexion repository", "https://github.com/noahshinn/reflexion")],
    "Voyager": [("Voyager arXiv record", "https://arxiv.org/abs/2305.16291"), ("Voyager repository", "https://github.com/MineDojo/Voyager")],
    "Toolformer": [("Toolformer arXiv record", "https://arxiv.org/abs/2302.04761")],
    "Let us Verify Step by Step": [("Let's Verify Step by Step arXiv record", "https://arxiv.org/abs/2305.20050")],
    "process supervision": [("Let's Verify Step by Step arXiv record", "https://arxiv.org/abs/2305.20050"), ("OpenAI process supervision overview", "https://openai.com/research/improving-mathematical-reasoning-with-process-supervision")],
    "retrieval-augmented generation": [("RAG original paper", "https://arxiv.org/abs/2005.11401"), ("RAG Hugging Face docs", "https://huggingface.co/docs/transformers/model_doc/rag")],
    "agentic RAG": [("LangGraph documentation", "https://langchain-ai.github.io/langgraph/"), ("LlamaIndex agents docs", "https://docs.llamaindex.ai/")],
    "long-context retrieval": [("LongBench arXiv record", "https://arxiv.org/abs/2308.14508"), ("RULER arXiv record", "https://arxiv.org/abs/2404.06654")],
    "citation faithfulness": [("ALCE arXiv record", "https://arxiv.org/abs/2305.14627"), ("RAGAS documentation", "https://docs.ragas.io/")],
    "uncertainty calibration": [("Selective prediction overview", "https://arxiv.org/abs/1705.08500"), ("TruthfulQA arXiv record", "https://arxiv.org/abs/2109.07958")],
    "source attribution": [("ALCE arXiv record", "https://arxiv.org/abs/2305.14627"), ("AIS arXiv search", "https://arxiv.org/search/?query=attribution+information+seeking&searchtype=all")],
    "multi-hop web QA": [("HotpotQA arXiv record", "https://arxiv.org/abs/1809.09600"), ("2WikiMultihopQA paper", "https://aclanthology.org/2020.coling-main.580/")],
    "retriever learning": [("DPR arXiv record", "https://arxiv.org/abs/2004.04906"), ("BEIR arXiv record", "https://arxiv.org/abs/2104.08663")],
    "answer abstention": [("Selective prediction overview", "https://arxiv.org/abs/1705.08500"), ("TruthfulQA arXiv record", "https://arxiv.org/abs/2109.07958")],
    "evidence sufficiency": [("FEVER arXiv record", "https://arxiv.org/abs/1803.05355"), ("RAGAS documentation", "https://docs.ragas.io/")],
    # Deep-search / benchmark artifacts
    "OpenAI Deep Research": [("OpenAI Deep Research announcement", "https://openai.com/index/introducing-deep-research/"), ("OpenAI Deep Research help", "https://help.openai.com/en/articles/10500283-deep-research-faq")],
    "GPT Researcher": [("GPT Researcher repository", "https://github.com/assafelovic/gpt-researcher"), ("GPT Researcher docs", "https://docs.gptr.dev/")],
    "Open Deep Research": [("LangChain Open Deep Research repository", "https://github.com/langchain-ai/open_deep_research"), ("LangGraph documentation", "https://langchain-ai.github.io/langgraph/")],
    "SearchClaw": [("SearchClaw repository search", "https://github.com/search?q=SearchClaw&type=repositories"), ("SearchClaw local project placeholder", "https://github.com/search?q=SearchClaw")],
    "Storm-MM": [("Storm-MM arXiv search", "https://arxiv.org/search/?query=Storm-MM&searchtype=all"), ("STORM repository", "https://github.com/stanford-oval/storm")],
    "Perplexity-style answer engines": [("Perplexity homepage", "https://www.perplexity.ai/"), ("Perplexity help center", "https://www.perplexity.ai/hub")],
    "LangChain Open Deep Research": [("Open Deep Research repository", "https://github.com/langchain-ai/open_deep_research"), ("LangChain blog", "https://blog.langchain.dev/")],
    "LangGraph research agents": [("LangGraph documentation", "https://langchain-ai.github.io/langgraph/"), ("LangGraph repository", "https://github.com/langchain-ai/langgraph")],
    "AutoGen research workflows": [("AutoGen documentation", "https://microsoft.github.io/autogen/"), ("AutoGen repository", "https://github.com/microsoft/autogen")],
    "CrewAI research workflows": [("CrewAI documentation", "https://docs.crewai.com/"), ("CrewAI repository", "https://github.com/crewAIInc/crewAI")],
    "OpenHands browsing agents": [("OpenHands repository", "https://github.com/All-Hands-AI/OpenHands"), ("OpenHands docs", "https://docs.all-hands.dev/")],
    "MiniWoB++": [("MiniWoB++ repository", "https://github.com/Farama-Foundation/miniwob-plusplus"), ("MiniWoB++ paper", "https://arxiv.org/abs/1802.08802")],
    "OSWorld": [("OSWorld arXiv record", "https://arxiv.org/abs/2404.07972"), ("OSWorld repository", "https://github.com/xlang-ai/OSWorld")],
    "SimpleQA": [("SimpleQA benchmark page", "https://openai.com/index/introducing-simpleqa/"), ("SimpleQA dataset", "https://github.com/openai/simple-evals")],
    "FreshQA": [("FreshQA arXiv record", "https://arxiv.org/abs/2310.03214"), ("FreshQA repository", "https://github.com/freshllms/freshqa")],
    "FRAMES": [("FRAMES dataset page", "https://huggingface.co/datasets/google/frames-benchmark"), ("FRAMES arXiv search", "https://arxiv.org/search/?query=FRAMES+benchmark&searchtype=all")],
    "HotpotQA": [("HotpotQA arXiv record", "https://arxiv.org/abs/1809.09600"), ("HotpotQA website", "https://hotpotqa.github.io/")],
    "Natural Questions": [("Natural Questions paper", "https://aclanthology.org/Q19-1026/"), ("Natural Questions dataset", "https://ai.google.com/research/NaturalQuestions")],
    "ELI5": [("ELI5 arXiv record", "https://arxiv.org/abs/1907.09190"), ("ELI5 dataset page", "https://huggingface.co/datasets/eli5")],
    "ASQA": [("ASQA arXiv record", "https://arxiv.org/abs/2204.06092"), ("ASQA dataset page", "https://github.com/google-research-datasets/asqa")],
    "ALCE": [("ALCE arXiv record", "https://arxiv.org/abs/2305.14627"), ("ALCE repository", "https://github.com/princeton-nlp/ALCE")],
    "Qasper": [("Qasper arXiv record", "https://arxiv.org/abs/2105.03011"), ("Qasper dataset", "https://allenai.org/data/qasper")],
    "MuSiQue": [("MuSiQue paper", "https://aclanthology.org/2022.tacl-1.31/"), ("MuSiQue repository", "https://github.com/stonybrooknlp/musique")],
    "2WikiMultihopQA": [("2WikiMultihopQA paper", "https://aclanthology.org/2020.coling-main.580/"), ("2WikiMultihopQA repository", "https://github.com/Alab-NII/2wikimultihop")],
    "KILT": [("KILT arXiv record", "https://arxiv.org/abs/2009.02252"), ("KILT benchmark", "https://ai.facebook.com/tools/kilt/")],
    "HaluBench": [("HaluBench search", "https://huggingface.co/datasets?search=HaluBench"), ("Hugging Face datasets", "https://huggingface.co/datasets")],
    "LongBench": [("LongBench arXiv record", "https://arxiv.org/abs/2308.14508"), ("LongBench repository", "https://github.com/THUDM/LongBench")],
    "RULER": [("RULER arXiv record", "https://arxiv.org/abs/2404.06654"), ("RULER repository", "https://github.com/NVIDIA/RULER")],
    "HELM": [("HELM paper", "https://arxiv.org/abs/2211.09110"), ("HELM website", "https://crfm.stanford.edu/helm/latest/")],
    "OpenCompass": [("OpenCompass repository", "https://github.com/open-compass/opencompass"), ("OpenCompass documentation", "https://opencompass.readthedocs.io/")],
    "Chatbot Arena": [("Chatbot Arena paper", "https://arxiv.org/abs/2403.04132"), ("LMSYS Chatbot Arena", "https://lmarena.ai/")],
    "BigCodeBench": [("BigCodeBench arXiv record", "https://arxiv.org/abs/2406.15877"), ("BigCodeBench website", "https://bigcode-bench.github.io/")],
    "LiveCodeBench": [("LiveCodeBench arXiv record", "https://arxiv.org/abs/2403.07974"), ("LiveCodeBench website", "https://livecodebench.github.io/")],
    "HumanEval": [("HumanEval paper", "https://arxiv.org/abs/2107.03374"), ("HumanEval repository", "https://github.com/openai/human-eval")],
    "MMLU-Pro": [("MMLU-Pro arXiv record", "https://arxiv.org/abs/2406.01574"), ("MMLU-Pro repository", "https://github.com/TIGER-AI-Lab/MMLU-Pro")],
    "Arena-Hard": [("Arena-Hard repository", "https://github.com/lm-sys/arena-hard-auto"), ("LMSYS blog", "https://lmsys.org/blog/")],
    # Technical tools
    "LangGraph": [("LangGraph documentation", "https://langchain-ai.github.io/langgraph/"), ("LangGraph repository", "https://github.com/langchain-ai/langgraph")],
    "LangChain": [("LangChain documentation", "https://python.langchain.com/"), ("LangChain repository", "https://github.com/langchain-ai/langchain")],
    "LlamaIndex": [("LlamaIndex documentation", "https://docs.llamaindex.ai/"), ("LlamaIndex repository", "https://github.com/run-llama/llama_index")],
    "AutoGen": [("AutoGen documentation", "https://microsoft.github.io/autogen/"), ("AutoGen repository", "https://github.com/microsoft/autogen")],
    "OpenAI Agents SDK": [("OpenAI Agents SDK docs", "https://openai.github.io/openai-agents-python/"), ("OpenAI Agents SDK repository", "https://github.com/openai/openai-agents-python")],
    "OpenAI Responses API": [("OpenAI Responses API docs", "https://platform.openai.com/docs/api-reference/responses"), ("OpenAI docs", "https://platform.openai.com/docs")],
    "OpenAI tracing": [("OpenAI Agents tracing docs", "https://openai.github.io/openai-agents-python/tracing/"), ("OpenAI platform docs", "https://platform.openai.com/docs")],
    "Anthropic MCP": [("Model Context Protocol docs", "https://modelcontextprotocol.io/"), ("MCP specification", "https://spec.modelcontextprotocol.io/")],
    "Haystack": [("Haystack documentation", "https://docs.haystack.deepset.ai/"), ("Haystack repository", "https://github.com/deepset-ai/haystack")],
    "RAGAS": [("RAGAS documentation", "https://docs.ragas.io/"), ("RAGAS repository", "https://github.com/explodinggradients/ragas")],
    "DSPy": [("DSPy documentation", "https://dspy.ai/"), ("DSPy repository", "https://github.com/stanfordnlp/dspy")],
    "CrewAI": [("CrewAI documentation", "https://docs.crewai.com/"), ("CrewAI repository", "https://github.com/crewAIInc/crewAI")],
    "OpenHands": [("OpenHands repository", "https://github.com/All-Hands-AI/OpenHands"), ("OpenHands docs", "https://docs.all-hands.dev/")],
    "SWE-agent": [("SWE-agent repository", "https://github.com/SWE-agent/SWE-agent"), ("SWE-agent docs", "https://swe-agent.com/")],
    "Aider": [("Aider documentation", "https://aider.chat/docs/"), ("Aider repository", "https://github.com/Aider-AI/aider")],
    "Pydantic AI": [("Pydantic AI documentation", "https://ai.pydantic.dev/"), ("Pydantic AI repository", "https://github.com/pydantic/pydantic-ai")],
    "Semantic Kernel": [("Semantic Kernel docs", "https://learn.microsoft.com/en-us/semantic-kernel/"), ("Semantic Kernel repository", "https://github.com/microsoft/semantic-kernel")],
    "Vercel AI SDK": [("Vercel AI SDK docs", "https://sdk.vercel.ai/docs"), ("AI SDK repository", "https://github.com/vercel/ai")],
    "Transformers Agents": [("Transformers agents docs", "https://huggingface.co/docs/transformers/agents"), ("Transformers repository", "https://github.com/huggingface/transformers")],
    "Hugging Face smolagents": [("smolagents docs", "https://huggingface.co/docs/smolagents/"), ("smolagents repository", "https://github.com/huggingface/smolagents")],
    "Qdrant": [("Qdrant documentation", "https://qdrant.tech/documentation/"), ("Qdrant repository", "https://github.com/qdrant/qdrant")],
    "Milvus": [("Milvus documentation", "https://milvus.io/docs"), ("Milvus repository", "https://github.com/milvus-io/milvus")],
    "Weaviate": [("Weaviate documentation", "https://weaviate.io/developers/weaviate"), ("Weaviate repository", "https://github.com/weaviate/weaviate")],
    "Chroma": [("Chroma documentation", "https://docs.trychroma.com/"), ("Chroma repository", "https://github.com/chroma-core/chroma")],
    "FAISS": [("FAISS documentation", "https://faiss.ai/"), ("FAISS repository", "https://github.com/facebookresearch/faiss")],
    "Elasticsearch vector search": [("Elasticsearch vector search docs", "https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html"), ("Elasticsearch docs", "https://www.elastic.co/guide/")],
    "Postgres pgvector": [("pgvector repository", "https://github.com/pgvector/pgvector"), ("PostgreSQL documentation", "https://www.postgresql.org/docs/")],
    "Cohere rerank": [("Cohere rerank docs", "https://docs.cohere.com/docs/rerank-2"), ("Cohere API docs", "https://docs.cohere.com/")],
    "Jina embeddings": [("Jina embeddings docs", "https://jina.ai/embeddings/"), ("Jina AI documentation", "https://jina.ai/")],
    "ColBERT": [("ColBERT paper", "https://arxiv.org/abs/2004.12832"), ("ColBERT repository", "https://github.com/stanford-futuredata/ColBERT")],
    "vLLM": [("vLLM documentation", "https://docs.vllm.ai/"), ("vLLM repository", "https://github.com/vllm-project/vllm")],
    "SGLang": [("SGLang documentation", "https://docs.sglang.ai/"), ("SGLang repository", "https://github.com/sgl-project/sglang")],
    "llama.cpp": [("llama.cpp repository", "https://github.com/ggml-org/llama.cpp"), ("llama.cpp docs", "https://github.com/ggml-org/llama.cpp/tree/master/docs")],
    "Ollama": [("Ollama documentation", "https://github.com/ollama/ollama/tree/main/docs"), ("Ollama repository", "https://github.com/ollama/ollama")],
    "LiteLLM": [("LiteLLM documentation", "https://docs.litellm.ai/"), ("LiteLLM repository", "https://github.com/BerriAI/litellm")],
    "Ray Serve": [("Ray Serve docs", "https://docs.ray.io/en/latest/serve/"), ("Ray repository", "https://github.com/ray-project/ray")],
    "FastAPI": [("FastAPI documentation", "https://fastapi.tiangolo.com/"), ("FastAPI repository", "https://github.com/fastapi/fastapi")],
    # Fact-checking / public evidence
    "FEVER": [("FEVER arXiv record", "https://arxiv.org/abs/1803.05355"), ("FEVER project site", "https://fever.ai/")],
    "Climate-FEVER": [("Climate-FEVER paper", "https://aclanthology.org/2020.fever-1.5/"), ("Climate-FEVER dataset", "https://www.sustainablefinance.uzh.ch/en/research/climate-fever.html")],
    "COVID-Fact": [("COVID-Fact arXiv search", "https://arxiv.org/search/?query=COVID-Fact&searchtype=all"), ("COVID-Fact dataset search", "https://huggingface.co/datasets?search=COVID-Fact")],
    "LIAR": [("LIAR paper", "https://aclanthology.org/P17-2067/"), ("LIAR dataset page", "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip")],
    "ClaimBuster": [("ClaimBuster website", "https://idir.uta.edu/claimbuster/"), ("ClaimBuster paper", "https://dl.acm.org/doi/10.1145/2872518.2890098")],
    "IFCN Code of Principles": [("IFCN Code of Principles", "https://ifcncodeofprinciples.poynter.org/"), ("Poynter IFCN", "https://www.poynter.org/ifcn/")],
    "PolitiFact": [("PolitiFact website", "https://www.politifact.com/"), ("PolitiFact methodology", "https://www.politifact.com/article/2018/feb/12/principles-truth-o-meter-politifacts-methodology-i/")],
    "Snopes": [("Snopes website", "https://www.snopes.com/"), ("Snopes fact-checking process", "https://www.snopes.com/about/")],
    "Full Fact": [("Full Fact website", "https://fullfact.org/"), ("Full Fact methodology", "https://fullfact.org/about/how-we-work/")],
    "AP Fact Check": [("AP Fact Check", "https://apnews.com/hub/ap-fact-check"), ("AP standards", "https://www.ap.org/about/news-values-and-principles/")],
    "Reuters Fact Check": [("Reuters Fact Check", "https://www.reuters.com/fact-check/"), ("Reuters trust principles", "https://www.thomsonreuters.com/en/about-us/trust-principles.html")],
    "AFP Fact Check": [("AFP Fact Check", "https://factcheck.afp.com/"), ("AFP fact-checking methodology", "https://factcheck.afp.com/fact-checking-how-we-work")],
    "Logically Facts": [("Logically Facts", "https://www.logicallyfacts.com/"), ("Logically Facts methodology", "https://www.logicallyfacts.com/en/about-us")],
    "EUvsDisinfo": [("EUvsDisinfo", "https://euvsdisinfo.eu/"), ("EUvsDisinfo methodology", "https://euvsdisinfo.eu/about/")],
    "Media Bias/Fact Check": [("Media Bias/Fact Check", "https://mediabiasfactcheck.com/"), ("MBFC methodology", "https://mediabiasfactcheck.com/methodology/")],
    "Google Fact Check Tools": [("Google Fact Check Tools", "https://toolbox.google.com/factcheck/explorer"), ("ClaimReview docs", "https://developers.google.com/search/docs/appearance/structured-data/factcheck")],
    "ClaimReview markup": [("ClaimReview docs", "https://developers.google.com/search/docs/appearance/structured-data/factcheck"), ("Schema.org ClaimReview", "https://schema.org/ClaimReview")],
    "Poynter fact-checking guidelines": [("Poynter IFCN", "https://www.poynter.org/ifcn/"), ("IFCN Code of Principles", "https://ifcncodeofprinciples.poynter.org/")],
    "WHO mythbusters pages": [("WHO health topics", "https://www.who.int/health-topics"), ("WHO news", "https://www.who.int/news")],
    "CDC health misinformation pages": [("CDC website", "https://www.cdc.gov/"), ("CDC health topics", "https://www.cdc.gov/health-topics.html")],
    "NASA climate evidence pages": [("NASA climate evidence", "https://climate.nasa.gov/evidence/"), ("NASA climate", "https://climate.nasa.gov/")],
    "NOAA climate summaries": [("NOAA climate", "https://www.noaa.gov/climate"), ("NOAA climate.gov", "https://www.climate.gov/")],
    "IPCC reports": [("IPCC reports", "https://www.ipcc.ch/reports/"), ("IPCC website", "https://www.ipcc.ch/")],
    "Our World in Data": [("Our World in Data", "https://ourworldindata.org/"), ("OWID grapher", "https://ourworldindata.org/grapher")],
    "World Bank data": [("World Bank Data", "https://data.worldbank.org/"), ("World Bank API", "https://datahelpdesk.worldbank.org/knowledgebase/topics/125589")],
    "UN data portal": [("UN Data", "https://data.un.org/"), ("UN Statistics Division", "https://unstats.un.org/")],
    "OECD data": [("OECD Data", "https://data.oecd.org/"), ("OECD Data Explorer", "https://data-explorer.oecd.org/")],
    "BLS statistics": [("BLS website", "https://www.bls.gov/"), ("BLS data tools", "https://www.bls.gov/data/")],
    "FRED economic data": [("FRED", "https://fred.stlouisfed.org/"), ("FRED API docs", "https://fred.stlouisfed.org/docs/api/fred/")],
    "SEC company filings": [("SEC EDGAR", "https://www.sec.gov/edgar"), ("SEC company search", "https://www.sec.gov/edgar/search/")],
    # Data analysis / ML tooling
    "pandas": [("pandas documentation", "https://pandas.pydata.org/docs/"), ("pandas repository", "https://github.com/pandas-dev/pandas")],
    "NumPy": [("NumPy documentation", "https://numpy.org/doc/"), ("NumPy repository", "https://github.com/numpy/numpy")],
    "SciPy": [("SciPy documentation", "https://docs.scipy.org/doc/scipy/"), ("SciPy repository", "https://github.com/scipy/scipy")],
    "scikit-learn": [("scikit-learn documentation", "https://scikit-learn.org/stable/"), ("scikit-learn repository", "https://github.com/scikit-learn/scikit-learn")],
    "statsmodels": [("statsmodels documentation", "https://www.statsmodels.org/stable/"), ("statsmodels repository", "https://github.com/statsmodels/statsmodels")],
    "PyTorch": [("PyTorch documentation", "https://pytorch.org/docs/stable/"), ("PyTorch repository", "https://github.com/pytorch/pytorch")],
    "TensorFlow": [("TensorFlow documentation", "https://www.tensorflow.org/api_docs"), ("TensorFlow repository", "https://github.com/tensorflow/tensorflow")],
    "JAX": [("JAX documentation", "https://jax.readthedocs.io/"), ("JAX repository", "https://github.com/jax-ml/jax")],
    "XGBoost": [("XGBoost documentation", "https://xgboost.readthedocs.io/"), ("XGBoost repository", "https://github.com/dmlc/xgboost")],
    "LightGBM": [("LightGBM documentation", "https://lightgbm.readthedocs.io/"), ("LightGBM repository", "https://github.com/microsoft/LightGBM")],
    "CatBoost": [("CatBoost documentation", "https://catboost.ai/docs/"), ("CatBoost repository", "https://github.com/catboost/catboost")],
    "Polars": [("Polars documentation", "https://docs.pola.rs/"), ("Polars repository", "https://github.com/pola-rs/polars")],
    "DuckDB": [("DuckDB documentation", "https://duckdb.org/docs/"), ("DuckDB repository", "https://github.com/duckdb/duckdb")],
    "Apache Arrow": [("Apache Arrow documentation", "https://arrow.apache.org/docs/"), ("Apache Arrow repository", "https://github.com/apache/arrow")],
    "Dask": [("Dask documentation", "https://docs.dask.org/"), ("Dask repository", "https://github.com/dask/dask")],
    "Ray Data": [("Ray Data docs", "https://docs.ray.io/en/latest/data/data.html"), ("Ray repository", "https://github.com/ray-project/ray")],
    "Spark MLlib": [("Spark MLlib docs", "https://spark.apache.org/docs/latest/ml-guide.html"), ("Spark repository", "https://github.com/apache/spark")],
    "Vega-Lite": [("Vega-Lite documentation", "https://vega.github.io/vega-lite/"), ("Vega-Lite repository", "https://github.com/vega/vega-lite")],
    "Altair": [("Altair documentation", "https://altair-viz.github.io/"), ("Altair repository", "https://github.com/vega/altair")],
    "Plotly": [("Plotly Python docs", "https://plotly.com/python/"), ("Plotly.py repository", "https://github.com/plotly/plotly.py")],
    "Matplotlib": [("Matplotlib documentation", "https://matplotlib.org/stable/"), ("Matplotlib repository", "https://github.com/matplotlib/matplotlib")],
    "Seaborn": [("Seaborn documentation", "https://seaborn.pydata.org/"), ("Seaborn repository", "https://github.com/mwaskom/seaborn")],
    "Great Expectations": [("Great Expectations documentation", "https://docs.greatexpectations.io/"), ("Great Expectations repository", "https://github.com/great-expectations/great_expectations")],
    "dbt-core": [("dbt-core documentation", "https://docs.getdbt.com/"), ("dbt-core repository", "https://github.com/dbt-labs/dbt-core")],
    "OpenML Python": [("OpenML Python docs", "https://openml.github.io/openml-python/"), ("OpenML Python repository", "https://github.com/openml/openml-python")],
    "Kaggle datasets": [("Kaggle datasets", "https://www.kaggle.com/datasets"), ("Kaggle API repository", "https://github.com/Kaggle/kaggle-api")],
    "Hugging Face Datasets": [("Hugging Face Datasets docs", "https://huggingface.co/docs/datasets/"), ("Datasets repository", "https://github.com/huggingface/datasets")],
    "UCI Machine Learning Repository": [("UCI ML Repository", "https://archive.ics.uci.edu/"), ("UCI repository datasets", "https://archive.ics.uci.edu/datasets")],
    "BigQuery public datasets": [("BigQuery public datasets docs", "https://cloud.google.com/bigquery/public-data"), ("BigQuery docs", "https://cloud.google.com/bigquery/docs")],
    "DataHub": [("DataHub documentation", "https://datahubproject.io/docs/"), ("DataHub repository", "https://github.com/datahub-project/datahub")],
    "Evidently AI": [("Evidently documentation", "https://docs.evidentlyai.com/"), ("Evidently repository", "https://github.com/evidentlyai/evidently")],
    "WhyLabs": [("WhyLabs documentation", "https://docs.whylabs.ai/"), ("WhyLabs homepage", "https://whylabs.ai/")],
    "MLflow": [("MLflow documentation", "https://mlflow.org/docs/latest/"), ("MLflow repository", "https://github.com/mlflow/mlflow")],
    "Weights and Biases": [("Weights & Biases docs", "https://docs.wandb.ai/"), ("Weights & Biases homepage", "https://wandb.ai/")],
    "DVC": [("DVC documentation", "https://dvc.org/doc"), ("DVC repository", "https://github.com/iterative/dvc")],
}


TASK_RUBRIC = {
    "conceptual_methodology": (30, 25, 20, 10, 10, 5),
    "comparative_research": (25, 25, 20, 15, 10, 5),
    "adversarial_conflict_resolution": (20, 15, 20, 15, 25, 5),
    "current_status_verification": (25, 20, 20, 15, 15, 5),
    "multi_source_timeline_reconstruction": (20, 25, 20, 15, 15, 5),
    "leaderboard_verification": (25, 20, 20, 20, 10, 5),
    "implementation_verification": (25, 20, 20, 20, 10, 5),
    "repo_implementation_audit": (25, 20, 20, 20, 10, 5),
    "documentation_change_tracking": (25, 20, 20, 20, 10, 5),
    "live_fact_verification": (25, 20, 25, 15, 10, 5),
    "claim_decomposition_and_verification": (25, 25, 20, 15, 10, 5),
    "unanswerable_detection": (20, 15, 20, 15, 25, 5),
    "negative_evidence_search": (20, 10, 20, 15, 35, 0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic-public",
        default="raes-bench/raes-bench-v12_process/public_questions_1k_synthetic_additions.jsonl",
    )
    parser.add_argument("--base-gold", default="raes-bench/raes-bench-v12_process/gold.jsonl")
    parser.add_argument(
        "--output-synthetic-gold",
        default="raes-bench/raes-bench-v12_process/gold_synthetic_silver_600.jsonl",
    )
    parser.add_argument(
        "--output-mixed-gold",
        default="raes-bench/raes-bench-v12_process/gold_1k_mixed_silver.jsonl",
    )
    parser.add_argument(
        "--output-dev-with-gold",
        default="raes-bench/raes-bench-v12_process/dev_with_gold_1k_mixed_silver.jsonl",
    )
    parser.add_argument(
        "--report",
        default="raes-bench/raes-bench-v12_process/gold_synthetic_silver_600_report.json",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def slug(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return cleaned[:80] or "source"


def infer_subject(question: str) -> str:
    text = question.split(" Include a secondary check on ", 1)[0]
    patterns = [
        r"^For (.*?), verify ",
        r"^For (.*?), search ",
        r"^For (.*?), determine ",
        r"^Track documentation/version evidence for (.*?):",
        r"^Compare (.*?) with ",
        r"^As of [0-9-]+, compare the public evidence for (.*?) and ",
        r"^As of [0-9-]+, verify a public claim involving (.*?);",
        r"^Reconstruct the public evidence timeline for (.*?) as ",
        r"^Audit (.*?):",
        r"^As of [0-9-]+, verify whether .*? about (.*?) is ",
        r"^Decompose .*? about (.*?) into ",
        r"^Resolve conflicting .*? about (.*?)(?::|;)",
        r"^As of [0-9-]+, verify the current public status of (.*?) and ",
        r"^Determine whether public sources are sufficient to answer a narrow question about (.*?) with ",
        r"^As of [0-9-]+, determine whether public sources are sufficient to answer a narrow question about (.*?) with ",
        r"^As of [0-9-]+, decide whether public evidence is sufficient to answer a narrow claim about (.*?) with ",
        r"^Verify the current status of (.*?) with ",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return text[:80].strip()


def infer_answerability(task_type: str) -> str:
    if task_type == "adversarial_conflict_resolution":
        return "conflicting_no_resolution"
    if task_type in {"negative_evidence_search", "unanswerable_detection"}:
        return "not_enough_public_evidence"
    if task_type in {"current_status_verification", "documentation_change_tracking", "live_fact_verification"}:
        return "answerable_with_uncertainty"
    return "answerable_with_uncertainty"


def infer_verification_mode(task_type: str) -> tuple[str, str]:
    if task_type == "documentation_change_tracking":
        return "api_doc_change_tracking", "API/doc change tracking"
    if task_type in {"implementation_verification", "repo_implementation_audit"}:
        return "documentation_lookup", "documentation lookup"
    return "multi_source_verification", "multi-source verification"


def source_type_for_url(url: str) -> str:
    host = urlparse(url).netloc.lower()
    path = urlparse(url).path.lower()
    if "arxiv.org" in host or "aclanthology.org" in host or "dl.acm.org" in host:
        return "paper"
    if "github.com" in host or "gitlab" in host:
        return "repository"
    if "huggingface.co/datasets" in url.lower() or "datasets" in path or "data." in host or "archive.ics.uci.edu" in host:
        return "dataset"
    if any(token in host for token in ["docs.", "readthedocs", "learn.microsoft.com", "developers.google.com"]):
        return "documentation"
    if any(token in host for token in ["sec.gov", "ifcncodeofprinciples", "schema.org"]):
        return "standard" if "schema.org" in host else "legal"
    if path in {"", "/"}:
        return "index"
    return "documentation"


def source_record(subject: str, title: str, url: str, index: int) -> dict[str, Any]:
    source_type = source_type_for_url(url)
    host = urlparse(url).netloc.replace("www.", "")
    mutable = source_type in {"documentation", "repository", "dataset", "index"}
    return {
        "id": f"synth_{slug(subject)}_{index}",
        "title": title,
        "url": url,
        "source_type": source_type,
        "publisher": host or "public web",
        "verified_at": VERIFIED_AT,
        "authority_level": "synthetic_silver_locator",
        "freshness_requirement": "current_required" if mutable else "stable",
        "expected_validity_window": "30 days" if mutable else "stable unless source changes",
        "requires_recheck_at_eval_time": True,
        "role": "answer_source",
        "supports_answer": True,
        "legacy_source_type": source_type,
        "primary_type": source_type,
        "peer_reviewed": source_type == "paper",
        "accessed_at": VERIFIED_AT,
    }


def fallback_sources(subject: str, category: str) -> list[tuple[str, str]]:
    encoded = subject.replace(" ", "+")
    if category == "academic":
        return [(f"{subject} arXiv search", f"https://arxiv.org/search/?query={encoded}&searchtype=all")]
    if category == "technical":
        return [(f"{subject} GitHub search", f"https://github.com/search?q={encoded}&type=repositories")]
    if category == "data_analysis":
        return [(f"{subject} documentation search", f"https://github.com/search?q={encoded}&type=repositories")]
    if category == "fact_check":
        return [(f"{subject} public evidence search", f"https://www.google.com/search?q={encoded}")]
    return [(f"{subject} public evidence search", f"https://www.google.com/search?q={encoded}")]


def gold_sources_for(subject: str, category: str) -> tuple[list[dict[str, Any]], bool]:
    entries = SOURCE_CATALOG.get(subject)
    used_fallback = False
    if not entries:
        entries = fallback_sources(subject, category)
        used_fallback = True
    return [source_record(subject, title, url, idx + 1) for idx, (title, url) in enumerate(entries[:3])], used_fallback


def required_facets(subject: str, task_type: str) -> dict[str, list[str]]:
    return {
        "content_facets": [
            f"Identify the direct public evidence boundary for {subject}",
            "Separate directly supported claims from contextual or adjacent evidence",
        ],
        "evidence_facets": [
            "Cite opened primary or official sources rather than search snippets",
            "Use enough independent public evidence for the task scope",
        ],
        "reasoning_facets": [
            "State uncertainty when evidence is indirect, stale, conflicting, or absent",
            f"Apply the task-specific verification mode for {task_type}",
        ],
    }


def source_policy(gold_sources: list[dict[str, Any]], task_type: str) -> dict[str, Any]:
    required_types = sorted({source["source_type"] for source in gold_sources[:2]})
    multi = task_type not in {"implementation_verification"} or len(gold_sources) > 1
    return {
        "required_primary_sources": required_types or ["documentation"],
        "secondary_sources_allowed": True,
        "third_party_sources_allowed_as": "context_only",
        "notes": "Synthetic silver policy: prefer direct topic-specific public sources; adjacent pages only support context.",
        "source_count_mode": "multi_source_verification" if multi else "single_source_verification",
        "source_count_label": "multi source verification" if multi else "single source verification",
        "single_source_allowed": not multi,
        "multi_source_required": multi,
    }


def min_evidence_policy(gold_sources: list[dict[str, Any]], task_type: str) -> dict[str, Any]:
    min_citations = 3 if task_type in {"multi_source_timeline_reconstruction", "adversarial_conflict_resolution"} else min(2, max(1, len(gold_sources)))
    min_domains = min(2, len({urlparse(s["url"]).netloc for s in gold_sources}))
    return {
        "min_citations": min_citations,
        "min_domains": max(1, min_domains),
        "direct_support_required": True,
        "must_resolve_conflicts": task_type == "adversarial_conflict_resolution",
        "evidence_scope_label": "synthetic silver multi-source verification",
        "min_gold_sources": min(len(gold_sources), min_citations),
        "multi_source_required": min_citations > 1,
        "multi_source_justification": "Synthetic silver item for trajectory filtering; exact source support should be rechecked before expert evaluation.",
    }


def build_gold_answer(row: dict[str, Any], subject: str, answerability: str) -> dict[str, Any]:
    task_type = row["task_type"]
    final = (
        f"A correct answer should verify {subject} using direct public evidence, identify the boundary of what the cited sources support, "
        "distinguish primary or official evidence from contextual sources, and avoid stronger claims than the sources justify. "
        "If current status, conflict, negative evidence, or documentation versioning is involved, the answer should explicitly state uncertainty and explain what cannot be concluded."
    )
    facets = required_facets(subject, task_type)
    return {
        "final_answer": final,
        "must_include": {
            "content_points": facets["content_facets"] + [subject, task_type.replace("_", " ")],
            "evidence_points": facets["evidence_facets"],
            "reasoning_points": facets["reasoning_facets"],
            "boundary_conditions": [
                "Use direct topic-specific public sources first.",
                "Do not use search snippets as answer evidence.",
                "Do not claim certainty for stale, missing, indirect, or conflicting evidence.",
            ],
            "uncertainty_statements": [
                "State uncertainty when direct support is incomplete or current status depends on mutable web documentation."
            ],
        },
        "must_not_include": [
            "Unsupported claims beyond the cited source boundary.",
            "Definitive claims from snippets or broad secondary summaries alone.",
        ],
        "acceptable_variants": [
            "Equivalent wording is acceptable if it preserves direct-source grounding and uncertainty boundaries.",
            "Concise answers are acceptable when all required facets and citations are covered.",
        ],
        "unacceptable_answers": [
            "Answering from search snippets without opening the cited page.",
            "Citing only a topically related source for a narrow verification claim.",
            "Omitting uncertainty for conflicting, stale, or insufficient public evidence.",
        ],
        "answerability": answerability,
        "uncertainty_requirement": "Required when direct public support is incomplete, mutable, conflicting, or absent.",
        "reference_answers": [
            {
                "text": final,
                "style": "synthetic_silver",
                "authored_by": "script",
                "score_expected": 80,
            }
        ],
    }


def facet_evidence(facets: dict[str, list[str]], gold_sources: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    if not gold_sources:
        return out
    flat_facets = facets["content_facets"] + facets["evidence_facets"] + facets["reasoning_facets"]
    for idx, facet in enumerate(flat_facets):
        source = gold_sources[idx % len(gold_sources)]
        out[facet] = [
            {
                "source_id": source["id"],
                "claim_supported": f"Synthetic silver locator for required facet: {facet}",
                "evidence_quote": "",
                "evidence_summary": (
                    f"{source['title']} is a synthetic silver locator for checking the facet '{facet}'. "
                    "The exact quote was not stored; an evaluator should fetch and verify the page before treating this as expert gold."
                ),
                "quote_status": "synthetic_locator_no_exact_quote",
                "locator": {
                    "source_title": source["title"],
                    "url": source["url"],
                    "section": "Locate relevant public evidence in the opened source.",
                },
                "supports": "direct_or_contextual_locator",
            }
        ]
    return out


def known_conflicts(task_type: str) -> list[dict[str, Any]]:
    if task_type != "adversarial_conflict_resolution":
        return []
    return [
        {
            "conflict_id": "synthetic_c1",
            "conflict_type": "source_boundary_conflict",
            "claim_a": "A broad or stale source may appear to support the claim.",
            "claim_b": "Direct topic-specific public evidence may be narrower, version-dependent, or missing.",
            "preferred_resolution": "Prefer direct public evidence and preserve uncertainty when support is incomplete.",
            "resolution_basis": ["source_authority", "freshness_requirement", "directness_of_support"],
        }
    ]


def conflict_evidence(task_type: str, gold_sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conflicts = known_conflicts(task_type)
    if not conflicts or len(gold_sources) < 2:
        return []
    return [
        {
            "conflict_id": conflicts[0]["conflict_id"],
            "source_a": gold_sources[0]["id"],
            "source_b": gold_sources[1]["id"],
            "conflict_dimension": "source boundary or freshness",
            "resolution_rule": conflicts[0]["preferred_resolution"],
            "expected_answer_behavior": "Do not over-confirm unsupported or stale claims.",
        }
    ]


def stop_condition(policy: dict[str, Any], facets: dict[str, list[str]], task_type: str) -> dict[str, Any]:
    return {
        "minimum_sources": policy["min_citations"],
        "required_source_types": ["topic_specific_public_source", "direct_project_or_benchmark_source"],
        "required_facets_covered": "all_required_facets",
        "conflict_resolution_required": task_type == "adversarial_conflict_resolution",
        "stop_when": "Stop after direct public evidence, answerability, and uncertainty boundaries are established.",
        "expected_result": "answer_or_refuse_with_evidence_boundary",
    }


def process_diagnostics(policy: dict[str, Any], facets: dict[str, list[str]], failure_modes: list[str], gold_sources: list[dict[str, Any]]) -> dict[str, Any]:
    facet_count = sum(len(items) for items in facets.values())
    return {
        "label_version": "raes-v12-synthetic-silver",
        "target": "calibrated_evidence_process_control",
        "acceptable_next_actions": ["targeted_followup", "diversify_sources", "resolve_conflict"],
        "stop_continue_labels": [
            {
                "trace_state": "no_evidence",
                "evidence_count": 0,
                "covered_facets": 0,
                "should_continue": True,
                "preferred_action": "targeted_followup",
                "failure_if_stop": "unsupported_answer",
            },
            {
                "trace_state": "partial_evidence",
                "evidence_count": max(1, policy["min_citations"] - 1),
                "covered_facets": max(1, facet_count // 2),
                "should_continue": True,
                "preferred_action": "diversify_sources",
                "failure_if_stop": "premature_stop_or_missing_facet",
            },
            {
                "trace_state": "minimum_evidence_met",
                "evidence_count": policy["min_citations"],
                "covered_facets": facet_count,
                "should_continue": False,
                "preferred_action": "stop_and_summarize",
                "failure_if_stop": "none",
            },
        ],
        "stop_allowed_when": {
            "minimum_citations": policy["min_citations"],
            "minimum_domains": policy["min_domains"],
            "required_facets": "all_required_facets",
            "conflict_resolution_required": policy["must_resolve_conflicts"],
        },
        "failure_modes_to_predict": failure_modes,
        "hard_negative_sources": [
            {
                "description": "search snippets, stale tutorials, broad project pages, or adjacent sources",
                "negative_type": "snippet_or_adjacent_source_trap",
                "reason": "The source may mention the topic without supporting the exact claim or current version.",
                "expected_handling": "Open direct sources and preserve uncertainty when support is indirect.",
            }
        ],
        "fixed_corpus_sources": [
            {
                "source_id": source["id"],
                "url": source["url"],
                "source_type": source["source_type"],
                "supports_answer": source["supports_answer"],
            }
            for source in gold_sources
        ],
    }


def scoring_rubric(task_type: str) -> dict[str, Any]:
    answer, facet, citation, source, uncertainty, efficiency = TASK_RUBRIC.get(task_type, (25, 20, 20, 15, 15, 5))
    return {
        "total_points": 100,
        "rubric_template_id": task_type,
        "rubric_rationale": "Synthetic silver rubric for trajectory filtering; use expert rubric for final RAES reporting.",
        "criteria": [
            {"criterion": "answer_correctness", "points": answer, "required": True, "description": "Final conclusion matches the evidence boundary."},
            {"criterion": "facet_coverage", "points": facet, "required": True, "description": "Required content, evidence, and reasoning facets are covered."},
            {"criterion": "citation_faithfulness", "points": citation, "required": True, "description": "Citations directly support attached claims."},
            {"criterion": "source_quality", "points": source, "required": True, "description": "Authoritative, current, and sufficiently diverse sources are prioritized."},
            {"criterion": "conflict_uncertainty_handling", "points": uncertainty, "required": True, "description": "Conflicts, caveats, and uncertainty are handled correctly."},
            {"criterion": "search_efficiency", "points": efficiency, "required": efficiency > 0, "description": "The agent avoids premature stopping and unnecessary over-search."},
        ],
    }


def build_gold_row(row: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    subject = infer_subject(row["question"])
    answerability = infer_answerability(row["task_type"])
    verification_mode, verification_mode_label = infer_verification_mode(row["task_type"])
    facets = required_facets(subject, row["task_type"])
    gold_sources, used_fallback = gold_sources_for(subject, row["category"])
    min_policy = min_evidence_policy(gold_sources, row["task_type"])
    failure_modes = ["unsupported_claim", "stale_source_error", "source_claim_mismatch", "snippet_only_answer", "missing_uncertainty"]
    metadata = {
        "created_by": "scripted synthetic silver generator",
        "verified_at": VERIFIED_AT,
        "version": "v12-synthetic-silver",
        "source_version": "public_questions_1k_synthetic_additions",
        "release": "v1.2_process_synthetic_1k",
        "split": row["split"],
        "gold_quality": "synthetic_silver_unverified",
        "requires_expert_review": True,
        "subject": subject,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return (
        {
            **row,
            "answerability": answerability,
            "gold_answer": build_gold_answer(row, subject, answerability),
            "required_facets": facets,
            "source_policy": source_policy(gold_sources, row["task_type"]),
            "minimum_evidence_policy": min_policy,
            "gold_sources": gold_sources,
            "facet_evidence": facet_evidence(facets, gold_sources),
            "known_conflicts": known_conflicts(row["task_type"]),
            "conflict_evidence": conflict_evidence(row["task_type"], gold_sources),
            "misleading_sources": [
                {
                    "source_description": "search snippets, stale tutorials, broad project pages, or adjacent methodology sources",
                    "misleading_type": "snippet_or_adjacent_source_trap",
                    "misleading_reason": "The source may mention the topic without supporting the exact claim or current version.",
                    "expected_handling": "Open direct sources and preserve uncertainty when support is indirect.",
                }
            ],
            "common_wrong_answers": [
                "Giving a definitive answer when only indirect evidence is available.",
                "Citing a broad source for a narrow claim that needs a direct locator.",
                "Ignoring temporal caveats for mutable documentation, datasets, or leaderboards.",
            ],
            "failure_modes": failure_modes,
            "stop_condition": stop_condition(min_policy, facets, row["task_type"]),
            "uncertainty_source": "synthetic_silver_requires_uncertainty_for_indirect_or_mutable_evidence",
            "verification_mode": verification_mode,
            "verification_mode_label": verification_mode_label,
            "metadata": metadata,
            "process_diagnostics": process_diagnostics(min_policy, facets, failure_modes, gold_sources),
            "scoring_rubric": scoring_rubric(row["task_type"]),
        },
        used_fallback,
    )


def validate(rows: list[dict[str, Any]]) -> None:
    required = {
        "id",
        "category",
        "domain",
        "task_type",
        "difficulty",
        "question",
        "answerability",
        "gold_answer",
        "required_facets",
        "source_policy",
        "minimum_evidence_policy",
        "gold_sources",
        "facet_evidence",
        "known_conflicts",
        "conflict_evidence",
        "misleading_sources",
        "common_wrong_answers",
        "failure_modes",
        "stop_condition",
        "uncertainty_source",
        "verification_mode",
        "verification_mode_label",
        "metadata",
        "split",
    }
    seen = set()
    for row in rows:
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"{row.get('id')} missing fields: {missing}")
        if row["id"] in seen:
            raise ValueError(f"duplicate id: {row['id']}")
        seen.add(row["id"])
        if not row["gold_sources"]:
            raise ValueError(f"{row['id']} has no gold_sources")


def main() -> None:
    args = parse_args()
    synthetic_public = load_jsonl(Path(args.synthetic_public))
    base_gold = load_jsonl(Path(args.base_gold))
    synthetic_gold: list[dict[str, Any]] = []
    fallback_subjects: Counter[str] = Counter()
    source_type_counts: Counter[str] = Counter()
    answerability_counts: Counter[str] = Counter()
    for row in synthetic_public:
        gold_row, used_fallback = build_gold_row(row)
        synthetic_gold.append(gold_row)
        if used_fallback:
            fallback_subjects[infer_subject(row["question"])] += 1
        for source in gold_row["gold_sources"]:
            source_type_counts[source["source_type"]] += 1
        answerability_counts[gold_row["answerability"]] += 1

    mixed_gold = base_gold + synthetic_gold
    validate(synthetic_gold)
    validate(mixed_gold)

    write_jsonl(Path(args.output_synthetic_gold), synthetic_gold)
    write_jsonl(Path(args.output_mixed_gold), mixed_gold)
    write_jsonl(Path(args.output_dev_with_gold), [row for row in mixed_gold if row.get("split") == "dev"])

    report = {
        "synthetic_public": args.synthetic_public,
        "base_gold": args.base_gold,
        "output_synthetic_gold": args.output_synthetic_gold,
        "output_mixed_gold": args.output_mixed_gold,
        "output_dev_with_gold": args.output_dev_with_gold,
        "base_gold_rows": len(base_gold),
        "synthetic_gold_rows": len(synthetic_gold),
        "mixed_gold_rows": len(mixed_gold),
        "quality_label": "synthetic_silver_unverified",
        "warning": "Generated records are for trajectory filtering and pilot scoring only; they are not expert-verified benchmark gold.",
        "fallback_subjects": dict(fallback_subjects),
        "answerability": dict(answerability_counts),
        "source_types": dict(source_type_counts),
        "split": dict(Counter(row["split"] for row in synthetic_gold)),
        "category": dict(Counter(row["category"] for row in synthetic_gold)),
        "task_type": dict(Counter(row["task_type"] for row in synthetic_gold)),
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.report).with_suffix(".md").write_text(
        "# Synthetic Silver Gold Report\n\n"
        f"- Synthetic rows: `{len(synthetic_gold)}`\n"
        f"- Mixed rows: `{len(mixed_gold)}`\n"
        "- Quality label: `synthetic_silver_unverified`\n\n"
        "These records are intended for trajectory filtering and pilot scoring. "
        "They should be expert-reviewed before being used as benchmark gold.\n\n"
        "## Report\n\n```json\n"
        + json.dumps(report, ensure_ascii=False, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
