# LLM-Blending
all source code of llm blending , (source code only), large files are gitignored

-----

# LLM Ensemble with Rank-Based Fusion

This project implements an advanced LLM ensemble strategy for generating high-quality responses by combining outputs from multiple Large Language Models (LLMs). It features parallel response generation, a self-ranking mechanism, Borda Count for rank aggregation, and a final fusion step by the top-performing LLM.

-----

## Features

  * **Parallel LLM Response Generation:** Efficiently generate candidate responses from multiple LLMs simultaneously using the Hugging Face `datasets` library for data loading and `concurrent.futures` for parallel processing.
  * **Self-Ranking Mechanism:** Each participating LLM ranks the generated responses from all LLMs, providing diverse perspectives on quality.
  * **Borda Count Rank Aggregation:** Combines individual LLM rankings into a robust collective ranking using the Borda Count method.
  * **Top-K Candidate Selection:** Selects the highest-ranked candidate responses for further refinement.
  * **Fusion by Best LLM:** The overall best-performing LLM (based on Borda Count) synthesizes the top-selected candidates into a final, coherent answer.
  * **Support for Conversational Datasets:** Specifically adapted for conversational datasets like DoQA, maintaining conversation context for LLM generations.
  * **Comprehensive Experimentation Framework:** Includes predefined policies (Full Ranking, Dynamic Elimination, Alternate Ranking, Fixed Interval Elimination) to evaluate different blending strategies.
  * **Performance Tracking:** Logs generation, ranking, and fusion times for detailed performance analysis.
  * **BERTScore Evaluation:** Automatically calculates and reports BERTScore F1 for fused answers against ground-truth references.

-----

## How it Works

The algorithm orchestrates a multi-step process for each conversational turn:

1.  **Candidate Generation:** For a given question in a conversation, responses are generated in parallel from all configured LLMs (e.g., Mistral, Llama3, Gemma). Crucially, each LLM maintains its own conversational context to ensure coherent follow-up responses.
2.  **Ranking:** All LLMs are prompted to rank the generated responses from *all* participating LLMs. This involves a specialized ranking prompt that presents each candidate response with a unique identifier.
3.  **Aggregation:** The individual rankings from each LLM are then aggregated using the Borda Count method. This method assigns points based on rank position (e.g., 10 points for 1st, 9 for 2nd, etc.), providing a robust overall consensus ranking.
4.  **Selection:** The responses from the top 3 LLMs, as determined by the Borda Count, are chosen as the best candidates.
5.  **Fusion:** The single LLM that achieved the highest aggregate rank (the "best LLM") is tasked with taking these top 3 candidate responses and synthesizing them into a single, refined, and comprehensive final answer using a dedicated fusion prompt.

This iterative process continues for each turn of the conversation, leveraging the strengths of multiple models while dynamically adapting based on their perceived performance.

-----

## Setup and Installation

### Prerequisites

  * **Python 3.8+**
  * **Ollama:** Ensure Ollama is installed and running, and that the specified LLMs are pulled. You can download Ollama from [ollama.com](https://ollama.com/).
  * **Hugging Face `datasets` library:** For loading the DoQA dataset.

### Install Dependencies

First, create a virtual environment and install the required Python packages:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install numpy llm_blender requests evaluate datasets
```

### Pull LLMs with Ollama

The `LLMBlenderExperiment` class includes a `install_llms_parallel` method to help with this. However, you can also manually pull them via Ollama:

```bash
ollama pull mistral
ollama pull llama3
ollama pull gemma:2b
ollama pull phi3
ollama pull qwen:4b
ollama pull deepseek-llm
ollama pull stablelm2
```

Ensure Ollama is running before executing the script (`ollama run`).

-----

## Usage

To run the full suite of experiments, execute the `main.py` script:

```bash
python main.py
```

The script will:

1.  Set up necessary directories.
2.  Load or sample conversations from the **DoQA dataset**.
3.  Check for and generate individual LLM responses if they don't already exist (this can be time-consuming for the first run).
4.  Initialize the LLM Blender models (PairRM ranker and gen\_fuser\_3b).
5.  Run each of the defined experiment policies.
6.  Save results (fused answers, timing logs, BERTScore) to the `Experiment_Results` directory.
7.  Print a summary of BERTScore results for all policies.

-----

## Configuration

You can customize the following parameters in the `LLMBlenderExperiment` class:

  * `self.llm_list`: A list of LLM models to use. Ensure these models are available via your Ollama instance.
  * `self.api_url`: The endpoint for your Ollama API (default: `'http://127.0.0.1:11434/api/generate'`).
  * `self.results_dir`: Directory for saving experiment results.
  * `self.testing_dir`: Directory for saving individual LLM outputs.
  * `self.doqa_dataset_path`: The local file path where the sampled DoQA conversations will be stored (default: `'doqa_conversations.json'`).

-----

## Experiments and Policies

The framework evaluates three distinct blending policies in llm blending, llm blending with k=1 and peer reviewed blending

-----

## Evaluation

The primary evaluation metric used is **BERTScore F1**. For each experiment policy, the generated fused answers are compared against the ground-truth answers from the DoQA dataset. The average BERTScore F1 is calculated for all turns across all sampled conversations.

  * **Fusing Strategy Performance:** Measures how well the blended output matches the reference.
  * **Timing Analysis:** Logs granular timing data for LLM generation, ranking, and fusion steps, allowing for an analysis of the computational efficiency of each policy.

-----

## Directory Structure

```
.
├── main.py                     # Main script to run experiments
├── input.json                  # Ollama API request template
├── doqa_conversations.json     # Sampled DoQA conversations (generated by script)
├── Experiment_Results/         # Stores evaluation results and timing logs
│   ├── fused_answers_*.txt     # Fused answers for each policy
│   ├── timing_*.json           # Detailed timing logs for each policy
│   └── bertscore_*.txt         # BERTScore results for each policy
└── testing/                    # Stores individual LLM outputs
    ├── op_mistral.txt          # Responses from Mistral for all turns
    ├── op_llama3.txt           # Responses from Llama3 for all turns
    └── ...                     # And so on for other LLMs
```

# Multi-LLM Response Blending with Rank-Based Fusion

This repository implements a robust algorithm for generating high-quality, synthesized responses by combining the strengths of multiple Large Language Models (LLMs). It employs a novel self-ranking mechanism, a democratic rank aggregation approach, and a final fusion step by the most capable LLM in the ensemble.

-----

## Overview

The core idea is to harness the diverse capabilities of multiple LLMs to produce a more comprehensive and accurate answer than any single LLM might generate on its own. This is particularly useful in scenarios where different LLMs might excel at different aspects of a query (e.g., factual recall, creative writing, summarization).

-----

## Algorithm Steps

The process unfolds in six distinct stages:

### 1\. Parallel Candidate Generation

For any given input query, responses are generated concurrently from a pre-defined set of ten distinct LLMs. This parallel processing is efficiently handled using the **Hugging Face `parallel` library**. Each LLM produces its own unique answer to the query.

### 2\. Self-Ranking by All LLMs

All ten LLMs are then tasked with ranking the responses of *all* other LLMs (including their own).

  * A specific prompt is constructed for this purpose. This prompt concatenates the original question with all ten generated responses, each clearly labeled (e.g., "1: [LLM1\_Response], 2: [LLM2\_Response], ...").
  * Each LLM uses the prompt template defined in `ranking-prompt.json` to evaluate the quality of the candidate responses and produce its own ranked list.

### 3\. Rank Aggregation (Borda Count)

To consolidate the individual rankings from each LLM, the **Borda Count method** is applied.

  * In Borda Count, points are assigned to each candidate response based on its position in every LLM's ranked list (e.g., if there are 10 LLMs, 1st place gets 10 points, 2nd gets 9, and so on).
  * The total Borda score for each candidate response is calculated, yielding a robust, aggregated ranking that reflects the collective preference of the entire LLM ensemble.

### 4\. Top-3 Candidate Selection

Based on the aggregated Borda Count scores, the responses from the **top three performing LLMs** are selected. These are considered the strongest candidates for the final synthesized answer.

### 5\. Final Fusion by Best LLM

The LLM that achieved the highest overall Borda Count score (i.e., was deemed the "best" by the ensemble) is then designated as the "fusion LLM" for that specific query.

  * The three selected candidate responses (from step 4) are passed to this best LLM.
  * This fusion LLM uses a specialized prompt template (defined in `fusion-prompt.json`) to synthesize, summarize, and refine these top-tier responses into a single, cohesive, and high-quality final answer.

### 6\. Final Answer

The summarized output from the "best LLM" in the fusion step is returned as the ultimate answer to the user's original query.

-----

## Installation

(Assuming Python and Hugging Face libraries are already set up)

1.  Clone this repository:
    ```bash
    git clone [repository_url]
    cd [repository_name]
    ```
2.  Install required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
    (Ensure `requirements.txt` includes `transformers`, `accelerate`, and any other necessary libraries for your LLMs and parallel processing.)

-----

## Usage

To run the algorithm, you'll need to define your LLMs and provide the prompt templates.

1.  **Configure LLMs:**

      * Specify the paths or Hugging Face model IDs for the ten LLMs you want to use in your ensemble.
      * Ensure your environment has access to these models (e.g., pre-downloaded, or accessible via Hugging Face Hub with appropriate authentication if needed).

2.  **Define Prompt Templates:**

      * `ranking-prompt.json`: This file should contain the template for the LLM self-ranking prompt. It should guide the LLMs on how to evaluate and rank the provided responses.
      * `fusion-prompt.json`: This file should contain the template for the final summarization/fusion prompt. It should instruct the best LLM on how to combine the top 3 responses into a single, coherent answer.

    *(Example structure for prompt JSONs will be provided in a separate section or example files.)*

3.  **Run the Algorithm:**
    (A hypothetical example of how you might call the main function)

    ```python
    from your_module import run_multi_llm_blender

    question = "What are the benefits of renewable energy?"
    final_answer = run_multi_llm_blender(question, llm_configs, ranking_prompt_path, fusion_prompt_path)

    print("Final Answer:", final_answer)
    ```

-----

## Prompt Templates

### `ranking-prompt.json` Example

```json
{
  "template": "Here is a question: \"{question}\"\n\nBelow are ten different answers from various AI models. Please rank these answers from best (1) to worst (10) based on their accuracy, completeness, coherence, and relevance to the question. Provide your ranking as a comma-separated list of answer numbers.\n\n1: {response1}\n2: {response2}\n3: {response3}\n4: {response4}\n5: {response5}\n6: {response6}\n7: {response7}\n8: {response8}\n9: {response9}\n10: {response10}\n\nMy ranking (e.g., 1,3,2,4...):"
}
```

### `fusion-prompt.json` Example

```json
{
  "template": "You are the top-performing AI. Here is the original question: \"{question}\"\n\nBelow are the top three responses from other advanced AI models for this question. Your task is to synthesize these three responses into a single, comprehensive, and concise answer. Highlight the key points and ensure accuracy.\n\nTop Response 1: {best_response1}\nTop Response 2: {best_response2}\nTop Response 3: {best_response3}\n\nMy synthesized answer:"
}
```

-----

## Contributing

We welcome contributions to improve this algorithm\! Please feel free to open issues or submit pull requests.

-----

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
