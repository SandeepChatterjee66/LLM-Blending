import llm_blender
from llm_blender.blender.blender_utils import get_topk_candidates_from_ranks
import requests
import json
import time
import os
import random
import concurrent.futures
from datasets import load_dataset  # Add this line to import the necessary function

def install_llms_parallel(llm_list) -> list:
    print('#########################    Installing LLMS in Parallel    #########################')

    def pull_llm(llm):
        os.system(f'ollama pull {llm}')
        print(f'{llm} installed')

    # Use ThreadPoolExecutor for parallel pulling
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(pull_llm, llm_list)

    return llm_list


def load_inputs_from_file() -> list:
    """Load input questions from the existing input file."""
    if os.path.exists('input_questions.txt'):
        print("Loading input questions from input_questions.txt")
        with open('input_questions.txt', 'r') as f:
            inputs = [line.strip() for line in f.readlines()]
        return inputs
    else:
        return None


def dataset_init() -> list:
    """Initialize the dataset by loading the questions."""
    print('#########################    DatasetInit    #########################')
    dataset = load_dataset("conv_questions")

    # Get a sample of input questions
    test = random.sample(list(dataset['test']), 1)
    inputs = [test[i]['questions'] for i in range(len(test))]

    # Save the input questions to a file
    with open('input_questions.txt', 'w') as f:
        for input_set in inputs:
            f.write(f"{input_set}\n")

    return inputs


def lone_llm_output_parallel(llm, inputs) -> list:
    print(f'#########################    Lone LLM Outputs for {llm}    #########################')

    # Check if the output file already exists
    filename = f'testing/op_{llm.replace("/","-")}.txt'
    if os.path.exists(filename):
        print(f'File "{filename}" already exists. Skipping LLM response generation for {llm}.')
        with open(filename, 'r') as f:
            outputs = [line.strip() for line in f.readlines()]
        return outputs

    # If file does not exist, proceed with generating responses
    json_str = "{\"model\":\"\",\"prompt\":\"\",\"stream\":false}"
    with open('input.json', 'w') as f:
        f.write(json_str)

    url = 'http://127.0.0.1:11434/api/generate'

    def get_response(prompt):
        request_data = {
            "model": llm,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(url, json=request_data)
        print(response)
        assert response.status_code == 200
        response_data = json.loads(response.text)
        return response_data['response']

    # Use ThreadPoolExecutor for parallel LLM requests
    outputs = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        op = list(executor.map(get_response, inputs))
        outputs.extend(op)

    # Writing outputs to file
    with open(filename, 'w') as f:
        for line in outputs:
            f.write(f"{line}\n")

    return outputs


def load_candidates_from_file(llm):
    """Load candidates from the existing output file."""
    filename = f'testing/op_{llm.replace("/","-")}.txt'
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            candidates = [line.strip() for line in f.readlines()]
        print(f"Loaded candidates from {filename}.")
        return candidates
    else:
        print(f"File {filename} does not exist.")
        return []


def safe_concatenate(candidates):
    # Ensure all candidates are valid strings, remove empty candidates
    return [str(candidate).strip() if candidate else "" for candidate in candidates]


def llm_blender_nonConv(llm_list, inputs):
    print('######################### Non-Conversational LLM Blender #########################')

    blender = llm_blender.Blender()
    blender.loadranker("llm-blender/PairRM")
    blender.loadfuser("llm-blender/gen_fuser_3b")

    fused_answers = []
    for i in range(len(inputs)):
        start_time = time.time()
        ans_set = []
        for j in range(len(inputs[i])):
            candidates_texts = []
            for k in range(len(llm_list)):
                print(llm_list[k])
                print(load_candidates_from_file(llm_list[k]))
                # Load candidates from the existing output files
                candidates_texts.append(load_candidates_from_file(llm_list[k])[i][j])


            # Ensure candidates are safe for concatenation
            candidates_texts = safe_concatenate(candidates_texts)

            # Check for empty candidates
            if len(candidates_texts) == 0 or all(candidate == "" for candidate in candidates_texts):
                print(f"Warning: No valid candidates found for question {inputs[i][j]}")
                continue  # Skip the fusion if there are no valid candidates
            
            print(f"Candidates for fusion: {candidates_texts}")
            
            # Rank the candidates
            ranks = blender.rank([inputs[i][j]], [candidates_texts], return_scores=False, batch_size=1)
            print(ranks)
            
            # Get the top candidates based on ranks
            topk_candidates = get_topk_candidates_from_ranks(ranks, [candidates_texts], top_k=2)
            
            # Fuse the top candidates to generate the final answer
            fuse_generations = blender.fuse([inputs[i][j]], [topk_candidates], batch_size=2)
            ans_set.append(fuse_generations)
        
        fused_answers.append(ans_set)

        # Log time
        end_time = time.time()
        print(f"Time taken for question {i + 1}: {end_time - start_time:.2f} seconds")

    # Write to file
    filename = 'op_fused_nonConv.txt'
    with open(filename, 'w') as f:
        for line in fused_answers:
            f.write(f"{line}\n")
    return fused_answers


def llm_blender_Conv(llm_list, inputs):
    print('######################### Conversational LLM Blender #########################')
    blender = llm_blender.Blender()
    blender.loadranker("llm-blender/PairRM")
    blender.loadfuser("llm-blender/gen_fuser_3b")

    fused_answers = []
    url = 'http://127.0.0.1:11434/api/generate'

    for i in range(len(inputs)):
        start_time = time.time()  # Start timer for this question
        ans_set = []
        context = ""
        for j in range(len(inputs[i])):
            question = inputs[i][j] + context
            candidates_texts = []
            for llm in llm_list:
                # Load candidates from the existing output files
                candidates = load_candidates_from_file(llm)
                if candidates:
                    candidates_texts.append(candidates[i][j])
                else:
                    print(f"No candidates found for {llm}. Skipping.")

            # Ensure candidates are safe for concatenation
            candidates_texts = safe_concatenate(candidates_texts)

            # Check for empty candidates
            if len(candidates_texts) == 0 or all(candidate == "" for candidate in candidates_texts):
                print(f"Warning: No valid candidates found for question {inputs[i][j]}")
                continue  # Skip fusion if there are no valid candidates

            start_time_rank = time.time()
            ranks = blender.rank([inputs[i][j]], [candidates_texts], return_scores=False, batch_size=1)
            end_time_rank = time.time()
            print(ranks, end_time_rank - start_time_rank)

            topk_candidates = get_topk_candidates_from_ranks(ranks, [candidates_texts], top_k=2)
            fuse_generations = blender.fuse([inputs[i][j]], topk_candidates, batch_size=2)
            context = context + fuse_generations[0]
            ans_set.append(fuse_generations[0])
        
        fused_answers.append(ans_set)

        # Calculate and print time taken for the current question
        end_time = time.time()
        print(f"Time taken for question {i + 1}: {end_time - start_time:.2f} seconds")

    filename = 'op_fused_Conv.txt'
    with open(filename, 'w') as f:
        for line in fused_answers:
            f.write(f"{line}\n")
    return fused_answers


if __name__ == '__main__':
    print("Hi")

    # List of LLMs
    llm_list = ['mistral', 'llama3.1', 'gemma:2b', 'phi3', 'qwen:4b', 
                'phi', 'tinydolphin', 'deepseek-llm', 'stablelm2', 'dog/arcee-lite']
    
    # Attempt to load inputs from file
    inputs = load_inputs_from_file()

    if inputs is None:
        # Initialize dataset and inputs if input file does not exist
        inputs = dataset_init()

    # Check if outputs exist and skip installation if so
    output_files_exist = all(os.path.exists(f'testing/op_{llm.replace("/", "-")}.txt') for llm in llm_list)

    if not output_files_exist:
        install_llms_parallel(llm_list)

    lone_outputs = []
    for llm in llm_list:
        lone_outputs.append(lone_llm_output_parallel(llm, inputs))

    llm_blender_nonConv(llm_list, inputs)
    llm_blender_Conv(llm_list, inputs)

    print("Bye")
