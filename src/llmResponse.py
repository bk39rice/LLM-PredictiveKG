import subprocess
import json
import re
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, KnowledgeGraphIndex
from llama_index.core.graph_stores import SimpleGraphStore
import os
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.llms.ollama import Ollama

def extract_and_save_single_article(source, target):
    # Open and read the source file
    with open(source, 'r') as source_file:
        try:
            # Parse the file content as JSON
            json_data = json.load(source_file)

            # Extract the 'article' section
            article_content = json_data['article']

            # Save the article content to a new file in the target directory
            with open(target, 'w') as target_file:
                target_file.write(article_content)
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file")

def extract_and_save_articles(source_directory, target_directory):
    # Ensure the target directory exists
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    
    # Iterate over all files in the source directory
    for filename in os.listdir(source_directory):
        if filename.endswith('.txt'):  # Check if the file is a .txt file
            source_file_path = os.path.join(source_directory, filename)
            target_file_path = os.path.join(target_directory, filename)

            # Open and read the source file
            with open(source_file_path, 'r') as source_file:
                try:
                    # Parse the file content as JSON
                    json_data = json.load(source_file)

                    # Extract the 'article' section
                    article_content = json_data['article']

                    # Save the article content to a new file in the target directory
                    with open(target_file_path, 'w') as target_file:
                        target_file.write(article_content)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from file: {filename}")

def generate_response(prompt):
    curl_command = f"""curl -s http://localhost:11434/api/generate -d '{{"model": "orca-mini", "prompt":"{prompt}"}}'"""
    
    process = subprocess.Popen(curl_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    full_response = ""


    while True:
        output_line = process.stdout.readline()
        if not output_line and process.poll() is not None:
            break
        if output_line:
            try:
                response_data = json.loads(output_line.strip())
                full_response += response_data.get("response", "")
            except json.JSONDecodeError:
                return "Invalid response format", 500


    return full_response


def get_user_input_and_generate():
    prompt = input("Enter a prompt: ")
    response = generate_response(prompt)
    print("Response:", response)

def formatString(string):
    string = string.replace("\n", "")
    string = string.replace("\'", "")
    string = string.replace("\"", "")
    string = string.encode('ascii', errors = "ignore").decode()
    return string

def formatOptions(options):
    toReturn = ""
    for i in range(len(options)):
        toReturn += f'({i}) {formatString(options[0])}\\n'
    return toReturn

def shortStoryAccuracy(filePath):
    # parse the question, answer choices, answers, and the article
    pattern = r'([ABCD])'
    with open(filePath, 'r') as file:
        json_data = json.load(file)
    
    answers = json_data['answers']
    options = json_data['options']
    article = formatString(json_data['article'])
    questions = json_data['questions']

    # form the prompt for LLM
    hallucinate = 0
    responses = []
    for i in range(len(questions)):
        prompt = f"From the article:\\n{article}: \\n\\n{formatString(questions[i])}??\\n{formatOptions(options[i])}\\nWhat is the answer? Respond with only the nu"
        response = generate_response(prompt)
        match = re.findall(pattern, response)
        if match:
            responses.append(match[0])
        else:
            responses.append('X')
            print(f'ERROR: Invalid response: {response}')
            hallucinate += 1
    correct = 0
    total = 0
    for i in range(len(responses)):
        for j in range(len(responses[i])):
            if responses[i][j] == answers[i][j]:
                correct += 1
            total += 1
    
    return correct, total, hallucinate

def batchResults(filepath):
    directory_path = Path(filepath)

    # Iterate through the files in the specified directory
    correct = 0
    total = 0
    count = 0
    hallucinate = 0
    for file_path in directory_path.iterdir():
        # Check if the current path is a file and not a directory
        step_correct, step_total, step_hallucinate = shortStoryAccuracy(file_path)
        correct += step_correct
        total += step_total
        hallucinate += step_hallucinate
        print(count)
        count += 1
    print(f'Percentage: {correct/total} Correct: {correct} Total: {total} Hallucinated {hallucinate} times')

def kg_query_engine(article_path):
    reader = SimpleDirectoryReader(input_files=[article_path])
    all_docs = reader.load_data()

    # define LLM
    llm = Ollama(model="orca-mini", request_timeout=60.0)
    Settings.llm = llm
    Settings.chunk_size = 512

    graph_store = SimpleGraphStore()
    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    # NOTE: can take a while!
    index = KnowledgeGraphIndex.from_documents(
        all_docs,
        max_triplets_per_chunk=2,
        storage_context=storage_context,
    )
    query_engine = index.as_query_engine(include_text=True, response_mode="tree_summarize")

    return query_engine

def parse_questions(questions):
    aggregated_questions = []
    for question in questions:
        question_map = dict()
        question_map['question'] = question['question']
        question_map['options'] = []
        for option in question['options']:
            question_map['options'].append(option)
        question_map['answer'] = question['gold_label']
        aggregated_questions.append(question_map)
    return aggregated_questions

def ask_questions(questions, query_engine):
    responses = []
    answers = []
    correct = 0
    for i in range(len(questions)):
        prompt = f"From the article: \\n\\n{questions[i]['question']}??\\n{formatOptions(questions[i]['options'])}\\nWhat is the answer? Respond with only the number"
        response = query_engine.query(prompt)

        # check for correctness
        
        # parse the question, answer choices, answers, and the article
        pattern = r'([1234])'

        # form the prompt for LLM
        hallucinate = 0
        match = re.findall(pattern, response)
        if match:
            response.append(match[0])
        else:
            response.append('X')
            print(f'ERROR: Invalid response: {response}')
            hallucinate += 1
        if response[-1] == questions['answer']:
            correct += 1
        
    return correct, len(questions)

def createKnowledgeGraph(dataset_filepath):
    # reader = SimpleDirectoryReader(input_dir=filepath, recursive=True)
    # all_docs = []
    # for docs in reader.iter_data():
    #     # <do something with the documents per file>
    #     print(docs)
    #     all_docs.extend(docs)

    json_data = load_json_by_line(dataset_filepath)
    total_correct = 0
    total_questions = 0
    for single_article in json_data:
        article_id = single_article['article_id']
        article = single_article['article']
        questions = single_article['questions']
        article_path = f'./quality/data/v1.0.1/devArticles/{article_id}.txt'
        if not os.path.exists(article_path):
            with open(article_path, 'w') as f:
                f.write(article)
                f.close()
        parsed_questions = parse_questions(questions)
        query_engine = kg_query_engine(article_path)
        correct, total = ask_questions(parsed_questions, query_engine)
        total_correct += correct
        total_questions += total
    return total_correct, total_questions


def load_json_by_line(filepath):
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            try:
                data.append(json.loads(line))
            except:
                print(f"Error parsing JSON from line")
    return data


createKnowledgeGraph('./quality/data/v1.0.1/QuALITY.v1.0.1.htmlstripped.dev')

# batchResults('./race-c/data/dev/')
# createKnowledgeGraph('./race-c/data/dev/11.txt')
# if __name__ == '__main__':
#     get_user_input_and_generate()



