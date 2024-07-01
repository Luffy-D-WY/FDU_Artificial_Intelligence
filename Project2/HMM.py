import numpy as np
import os
import random

# Define tags for English and Chinese
english_tags = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

chinese_tags = [
    'O', 'B-NAME', 'M-NAME', 'E-NAME', 'S-NAME',
    'B-CONT', 'M-CONT', 'E-CONT', 'S-CONT',
    'B-EDU', 'M-EDU', 'E-EDU', 'S-EDU',
    'B-TITLE', 'M-TITLE', 'E-TITLE', 'S-TITLE',
    'B-ORG', 'M-ORG', 'E-ORG', 'S-ORG',
    'B-RACE', 'M-RACE', 'E-RACE', 'S-RACE',
    'B-PRO', 'M-PRO', 'E-PRO', 'S-PRO',
    'B-LOC', 'M-LOC', 'E-LOC', 'S-LOC'
]

# Load data from file
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    observations = []
    states = []

    for line in lines:
        if not line.strip():
            continue
        observation, state = line.strip().split()
        observations.append(observation)
        states.append(state)

    return observations, states

# Initialize parameters for HMM
def initialize_parameters(tags):
    n_tags = len(tags)
    tag_to_index = {tag: i for i, tag in enumerate(tags)}

    transition_matrix = np.zeros((n_tags, n_tags))
    emission_matrix = {}
    initial_probabilities = np.zeros(n_tags)

    return transition_matrix, emission_matrix, initial_probabilities, tag_to_index

# Train the HMM model
def train_hmm(file_path, tags):
    transition_matrix, emission_matrix, initial_probabilities, tag_to_index = initialize_parameters(tags)
    seen_words = []

    with open(file_path, "r", encoding='utf-8') as f:
        lines = f.readlines()
    previous_tag = None
    for line in lines:
        line = line.strip()

        if line:
            word, tag = line.split()
            tag_index = tag_to_index[tag]

            if word not in emission_matrix:
                emission_matrix[word] = np.zeros(len(tag_to_index))
            emission_matrix[word][tag_index] += 1
            seen_words.append(word)

            if previous_tag is None:
                initial_probabilities[tag_index] += 1
            else:
                previous_tag_index = tag_to_index[previous_tag]
                transition_matrix[previous_tag_index][tag_index] += 1

            previous_tag = tag
        else:
            # Reset previous_tag to None for the next sentence
            previous_tag = None

    # Normalize initial probabilities
    initial_probabilities = initial_probabilities / np.sum(initial_probabilities)
    initial_probabilities = np.log(initial_probabilities + 1e-10)  # Avoid log(0)

    # Normalize transition matrix
    transition_matrix = np.divide(transition_matrix, np.sum(transition_matrix, axis=1, keepdims=True), out=np.zeros_like(transition_matrix), where=np.sum(transition_matrix, axis=1, keepdims=True)!=0)
    transition_matrix = np.log(transition_matrix + 1e-10)  # Avoid log(0)

    # Normalize emission matrix
    for word in emission_matrix:
        emission_matrix[word] = emission_matrix[word] / np.sum(emission_matrix[word])
        emission_matrix[word] = np.log(emission_matrix[word] + 1e-10)  # Avoid log(0)

    # Calculate average emission probabilities
    avg_emission_probabilities = np.zeros(len(tag_to_index))
    for word in emission_matrix:
        avg_emission_probabilities += np.exp(emission_matrix[word])  # Convert back from log-space
    avg_emission_probabilities = avg_emission_probabilities / len(emission_matrix)
    avg_emission_probabilities = np.log(avg_emission_probabilities + 1e-10)  # Convert back to log-space

    return transition_matrix, emission_matrix, initial_probabilities, tag_to_index, seen_words, avg_emission_probabilities

# Viterbi algorithm for decoding
def viterbi(observation_sequence, tags, transition_matrix, emission_matrix, initial_probabilities, avg_emission_probabilities):
    sequence_length = len(observation_sequence)
    num_tags = len(tags)

    dp = np.zeros((num_tags, sequence_length))
    path = np.zeros((num_tags, sequence_length), dtype=int)

    for i in range(num_tags):
        word = observation_sequence[0]
        if word in emission_matrix:
            dp[i][0] = initial_probabilities[i] + emission_matrix[word][i]
        else:
            dp[i][0] = initial_probabilities[i] + avg_emission_probabilities[i]

    for t in range(1, sequence_length):
        for i in range(num_tags):
            word = observation_sequence[t]
            if word in emission_matrix:
                probability = dp[:, t-1] + transition_matrix[:, i] + emission_matrix[word][i]
            else:
                probability = dp[:, t-1] + transition_matrix[:, i] + avg_emission_probabilities[i]
            dp[i][t] = np.max(probability)
            path[i][t] = np.argmax(probability)

    tags_sequence = np.zeros(sequence_length, dtype=int)
    tags_sequence[sequence_length-1] = np.argmax(dp[:, sequence_length-1])

    for t in range(sequence_length-2, -1, -1):
        tags_sequence[t] = path[tags_sequence[t+1], t+1]

    return [tags[i] for i in tags_sequence]

# Decode validation set and save results
def decode_and_save_results(input_file, output_file, tags, transition_matrix, emission_matrix, initial_probabilities, avg_emission_probabilities):
    with open(input_file, "r", encoding='utf-8') as f:
        lines = f.readlines()

    observations = []
    original_lines = []
    results = []
    for line in lines:
        line = line.strip()
        if line:
            word, _ = line.split()
            observations.append(word)
            original_lines.append(line)
        else:
            if observations:
                predicted_tags = viterbi(observations, tags, transition_matrix, emission_matrix, initial_probabilities, avg_emission_probabilities)
                for line, tag in zip(original_lines, predicted_tags):
                    word, _ = line.split()
                    results.append(f"{word} {tag}\n")
                results.append("\n")
            observations = []
            original_lines = []

    if observations:  # Handle the last sentence in the file
        predicted_tags = viterbi(observations, tags, transition_matrix, emission_matrix, initial_probabilities, avg_emission_probabilities)
        for line, tag in zip(original_lines, predicted_tags):
            word, _ = line.split()
            results.append(f"{word} {tag}\n")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding='utf-8') as f:
        f.writelines(results)

if __name__ == "__main__":
    # Process Chinese dataset
    chinese_train_file = 'Chinese/train.txt'
    chinese_val_file = 'Chinese/test.txt'
    chinese_output_file = 'example_data/1chinese_my_result.txt'

    # Process English dataset
    english_train_file = 'English/train.txt'
    english_val_file = 'English/test.txt'
    english_output_file = 'example_data/1english_my_result.txt'

    # Train Chinese HMM model
    chinese_transition_matrix, chinese_emission_matrix, chinese_initial_probabilities, chinese_tag_to_index, chinese_seen_words, chinese_avg_emission_probabilities = train_hmm(chinese_train_file, chinese_tags)
    decode_and_save_results(chinese_val_file, chinese_output_file, chinese_tags, chinese_transition_matrix, chinese_emission_matrix, chinese_initial_probabilities, chinese_avg_emission_probabilities)

    # Train English HMM model
    english_transition_matrix, english_emission_matrix, english_initial_probabilities, english_tag_to_index, english_seen_words, english_avg_emission_probabilities = train_hmm(english_train_file, english_tags)
    decode_and_save_results(english_val_file, english_output_file, english_tags, english_transition_matrix, english_emission_matrix, english_initial_probabilities, english_avg_emission_probabilities)
