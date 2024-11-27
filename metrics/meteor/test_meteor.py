import json
from meteor import Meteor
import os

file_path = 'inference.json'

def load_dictionaries(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    print(f"Dictionaries loaded from {file_path}.")
    return data["ref_dict"], data["cand_dict"]

# Load the dictionaries back
ref_dict, cand_dict = load_dictionaries(file_path)

meteor = Meteor()
score, scores = meteor.compute_score(ref_dict, cand_dict)
print(f'METEOR score: {score}')
