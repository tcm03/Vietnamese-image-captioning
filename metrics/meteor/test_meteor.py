import json
from meteor import Meteor
import os

file_path = 'inference.json'


def load_dictionaries(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    print(f"Dictionaries loaded from {file_path}.")
    return data["ref_dict"], data["cand_dict"]

# Load the dictionaries back
ref_dict, cand_dict = load_dictionaries(file_path)
# print(f'ref_dict: {dict(list(ref_dict.items())[:2])}')
# print(f'cand_dict: {dict(list(cand_dict.items())[:2])}')

meteor = Meteor()
score = meteor.compute_score(ref_dict, cand_dict)
print(f'METEOR score: {score}')