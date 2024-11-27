import sys
sys.path.append('.') # from root dir
from metrics.bleu.bleu import Bleu
from metrics.rouge.rouge import Rouge
from metrics.cider.cider import Cider
import json

file_path = 'baseline-2/inference.json'

def load_dictionaries(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    print(f"Dictionaries loaded from {file_path}.")
    return data["ref_dict"], data["cand_dict"]

ref_dict, cand_dict = load_dictionaries(file_path)


bleu1 = Bleu(n=1)
bleu1_score, _ = bleu1.compute_score(ref_dict, cand_dict)
print(f'BLEU-1 score: {bleu1_score}')

bleu4 = Bleu(n=4)
bleu4_score, _ = bleu4.compute_score(ref_dict, cand_dict)
print(f'BLEU-4 score: {bleu4_score}')

rouge = Rouge()
rouge_score, _ = rouge.compute_score(ref_dict, cand_dict)
print(f'ROUGE score: {rouge_score}')

cider = Cider()
cider_score = cider.compute_score(ref_dict, cand_dict)
print(f'CIDEr score: {cider_score}')