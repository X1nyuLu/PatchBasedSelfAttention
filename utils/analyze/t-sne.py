from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
sys.path.append("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/FinalResult/code")
from utils.DatasetDataLoader import split_smiles
from utils.analyze.classifyMoles import main as class_main

import torch
from torch.nn.functional import pad
import os
import chemplot

# Example SMILES strings
# smiles_list = ['CCO', 'OCC', 'CCN', 'C(C)C', 'CCC', 'C=C', 'C#C', 'CC(O)C(=O)O', 'CC(C)C', 'C1CCCCC1']
# data = pd.read_pickle("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/modified_ViT/30data.pkl")
# ori_smi_list = data["smiles"]

file_path = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/FinalResult/0directly_train/withFormula/ps64_bs128/accum1/translate_test"
save_path = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/FinalResult/code/utils/analyze/result"
smiles_dist = class_main(inference_path=os.path.join(file_path, "BeamSearch_BW{}_NB{}_result.txt".format(10, 10)),
                        tgt_path=os.path.join(file_path, 'tgt.txt'),
                        save_path=save_path,
                        n_beams=10)
dataset = {'smiles':[], 'target':[]}
for c in smiles_dist:
    for smi in smiles_dist[c]:
        dataset["smiles"].append(smi)
        dataset["target"].append(c)
print(dataset)


"""
# fingerprints = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smile), 2, nBits=1024) for smile in smiles_list]
# Convert the fingerprints to a numpy array
# np_fps = np.array([list(fp) for fp in fingerprints])

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=5.0, learning_rate=200.0, random_state=42)
tsne_results = tsne.fit_transform(smi_array_list)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1])

# for i, txt in enumerate(ori_smi_list):
#     plt.annotate(txt, (tsne_results[i, 0], tsne_results[i, 1]))

# plt.title('t-SNE visualization of SMILES')
# plt.xlabel('t-SNE dimension 1')
# plt.ylabel('t-SNE dimension 2')
plt.show()
# plt.savefig('test.png')
"""