from ctypes import ArgumentError
from typing import Dict, List, Tuple

import pandas as pd
import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
import os
import time




def strip(data: List[str]) -> List[str]:
    return [smiles.replace(" ", "") for smiles in data]


def load_data(
    inference_path: str, tgt_path: str, n_beams: int = 10
) -> Tuple[List[List[str]], List[str]]:
    # Load tgt
    with open(tgt_path, "r") as f:
        tgt = f.readlines()
    tgt = strip(tgt)

    # Load preds
    with open(inference_path, "r") as f:
        preds = f.readlines()
    preds_stripped = strip(preds)
    pred_batched = [
        preds_stripped[i * n_beams : (i + 1) * n_beams]
        for i in range(len(preds_stripped) // n_beams)
    ]

    return pred_batched, tgt


def match_group(mol: Chem.Mol, func_group) -> int:
    if type(func_group) == Chem.Mol:
        n = len(mol.GetSubstructMatches(func_group))
    else:
        n = func_group(mol)
    return 0 if n == 0 else 1



def match_smiles(
    tgt_smiles: str, pred_smiles: list, suffix: str = ""
) -> Dict[str, int]:
    if len(pred_smiles) == 0:
        return {
            "top1" + suffix: 0,
            "top5" + suffix: 0,
            "top10" + suffix: 0,
        }

    if tgt_smiles == pred_smiles[0]:
        return {"top1" + suffix: 1, "top5" + suffix: 1, "top10" + suffix: 1}
    elif len(pred_smiles) < 5 and tgt_smiles in pred_smiles:
        return {"top1" + suffix: 0, "top5" + suffix: 1, "top10" + suffix: 1}
    elif tgt_smiles in pred_smiles[:5]:
        return {"top1" + suffix: 0, "top5" + suffix: 1, "top10" + suffix: 1}
    elif tgt_smiles in pred_smiles:
        return {"top1" + suffix: 0, "top5" + suffix: 0, "top10" + suffix: 1}
    else:
        return {"top1" + suffix: 0, "top5" + suffix: 0, "top10" + suffix: 0}


def score(preds: List[List[str]], tgt: List[str]) -> pd.DataFrame:
    results = dict()

    for pred_smiles, tgt_smiles in tqdm.tqdm(zip(preds, tgt), total=len(tgt)):
        try:
            mol = Chem.MolFromSmiles(tgt_smiles)
            if mol is None: raise Exception("Invalid tgt smiles","Mol is None")
            tgt_smiles = Chem.MolToSmiles(mol)
        except Exception as e:
            print("{}: {}".format(tgt_smiles,e) )
            continue

        hac = rdMolDescriptors.CalcNumHeavyAtoms(mol)
        pred_smiles_canon = [None] * len(pred_smiles)

        for i, pred in enumerate(pred_smiles):
            try:
                pred_mol = Chem.MolFromSmiles(pred)

                if pred_mol is None:
                    raise ArgumentError("Invalid Smiles")

                pred_smiles_canon[i] = Chem.MolToSmiles(pred_mol)

            except ArgumentError:
                continue
            except Chem.rdchem.AtomValenceException:
                print("pred: ",pred_smiles)
                print("target: ",tgt_smiles)
                continue

        results[tgt_smiles] = {
            "hac": hac,
            "predictions": pred_smiles_canon,
        }

        # Score match of the smiles string
        results_smiles = match_smiles(tgt_smiles, pred_smiles_canon)
        results[tgt_smiles].update(results_smiles)

    return pd.DataFrame.from_dict(results, orient="index")


def main(inference_path: str, tgt_path: str, save_path: str, n_beams: int = 10, ):
    RDLogger.DisableLog("rdApp.*")

    preds, tgt = load_data(
        inference_path=inference_path, tgt_path=tgt_path, n_beams=n_beams
    )
    results_score = score(preds, tgt)

    with open(os.path.join(save_path, 'score_result.txt'), 'w') as f:
        f.write(time.strftime("%X_%d_%b")+'\n')
        f.write("Results: Smiles Match\n")
        f.write(
            "Top 1: {:.3f}, Top 5: {:.3f}, Top 10: {:.3f}\n".format(
                results_score["top1"].sum() / len(results_score) * 100,
                results_score["top5"].sum() / len(results_score) * 100,
                results_score["top10"].sum() / len(results_score) * 100,
            )
        )



if __name__ == "__main__":
    main()
