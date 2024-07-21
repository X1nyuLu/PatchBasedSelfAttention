import pandas as pd
from rdkit import Chem
from tqdm import tqdm
import os
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score


def clsSMILES(smi, fg_list):
    cls_onehot_list = [smi]
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        print("{} are unable to be converted to Chem.Mol.".format(smi))
        return cls_onehot_list + [-1] * len(fg_list)
    else:
        for i, fg in enumerate(fg_list):
            try:
                pattern = Chem.MolFromSmarts(fg)
                match = mol.HasSubstructMatch(pattern)
            except TypeError as e:
                break
            
            if match == True:
                cls_onehot_list.append(1)
            elif match == False:
                cls_onehot_list.append(0)
        return cls_onehot_list

def clsSMILESList(smi_list, fg_list, label_names, save_path, save_name):
    all_cls = []

    for smi in tqdm(smi_list):
        all_cls.append(clsSMILES(smi, fg_list))

    cls_df = pd.DataFrame(all_cls, columns=['smiles']+label_names)
    cls_df.to_csv(os.path.join(save_path,save_name))

def stat_cls_result(cls_dataset, label_names, save_path, save_name):

    sum_s = cls_dataset[label_names].sum()
    portion_s = sum_s/cls_dataset.shape[0]
    lable_count_portion = pd.concat([sum_s, portion_s], axis=1)
    lable_count_portion.columns = ['count', 'portion']
    lable_count_portion.to_csv(os.path.join(save_path,save_name))

def readPredictedResult(file_path):
    with open(file_path) as file:
        smi_list = file.readlines()
    smi_list = [smi.strip() for smi in smi_list]
    smi_list = [''.join(smi.split(' ')) for smi in smi_list]
    return smi_list


def calScores(pred, true):
    score_list = []
    score_list.append(round(f1_score(true, pred),3))
    score_list.append(round(roc_auc_score(true,pred),3))
    score_list.append(round(accuracy_score(true, pred),3))
    score_list.append(round(precision_score(true, pred),3))
    score_list.append(round(recall_score(true, pred),3))
    return score_list