import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from scipy import interpolate
from jcamp import jcamp_readfile
from tqdm import tqdm

from utils.DatasetDataLoader import split_formula, split_smiles


# Get Spectra from casID
def readjdx(jdxfile):
    jcampdict = jcamp_readfile(jdxfile)

    if 'xunits' not in jcampdict: xunit = None
    else: xunit = jcampdict['xunits']
    if 'yunits' not in jcampdict: yunit = None
    else: yunit = jcampdict['yunits']

    x = jcampdict['x']
    y = jcampdict['y']

    return x,y,xunit,yunit

def getValidData(raw_ir_dir, exp_data_info, smi_vocab, formula_vocab, save_path):
    """
    valid data: 
        - both tokens of molecular smiles and formula are in the corresponding vocab.
        - x unit: wavenumbers, y unit: absorbance
    """

    formula = []
    smiles = []
    casID = []
    spectra = []

    for i in range(exp_data_info.shape[0]):
        row = exp_data_info.iloc[i]
        
        if row["IR"] == True:
            try:
                smi_ = split_smiles(row["SMILES"])
                f = rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(row["SMILES"]))
                formula_ = split_formula(f)
            except TypeError as e:
                continue
            
            # Exclude stereoisomers
            if row["SMILES"] in smiles: 
                index = smiles.index(row["SMILES"])
                print('repeat molecule: {} | ori cas id: {} | new cas id: {}'.format(row["SMILES"],casID[index],row["casID"]))

                smiles.pop(index)
                formula.pop(index)
                casID.pop(index)
                spectra.pop(index)
                continue
            
            # Skip molecule with its SMILES or formula not in those vocabs.
            skip_row = False
            for token in smi_:
                if token not in smi_vocab:
                    skip_row = True
                    break
            for token in formula_:
                if token not in formula_vocab:
                    skip_row = True
                    break
            if skip_row: continue

            # Unify units of spectra.
            id = row["casID"]
            jdxfile = os.path.join(raw_ir_dir, "{}.jdx".format(id))
            x,y,xunit,yunit = readjdx(jdxfile)

            if xunit in ['1/CM','1/cm','cm-1']:
                pass
            elif xunit == 'MICROMETERS':
                x = 1 / (x * 1e-4)
            else: 
                continue

            if yunit == 'TRANSMITTANCE':
                y = np.log10(1/y)
            elif yunit in ['ABSORBANCE', "absorption index"]:
                pass
            else: 
                continue

            if x[0] > x[-1]:
                x = x[::-1]
                y = y[::-1]

            try:
                func = interpolate.interp1d(x, y, kind='slinear')
                newx = np.arange(400, 3982, 2)
                newy = np.zeros(newx.shape)
                posWithinX = np.where( (newx >= x[0]) & (newx <= x[-1]) )
                newy[posWithinX]= func(newx[posWithinX])
            except ValueError:
                continue
                
            formula.append(f)
            smiles.append(row["SMILES"])
            casID.append(id)
            spectra.append(newy)
    valid_ir_data = pd.DataFrame({'formula': formula, 'smiles': smiles, 'casID': casID, 'spectra': spectra})
    valid_ir_data[['formula', 'smiles', 'casID']].to_csv(os.path.join(save_path, "validExpData.csv"))
    valid_ir_data.to_pickle(os.path.join(save_path, "validExpData.pkl")) # .pkl file contains spectra data
    return valid_ir_data

def getBothData(sim_data, exp_valid_data, save_path):

    exp_valid_smi = exp_valid_data['smiles']
    sim_smi = sim_data['smiles']
    
    both_data_info = {'formula': [], 'smiles': [], 'casID': [], 'spectra': [], 'sim_spectra': []}
    for smi in tqdm(exp_valid_smi):
        try:
            mol = Chem.MolFromSmiles(smi)
            can_smi = Chem.MolToSmiles(mol)
        except TypeError as e:
            continue
        if not sim_smi[sim_smi==can_smi].empty:
            data = exp_valid_data[exp_valid_data['smiles'] == smi]

            try:
                both_data_info['smiles'].append(smi)
                both_data_info['formula'].append(data['formula'].item())      
                both_data_info['casID'].append(data['casID'].item())
                both_data_info['spectra'].append(data['spectra'].item())
                both_data_info['sim_spectra'].append(sim_data[sim_data['smiles']==smi]['spectra'].item())
            except ValueError as e:
                continue

    both_data_info = pd.DataFrame(both_data_info)
    both_data_info[['formula', 'smiles', 'casID']].to_csv(os.path.join(save_path, "bothExpData.csv"))
    both_data_info.to_pickle(os.path.join(save_path, "bothExpData.pkl")) # .pkl file contains sim and exp spectra data
