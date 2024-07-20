import torch
import sys
sys.path.append("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/FinalResult/code")
from utils.DatasetDataLoader import generateDataset, CreateDataloader, norm_spectrum, interpolate_spectrum, interpolate
import matplotlib as mpl

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

def vertical_noise(spec_x, spec_y, theta=0.01, alpha=1) -> np.ndarray:
    """
    theta: threshold of points which will have noise
    alpha: extent of noise.
    """
    spec_y = spec_y/max(spec_y) # y value of the highest peak is 1.
    print(spec_y)

    random_values = np.zeros(spec_y.shape)
    indices =  spec_y > theta # Only add noise on peaks, in case it changed the meaning of the sepctrum

    # Uniformly sample random number between [-1, alpha-1)
    random_values[indices] = (np.random.rand(sum(indices))*alpha - 1)
    print(random_values)
    aug_y = spec_y * (1 + random_values)
    print(aug_y)
    print(aug_y/max(aug_y))
    print(aug_y.shape)
    print(spec_x.shape)
    return interpolate_spectrum(aug_y, spec_x, orig_x=spec_x)[0]

def shift_horizontal(spectrum, shift, new_x, orig_x=None) -> np.ndarray:
    # if orig_x.all() == None:
    if orig_x is None:
        orig_x = np.arange(400, 3982, 2)
    shifted_x = orig_x + shift
    interp_func = interpolate.interp1d(shifted_x, spectrum, bounds_error=False, fill_value=0)
    shifted_spectrum = interp_func(new_x)
    return norm_spectrum(shifted_spectrum)

def plotAug(dataset,total_num_per_mole, save_path, save_name):

    createdataloader = CreateDataloader(pad_id=2,
                                        bs_id=0,
                                        eos_id=1,
                                        tgt_max_padding=40,
                                        formula=True, formula_max_pad=15)
    train_dataloader = createdataloader.dataloader(dataset, batch_size=total_num_per_mole, shuffle=False)
    print(len(dataset))
    for batch in train_dataloader:
        print(batch["spec"])
        print(batch['formula'])
        print(batch['smi'])

        s_x = np.linspace(400, 3980, 3200)
        spec0 = np.array(batch['spec'][0,:])
        spec1 = np.array(batch['spec'][1,:])
        spec2 = np.array(batch['spec'][2,:])
        plt.plot(s_x, spec0, label="spec_orig")
        plt.plot(s_x, spec1, label="spec_aug1",alpha=0.7)
        # plt.plot(s_x, new_y, label="spec_new",alpha=0.7)
        plt.plot(s_x, spec2, label="spec_aug2",alpha=0.5)
        plt.legend()
        plt.savefig(os.path.join(save_path,"{}.png".format(save_name)))
        # plt.savefig(os.path.join(save_path,"spec_horizontalAug.png"))
        plt.close()

        break

def plotOneData(smi, orig_spec, save_path):
    new_x = np.linspace(400, 3980, 3200)
    # print(orig_spec.shape)
    orig_spec = interpolate_spectrum(spectrum=orig_spec, new_x=new_x)
    # orig_spec = np.array(interpolate_spectrum(spectrum=orig_spec, new_x=new_x))
    # print('x shape: ',new_x.shape)
    # print('y shape: ',orig_spec.shape)
    shift_spec = shift_horizontal(orig_spec[0], 100, new_x=new_x, orig_x=new_x)
    plot2specIn1fig(new_x, orig_spec[0], 'original_spec', shift_spec,"shift_spec",
                    smi, savename="sim_{}_shift100.png".format(smi), save_path=save_path)
    
    noise_spec = vertical_noise(new_x, orig_spec[0], alpha=10, theta=0.01)
    plot2specIn1fig(new_x, orig_spec[0], 'original_spec', noise_spec, "spec_vertical_noise",
                    smi, savename="sim_{}_noise_alpha10.png".format(smi), save_path=save_path)
    
    noise_shift_spec = shift_horizontal(noise_spec, shift=100, new_x=new_x, orig_x=new_x)
    plot2specIn1fig(new_x, orig_spec[0], 'original_spec', noise_shift_spec, "spec_vertical_noise_shift",
                    smi, savename="sim_{}_noise_alpha10_shift100.png".format(smi), save_path=save_path)
    
    # plt.plot(new_x, orig_spec[0], label='original_spec')
    # plt.plot(new_x, shift_spec, label="shift_spec", alpha=0.7)
    # plt.legend()
    # plt.title(smi)
    # plt.savefig(os.path.join(save_path,))
    # plt.close()

    # plt.plot(new_x, orig_spec[0], label='original_spec')
    # plt.plot(new_x, noise_spec, label="spec_vertical_noise", alpha=0.7)
    # plt.legend()
    # plt.title(smi)
    # plt.savefig(os.path.join(save_path,"sim_{}_noise_alpha10.png".format(smi)))
    # plt.close()

def plot2specIn1fig(x, spec1, label1, spec2, label2, smi, savename, save_path):
    plt.plot(x, spec1, label=label1, color='olivedrab')
    plt.plot(x, spec2, label=label2, color='gold', alpha=0.8) 
    plt.legend()
    plt.title(smi)
    plt.xlabel('Wavenumbers(cm$^{-1}$)')
    plt.ylabel('Absorbance')
    plt.savefig(os.path.join(save_path,savename), transparent=True, dpi=200)
    plt.close()


    

if __name__ == "__main__":
    mol_info = pd.read_pickle("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/FinalResult/code/utils/spec_plot/24544089/mol_info.pkl")
    smi = mol_info.iloc[0]['smiles']
    print(mol_info)
    print(smi)
    print(CalcMolFormula(Chem.MolFromSmiles(smi)))
    orig_x = np.arange(400, 3982, 2)

    sim_spec = mol_info.iloc[0]["sim_spec"]
    # print(max(sim_spec))
    # sim_spec = sim_spec/max(sim_spec)
    sim_spec = interpolate_spectrum(sim_spec, orig_x, orig_x=orig_x)[0]
    print(sim_spec)
    # exp_spec = mol_info.iloc[0]["exp_spec"]
    # exp_spec = exp_spec/max(exp_spec)

    vn_spec = vertical_noise(orig_x,sim_spec,alpha=50)
    print(vn_spec)
    figure(figsize=(12,6))
    # mpl.rc('font',family='Times New Roman')

    # plt.plot(orig_x, vn_spec,color="#ec999b",alpha=0.8)

    # plt.plot(orig_x, vn_spec,color="royalblue",alpha=0.4, label='Simulated spectra')
    plt.plot(orig_x, sim_spec, color='red', label='Simulated spectra with adaptive noise')#,lw=0.8)
    # plt.plot(orig_x, vn_spec,color="#ec999b",alpha=0.8)
    plt.xticks([])
    plt.yticks([])
    # csfont = {'family':'Times New Roman'}
    # plt.legend(prop=csfont)
    # plt.legend()
    plt.savefig('test_ori.png',dpi=200)
    """
    # mol_spec = {}
    # exp_all_info = pd.read_csv("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/FinalResult/experimentalData/alldata.csv")
    # cas_id = 24544089
    # mol_spec["id"] = [24544089]
    # smi = exp_all_info[exp_all_info["casID"]==cas_id]["SMILES"].values
    # smi = str(smi[0])
    # print(smi)
    # mol_spec["smiles"] = [smi]

    # all_valid_data = pd.read_pickle("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/FinalResult/experimentalData/data/0withformula_noAug/all_data.pkl")
    # print(all_valid_data)
    # exp_spec = all_valid_data[all_valid_data['smiles']==smi]['spectra'].values
    # # exp_spec = exp_spec[0]
    # print(exp_spec)
    # mol_spec["exp_spec"] = exp_spec

    # sim_data = pd.read_pickle("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/IRtoMol/data/ir_data.pkl")
    # sim_spec = sim_data[sim_data['smiles']==smi]['spectra'].values
    # print(sim_spec[0])
    # mol_spec["sim_spec"] = sim_spec

    save_path = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/FinalResult/code/utils/spec_plot/24544089"
    # mol_spec_df = pd.DataFrame(mol_spec)
    # print(mol_spec_df)
    # mol_spec_df.to_pickle(os.path.join(save_path, "mol_info.pkl"))
    
    # ir_spec = ""
    mol_info = pd.read_pickle("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/FinalResult/code/utils/spec_plot/24544089/mol_info.pkl")
    smi = mol_info.iloc[0]['smiles']
    print(smi)
    sim_spec = mol_info.iloc[0]["sim_spec"]
    sim_spec = sim_spec/max(sim_spec)
    exp_spec = mol_info.iloc[0]["exp_spec"]
    exp_spec = exp_spec/max(exp_spec)

    orig_x = np.arange(400, 3982, 2)
    vn_spec = vertical_noise(orig_x,sim_spec,alpha=5)
    vn_spec = vn_spec/max(vn_spec)
    hs_spec = shift_horizontal(sim_spec,shift=100, new_x=orig_x, orig_x=orig_x)
    hs_spec = hs_spec/max(hs_spec)

    fig,axs = plt.subplots(3, sharex=True, sharey=True)
    axs[0].plot(orig_x, sim_spec, label="simulated spectrum", color='olivedrab')
    axs[0].plot(orig_x, exp_spec, label="experimental spectrum", color='darkgoldenrod',alpha=0.8)
    axs[0].legend(fontsize=15)
    axs[0].set_ylabel("Absorbance")
    # axs[0].set_title("experimental spectrum")
    # axs[0].set_ylabel("experimental spectrum")
    axs[1].plot(orig_x, sim_spec, label="simulated spectrum", color='olivedrab')
    axs[1].plot(orig_x, vn_spec, label="adding vertical noise", color='goldenrod',alpha=0.8)
    axs[1].legend(fontsize=15)
    axs[1].set_ylabel("Absorbance")
    # axs[1].set_title("adding vertical noise")

    axs[2].plot(orig_x, sim_spec, label="simulated spectrum", color='olivedrab')
    axs[2].plot(orig_x, hs_spec, label="horizontally shift", color='gold',alpha=0.8)
    axs[2].legend(fontsize=15)
    axs[2].set_ylabel("Absorbance")
    # fig.suptitle(smi)
    # axs[2].set_title("horizontally shift")

    plt.xlabel('Wavenumbers(cm$^{-1}$)')

    # plt.ylabel('Absorbance',loc='center')
    # fig.supylabel('Absorbance')

    # plt.plot(x, spec1, label=label1, color='olivedrab')
    # plt.plot(x, spec2, label=label2, color='gold', alpha=0.8) 
    # plt.legend()
    # plt.title(smi)
    # plt.xlabel('Wavenumbers(cm$^{-1}$)')
    # plt.ylabel('Absorbance')
    fig.set_size_inches(8,14)
    fig.tight_layout()
    plt.savefig(os.path.join(save_path,'3plots.png'), transparent=True, dpi=200)
    plt.close()
    
    # plot2specIn1fig(orig_x, sim_spec, "simulated spectrum", exp_spec, "experimental spectrum", smi, 
                    # "sim_exp.png", save_path)
    # plot2specIn1fig(orig_x, sim_spec, "simulated spectrum", vn_spec, "adding vertical noise", smi, 
                    # "sim_vn.png", save_path)
    # plot2specIn1fig(orig_x, sim_spec, "simulated spectrum", hs_spec, "horizontally shift", smi, 
                    # "sim_hs.png", save_path)
    
    # dataset = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/FinalResult/3with_experimental_data/directlyTrain/noise/5_alpha/aug2/data/val_set.pt"
    # dataset = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/FinalResult/2data_aug/patch16_spec200/horizontalShift/100_shift/train/data/val_set.pt"
    # dataset = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/FinalResult/3with_experimental_data/directlyTrain/noise_smiaug/noiseaug2-smiaug1/data/val_set.pt"
    # dataset = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/FinalResult/3with_experimental_data/directlyTrain/noise_smiaug/aug2-total9-2/data/val_set.pt"
    # dataset = torch.load(dataset)

    smi_vocab = torch.load("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/FinalResult/0directly_train/withFormula/ps8_bs128_EmbedPatchAttention/train1/vocab/vocab_smiles.pt")
    formula_vocab = torch.load("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/FinalResult/0directly_train/withFormula/ps8_bs128_EmbedPatchAttention/train1/vocab/vocab_formula.pt")
    data = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/modified_ViT/30data.pkl"
    dataset = generateDataset(pd.read_pickle(data), smiles_vocab=smi_vocab, spec_len=3200, formula=True, formula_vocab=formula_vocab, 
                                           aug_mode=['verticalNoise', 'SMILES'], aug_num=1, max_shift=None, theta=0.01, alpha=5,
                                           smi_aug_num=2)

    createdataloader = CreateDataloader(pad_id=2,
                                        bs_id=0,
                                        eos_id=1,
                                        tgt_max_padding=40,
                                        formula=True, formula_max_pad=15)
    train_dataloader = createdataloader.dataloader(dataset, batch_size=6, shuffle=False)
    print(len(dataset))
    for batch in train_dataloader:
        # print(batch["spec"])
        spec = batch["spec"]
        
        for i in [0,2]:
            print()
            for spec_i in spec:
            # print(torch.equal(spec,spec[0]))
            # print(torch.eq(spec,spec[2]))
            # print(torch.eq(spec,spec[4]))
                print(torch.equal(spec_i, spec[i]))
        # print(spec == spec[2])
        # print(spec == spec[4])
        print(batch['formula'])
        print(batch['smi'])
        break

    
    # data = pd.read_pickle(data).iloc[0]
    # save_path = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/FinalResult/code/utils/aug_check/plot"
    # plotAug(dataset,total_num_per_mole=3,save_path=save_path,save_name="sim_shift_max100")
    # print(data.iloc[0])
    # plotOneData(data['smiles'], data['spectra'],save_path)


    """
    