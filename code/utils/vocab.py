import os
from torchtext.vocab import build_vocab_from_iterator
import re
import torch

def build_vocab(data, vocab_save_path, vocab_filename_prefix, mode):
        """
        parameters:
            data: pd.dataframe or list only contains smiles or formula
            vocab_save_path
            vocab_filename_prefix
            mode: 'smiles' or 'formula'
                
        output: vocab
            vocab['#']: get index of the token
            len(vocab): get vocabulary size
            vocab.get_stoi(): str to int, {str: int}
            vocab.get_itos(): int to str, [str,...]
        """
        assert mode in ['formula', 'smiles'], "Please check mode of vocab building which should be 'formula' or 'smiles'"
        
        def token_iterator(data, mode):
            if mode == 'formula':
                for i in data: yield split_formula(i)
            elif mode == 'smiles':
                for i in data: yield split_smiles(i)
                    
        vocab = build_vocab_from_iterator(token_iterator(data, mode),
                                          specials=["<s>", "</s>", "<blank>", "<unk>"])
        vocab.set_default_index(vocab["<unk>"])
        
        with open(os.path.join(vocab_save_path, vocab_filename_prefix+'.txt'),'w') as f:
            for i,word in enumerate(vocab.get_itos()):
                f.write("{} {}\n".format(i,word))
        torch.save(vocab, os.path.join(vocab_save_path, vocab_filename_prefix+'.pt'))
        return vocab



def split_smiles(smile: str) -> str:
    pattern_full = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"

    regex = re.compile(pattern_full)
    tokens = [token for token in regex.findall(smile)]

    if smile != "".join(tokens):
        print(smile)
        raise ValueError(
            "Tokenised smiles does not match original: {} {}".format(tokens, smile)
        )

    return tokens

def split_formula(formula: str) -> str:
    segments = re.findall(r"[A-Z][a-z]*\d*", formula)
    formula_split = [a for segment in segments for a in re.split(r"(\d+)", segment)]
    formula_split = list(filter(None, formula_split))

    if "".join(formula_split) != formula:
        raise ValueError(
            "Tokenised smiles does not match original: {} {}".format(
                formula_split, formula
            )
        )
    return formula_split

    



    


         
