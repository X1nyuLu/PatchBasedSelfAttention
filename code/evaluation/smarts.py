# Define SMARTS

alkane = "[CX4]"
alkene = "[CX3]=[CX3]"
alkyne = "[CX2]#C"
arene = "[$([cX3](:*):*),$([cX2+](:*):*)]"
alcohol = "[#6][OX2H]"
ether = "[OX2;!$(OC=O)]([#6])[#6]"
aldehyde = "[CX3H1](=O)[#6]"
ketone = "[#6][CX3](=O)[#6]"
carboxylic_acid = "[CX3](=O)[OX2H1]"
ester = "[#6][CX3](=O)[OX2H0][#6]"
haloalkane = "[#6][F,Cl,Br,I]"
acyl_halide = "[CX3](=[OX1])[F,Cl,Br,I]"
amine = "[NX3;!$(NC=O)]"
amide = "[NX3][CX3](=[OX1])[#6]"
nitrile = "[NX1]#[CX2]"
sulfide = '[#16X2H0]'
thiol = '[#16X2H1]'
ortho_ = "*-!:aa-!:*"
meta_ = "*-!:aaa-!:*"
para_ = "*-!:aaaa-!:*"


fg_list_20 = [
        alkane,
        alkene,
        alkyne,
        arene, 
        alcohol,
        ether, 
        aldehyde,
        ketone, 
        carboxylic_acid, 
        ester, 
        haloalkane, 
        acyl_halide, 
        amine, 
        amide, 
        nitrile,
        sulfide,
        thiol,
        ortho_,
        meta_,
        para_,
        ]



label_names_20 = [
        'Alkane',
        'Alkene',
        'Alkyne',
        'Arene', 
        'Alcohol',
        'Ether', 
        'Aldehyde',
        'Ketone', 
        'Carboxylic acid', 
        'Ester', 
        'Haloalkane', 
        'Acyl halide',
        'Amine', 
        'Amide', 
        'Nitrile',
        'Sulfide',
        'Thiol',
        "ortho-",
        "meta-",
        "para-"]