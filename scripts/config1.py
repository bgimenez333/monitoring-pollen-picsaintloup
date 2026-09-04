

from pathlib import Path
import pickle
from collections import defaultdict
import numpy as np



REPO_DIR = Path(__file__).resolve().parent

# Data repository
DATA_DIR = Path("SCRIPTS FOR GITHUB REPOSITORY")


# Subfolders
CLASSIFICATION_DATA = DATA_DIR / "1.Classification method REPOSITORY" # AVAILABLE in a first ZENODO repository POLLEN54M : DOI 10.5281/zenodo.22011920
MONITORING_DATA = DATA_DIR / "2.Application monitoring REPOSITORY" # AVAILABLE in a second ZENODO repository POLLENmonit-PSL : DOI 10.5281/zenodo.22142820



##### LISTES
classes54 = ['Acacia', 'Acer', 'Alnus', 'Amarantaceae', 'Artemisia', 'Betulaceae', 'Brassicaceae', 'Buxus', 'Carduus', 'Caryophyllaceae', 
           'Cichorioideae', 'Cupressaceae', 'Cyperaceae', 'Echium', 'Ericaceae', 'Fagus', 'FraxinusExcelsior', 'FraxinusOrnus', 'Gallium', 
           'IndetBlurry', 'IndetCovered', 'Juglans', 'Lamiaceae', 'Liliaceae', 'Lycopodium', 'Moraceae', 'Morphotype1', 
           'Morphotype2', 'Myrtaceae', 'NonPollen', 'Olea', 'Other', 'Phillyrea', 'Pinaceae', 'Pistacia', 'Plantago', 
           'Platanus', 'Poaceae', 'PopulusSp', 'QuercusDeciduous', 'QuercusIlex', 'Ranunculaceae', 'Rhamnus', 'Rosaceae', 'Rumex', 'Salix',
           'Sanguisorba', 'Tilia', 'Ulmus', 'Urtica', 'ViburnumSambucusTp', 'VitisF', 'VitisS', 'XanthiumAmbrosia']

classes18 = ['QuercusDeciduous', 'QuercusIlex', 'Buxus', 'Phillyrea',  'Fraxinus',  'Olea',  'Cupressaceae',  'Pistacia',  'Poaceae','Plantago', 
               'VitisF', 'VitisS', 'Pinaceae', 'Other', 'IndetBlurry', 'IndetCovered', "NonPollen", "Lycopodium"] 

fig_class_order = ['QuercusDeciduous', 'QuercusIlex', 'Buxus', 'Phillyrea',  'Fraxinus', 'Olea',  'Cupressaceae', 
               'Pistacia',  'Poaceae','Plantago', 'VitisF', 'VitisS','Pinaceae', 'Other'] 


site_order= [ 'W1' ,'W2' ,'W3', 'W4', 'D1' ,'D2', 'D3']



##### functions for section "classification"

def get_predictions_scores(cval_id, mp, classes54, refenv="env"):
    """
    Extracting as list the corresponding true labels, predicted scores and image filenames for a given model
    
    with :
    cval_id - the cross-validation id of the model (e.g. "cval1")
    mp - the path to the folder containing the predictions of the models
    classes54 - the list of the 54 classes (target + non-target taxa)
    env - if "env", only the images environmental images from the test sets are considered for the evaluation
    """
    p_pred = mp / f"{cval_id}_pred_scores.pkl" 
    p_true = mp / f"{cval_id}_true_classes.pkl" 
    p_filename = mp / f"{cval_id}_images.pkl"

    with open(p_true, 'rb') as f:
        true = pickle.load(f)
    true=[classes54[int(el)] for el in true]

    with open(p_pred, 'rb') as f:
        predscores = pickle.load(f)

    with open(p_filename, 'rb') as f:
        filename = pickle.load(f)

    if refenv == "yes":
        true = [el for e, el in enumerate(true) if filename[e].split("/")[-1][:3]=='env']
        predscores  = [el for e, el in enumerate(predscores) if filename[e].split("/")[-1][:3]=='env']
        filename  = [el for e, el in enumerate(filename) if filename[e].split("/")[-1][:3]=='env']
    return true, predscores, filename





def get_predictions_classes(cval_id, mp, classes54,  envref="env"):
    """
    Extracting as list the corresponding true labels, predicted classes and image filenames for a given model
    
    with :
    cval_id - the cross-validation id of the model (e.g. "cval1")
    mp - the path to the folder containing the predictions of the models
    classes54 - the list of the 54 classes (target + non-target taxa)
    env - if "env", only the images environmental images from the test sets are considered for the evaluation
    """
    p_pred = mp / f"{cval_id}_pred_classes.pkl"
    p_true = mp / f"{cval_id}_true_classes.pkl"
    p_filename = mp / f"{cval_id}_images.pkl"

    with open(p_true, 'rb') as f:
        true = pickle.load(f)
    true=[classes54[int(el)] for el in true]

    with open(p_pred, 'rb') as f:
        pred = pickle.load(f)
    pred=[classes54[int(el)] for el in pred]

    with open(p_filename, 'rb') as f:
        filename = pickle.load(f)

    if envref == "env":
        true = [el for e, el in enumerate(true) if filename[e].split("/")[-1][:3]=='env']
        pred  = [el for e, el in enumerate(pred) if filename[e].split("/")[-1][:3]=='env']
        filename  = [el for e, el in enumerate(filename) if filename[e].split("/")[-1][:3]=='env']
    elif envref!="ref":
        print("error")
    return true, pred, filename




def get_18score_vector(cval_id, true_i, pred_i_54, weights_dict, sum_f1_by_model, classes18, classes54, class_sum_other):
    """ 
    input is a score vector of 54 classes
    Output is a vector of 18 classes,
    where the scores for the two fraxinus classes are summed together
    and where the scores for the non-target classes are summed together
    """
    class18_withoutFrax = [x for x in classes18 if x not in ['FraxinusExcelsior', 'FraxinusOrnus']]
    index_frax_excelsior = classes54.index("FraxinusExcelsior")
    index_frax_ornus = classes54.index("FraxinusOrnus")

    class_true = "Fraxinus" if true_i in ['FraxinusExcelsior', 'FraxinusOrnus'] else "Other" if true_i in class_sum_other else true_i
    dict_k_18 = {classes54[s]:float(score) for s, score in enumerate(pred_i_54) if classes54[s] in class18_withoutFrax}
    dict_k_18["Fraxinus"] = float(pred_i_54[index_frax_excelsior]) +  float(pred_i_54[index_frax_ornus])
    dict_k_18["Other"]=sum([float(score) for s, score in enumerate(pred_i_54) if classes54[s] in class_sum_other]) # sum to 1, with 18 keys

    # weighting the 18 scores with : * w / sum(w)
    dict_kj_weighted=defaultdict(float)
    for cl_i in classes18:
        score_kj = dict_k_18[cl_i]
        weight_kj = weights_dict[cval_id][cl_i] / sum_f1_by_model[cval_id]

        score_weighted = score_kj * weight_kj
        dict_kj_weighted[cl_i] = score_weighted
    sum_18scores = sum(dict_kj_weighted.values())
    norm_weightedscores_k={key : float(value/sum_18scores) for key,value in dict_kj_weighted.items()} # Sum to 1  print(sum(norm_weightedscores_k.values()))
    vector_k_18 = np.array([norm_weightedscores_k.get(k, 0.0) for k in classes18])
    return vector_k_18, class_true


##### Figures mapping



color_it_mapping = {
    r'$\it{Buxus}$': "seagreen", 
    r'$\it{Quercus}$ $\it{deciduous}$': '#33a02c',
    r'$\it{Quercus}$ $\it{ilex}$': 'darkgreen',
    'Cupressaceae': "goldenrod", # "khaki",
    r'$\it{Fraxinus}$' : "mediumaquamarine",
    r'$\it{Olea}$': "olivedrab",
    r'$\it{Phillyrea}$': '#b2df8a',
    'Pinaceae': "#c8d9e9", 
    r'$\it{Pistacia}$': "yellowgreen",
    r'$\it{Plantago}$': "darkorange",
    r'$\it{Vitis}$ fertile morph': "#7B1FA2",
    r'$\it{Vitis}$ sterile morph': "#CE93D8",
    'Poaceae': "gold",
    'Non-target taxa': "gainsboro",
}


color_mapping = {
    'Buxus':"seagreen", 
    'QuercusDeciduous': '#33a02c',
    'QuercusIlex' :  'darkgreen', 
    'Cupressaceae': "goldenrod", # khaki
    'Fraxinus': "mediumaquamarine",
    'Olea': "olivedrab",
    'Phillyrea':'#b2df8a', 
    'Pinaceae': "#c8d9e9", 
    'Pistacia': "yellowgreen", 
    'Plantago': "darkorange",
    'VitisF':"#7B1FA2", 
    'VitisS':"#CE93D8", 
    'Poaceae': "gold", 
    'Other': "gainsboro", 
}


class_it_mapping = {
    'Buxus':  r'$\it{Buxus}$', 
    'Cupressaceae':  'Cupressaceae', 
    'Fraxinus':  r'$\it{Fraxinus}$', 
    'IndetBlurry': 'Indet. (blurry)', 
    'IndetCovered': 'Indet. (covered)', 
    'Lycopodium':  r'$\it{Lycopodium}$', 
    'NonPollen': 'non pollen', 
    'Olea':  r'$\it{Olea}$', 
    'Phillyrea':  r'$\it{Phillyrea}$', 
    'Pinaceae': 'Pinaceae', 
    'Pistacia':  r'$\it{Pistacia}$', 
    'Plantago':  r'$\it{Plantago}$', 
    'Poaceae': 'Poaceae', 
    'QuercusDeciduous':  r'$\it{Quercus}$ $\it{deciduous}$', 
    'QuercusIlex':  r'$\it{Quercus}$ $\it{ilex}$', 
    'VitisF':  r'$\it{Vitis}$ fertile', 
    'VitisS':  r'$\it{Vitis}$ sterile',
    'Other': 'Non-target taxa'
}