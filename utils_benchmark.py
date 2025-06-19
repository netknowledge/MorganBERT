import numpy as np
import os
import pandas as pd
import simplejson as json
from os.path import exists, join
from rdkit import Chem

import model_classic
import model_chemprop
import model_molclr
import model_roberta
import utils_ml
import utils_mol
import utils_split


TASK_NAME_TO_INFO = {
    'ESOL': ('solubility_esol.csv', 'regression', 'logS'), 
    'AZ-sol': ('solubility_az.csv', 'regression', 'logS'), 
    'EPA-sol': ('solubility_epa.csv', 'regression', 'logS'), 
    'Biogen-sol': ('solubility_biogen.csv', 'regression', 'logS'), 
    #
    'AZ-lipo': ('lipophilicity_az.csv', 'regression', 'logD7.4'), 
    #
    'CSU-Caco2': ('permeability_csu_caco2.csv', 'regression', 'logPapp'), 
    'USTL-Caco2': ('permeability_ustl_caco2.csv', 'regression', 'logPapp'), 
    'Biogen-MDCK': ('permeability_biogen_mdck.csv', 'regression', 'LOG MDR1-MDCK ER (B-A/A-B)'), 
    #
    'Biogen-rPPB': ('PPB_r_biogen.csv', 'regression', 'LOG PLASMA PROTEIN BINDING (RAT) (% unbound)'), 
    'AZ-hPPB': ('PPB_h_az.csv', 'regression', 'log_pct_unbound'), 
    'Biogen-hPPB': ('PPB_h_biogen.csv', 'regression', 'LOG PLASMA PROTEIN BINDING (HUMAN) (% unbound)'), 
    #
    'Biogen-RLM': ('stability_biogen_rlm.csv', 'regression', 'LOG RLM_CLint (mL/min/kg)'), 
    'Biogen-HLM': ('stability_biogen_hlm.csv', 'regression', 'LOG HLM_CLint (mL/min/kg)'), 
    #
    'CYP3A4-CHEMBL1613886':  ('CYP3A4_CHEMBL1613886.csv', 'regression', 'pchembl_value'), 
    'CYP3A4-CHEMBL1614108':  ('CYP3A4_CHEMBL1614108.csv', 'regression', 'pchembl_value'), 
    'CYP3A4-CHEMBL1741324':  ('CYP3A4_CHEMBL1741324.csv', 'regression', 'pchembl_value'), 
    'CYP2C9-CHEMBL1614027':  ('CYP2C9_CHEMBL1614027.csv', 'regression', 'pchembl_value'), 
    'CYP2C9-CHEMBL1741325':  ('CYP2C9_CHEMBL1741325.csv', 'regression', 'pchembl_value'), 
    'CYP2C19-CHEMBL1613777': ('CYP2C19_CHEMBL1613777.csv', 'regression', 'pchembl_value'), 
    'CYP2C19-CHEMBL1741323': ('CYP2C19_CHEMBL1741323.csv', 'regression', 'pchembl_value'), 
    'CYP2D6-CHEMBL1614110':  ('CYP2D6_CHEMBL1614110.csv', 'regression', 'pchembl_value'), 
    'CYP2D6-CHEMBL1741321':  ('CYP2D6_CHEMBL1741321.csv', 'regression', 'pchembl_value'), 
    'CYP1A2-CHEMBL1741322':  ('CYP1A2_CHEMBL1741322.csv', 'regression', 'pchembl_value'), 
    #
    'NCATS-LD50-rat-subcutaneous': ('toxicity_ncats_LD50_rat_subcutaneous.csv', 'regression', 'rat_subcutaneous_LD50_(?log(mol/kg))'), 
    'NCATS-LD50-rat-intravenous': ('toxicity_ncats_LD50_rat_intravenous.csv', 'regression', 'rat_intravenous_LD50_(?log(mol/kg))'), 
    'NCATS-LD50-rat-intraperitoneal':('toxicity_ncats_LD50_rat_intraperitoneal.csv', 'regression', 'rat_intraperitoneal_LD50_(?log(mol/kg))'), 
    'NCATS-LD50-rat-oral': ('toxicity_ncats_LD50_rat_oral.csv', 'regression', 'rat_oral_LD50_(?log(mol/kg))'), 
    'NCATS-LD50-mouse-subcutaneous': ('toxicity_ncats_LD50_mouse_subcutaneous.csv', 'regression', 'mouse_subcutaneous_LD50_(?log(mol/kg))'), 
    'NCATS-LD50-mouse-intravenous': ('toxicity_ncats_LD50_mouse_intravenous.csv', 'regression', 'mouse_intravenous_LD50_(?log(mol/kg))'), 
    'NCATS-LD50-mouse-oral': ('toxicity_ncats_LD50_mouse_oral.csv', 'regression', 'mouse_oral_LD50_(?log(mol/kg))'), 
    'NCATS-LD50-mouse-intraperitoneal': ('toxicity_ncats_LD50_mouse_intraperitoneal.csv', 'regression', 'mouse_intraperitoneal_LD50_(?log(mol/kg))_(?log(mol/kg))'), 
}


MODEL_LIST = (
    'KNN', 
    'RF+MACCS', 'RF+PubChemFP', 'RF+ECFP2', 'RF+ECFP4', 'RF+Daylight', 'RF+RDKitFP', 'RF+Mol2vec', 
    'D-MPNN', 'MolCLR', 
    'ChemBERTa-10M-MLM', 'ChemBERTa-77M-MLM', 
    'MolFormer', 
    'MorganBERT_base_full_r_0_s_0', 
    'MorganBERT_base_full_r_1_s_0_atomFirst_f_300', 
    'MorganBERT_base_full_r_1_s_0_radiusFirst_f_300', 
    'MorganBERT_base_full_r_2_s_0_atomFirst_f_2300', 
    'MorganBERT_base_full_r_2_s_0_radiusFirst_f_2300', 
)


def print_perf_table(model_perf, metrics=None):
    """"""
    if metrics is None:
        metrics = model_perf['RF+MACCS'].columns
    for metric in metrics:
        print('-'*20, metric, '-'*20)
        for model in MODEL_LIST:
            if model in model_perf:
                perf_df = model_perf[model]
                if metric in perf_df.columns:
                    print(model, '{:.3f}'.format(perf_df[metric].mean()), r'$\pm$', '{:.3f}'.format(perf_df[metric].std()))
                else:
                    print(model)
            else:
                print(model)
        print('\n')


def load_model_perf(task_name, data_dir='./benchmarks'):
    """"""
    file_name, task, label_col = TASK_NAME_TO_INFO[task_name]
    in_path = join(data_dir, '%s_model_perf.json' % file_name.split('.')[0])
    model_perf = {}
    if exists(in_path):
        with open(in_path) as fin:
            result = json.load(fin)
        for k, v in result.items():
            model_perf[k] = pd.DataFrame(v)
    return model_perf


def save_model_perf(model_perf, out_path):
    """"""
    with open(out_path, 'w') as fout:
        json.dump({k: v.to_dict('records') for k, v in model_perf.items()}, fout, indent=True)


def get_summary_performance_df(task_names, reg_metric='eval_pcc', cls_metric='eval_auc', data_dir='/mnt/data2/morgan-bert/benchmarks'):
    """"""
    results = []
    for model in MODEL_LIST:
        row = []
        for task_name in task_names:
            try:
                file_name, task, _ = TASK_NAME_TO_INFO[task_name]
                metric = reg_metric if task == 'regression' else cls_metric
                with open(join(data_dir, '%s_model_perf.json' % file_name.split('.')[0])) as fin:
                    perf_df = pd.DataFrame(json.load(fin)[model])
                perf = '{:.3f}'.format(perf_df[metric].mean()) + r'\std{%s}' % '{:.3f}'.format(perf_df[metric].std())
                row.append(perf)
            except:
                row.append(None)
        results.append(row)
    df = pd.DataFrame(results, index=MODEL_LIST, columns=task_names)
    for col in df.columns:
        avg = df[col].dropna().apply(lambda x: float(x.split(r'\std{')[0]))
        if len(avg) > 0:
            if TASK_NAME_TO_INFO[col][1] == 'regression' and reg_metric not in ['eval_r2', 'eval_pcc']:
                best_row = avg.idxmin()
            else:
                best_row = avg.idxmax()
            df.loc[best_row,col] = '\\textbf{' + df.loc[best_row,col] + '}'
    return df


def test_model_performance(data_df, task_name, task, label_col, smiles_col='canonical_smiles', mol_col='mol', scaffold_split_trial=5, scaffold_split_outpath=None, model_perf=None):
    """
    Parameters
    ----------
    data_df : Pandas DataFrame
        The dataset
    task_name : str
        The dataset name
    label_col : str
        Column name of the y variable
    scaffold_split_outpath : str
    """
    if model_perf is None:
        model_perf = {}
    if mol_col not in data_df.columns:
        data_df[mol_col] = data_df[smiles_col].apply(Chem.MolFromSmiles)
    utils_mol.append_morgan_sentence(data_df)
    #
    if (scaffold_split_outpath is None) or (not os.path.exists(scaffold_split_outpath)):
        tt_split_seeds = list(range(scaffold_split_trial))
        train_test_index = [utils_split.scaffold_split(mol_series=data_df[mol_col], seed=i) for i in tt_split_seeds]
        # Further split train set for models with hyperparameter tuning
        tvt_split_seeds = list(range(scaffold_split_trial, scaffold_split_trial*2))
        train_valid_test_index = []
        for i, (train_index, test_index) in zip(tvt_split_seeds, train_test_index):
            train_valid_index = utils_split.scaffold_split(mol_series=data_df[mol_col].loc[train_index], sizes=(.85,.15), seed=i)
            train_valid_test_index.append(train_valid_index + (test_index,))
        if scaffold_split_outpath is not None:
            split_index = {
                'train_test_index': dict(zip(tt_split_seeds, train_test_index)), 
                'train_valid_test_index': dict(zip(tvt_split_seeds, train_valid_test_index))
            }
            with open(scaffold_split_outpath, 'w') as fout:
                json.dump(split_index, fout)
    else:
        with open(scaffold_split_outpath) as fin:
            split_index = json.load(fin)
        train_test_index = list(split_index['train_test_index'].values())
        train_valid_test_index = list(split_index['train_valid_test_index'].values())
    #
    model = 'KNN'
    if model not in model_perf:
        model_perf[model] = model_classic.run_knn(data_df[mol_col], data_df[label_col], train_test_index, task)
    #
    classics = [
        ('RF+MACCS', utils_mol.featurize_maccs, {}), 
        ('RF+PubChemFP', utils_mol.featurize_pubchem_fp, {}), 
        ('RF+ECFP2', utils_mol.featurize_ecfp, {}), 
        ('RF+ECFP4', utils_mol.featurize_ecfp, {'radius': 2,'nbits': 2048}), 
        ('RF+Daylight', utils_mol.featurize_daylight_fp, {}), 
        ('RF+RDKitFP', utils_mol.featurize_rdkit_fp, {}), 
    ]
    for model, featurize_func, para in classics:
        if model not in model_perf:
            X = featurize_func(data_df, **para)
            model_perf[model] = model_classic.run_randomforest(X, data_df[label_col], train_test_index, task, model)
    #
    model = 'D-MPNN'
    if model not in model_perf:
        model_perf[model] = model_chemprop.run_chemprop(data_df[smiles_col], data_df[label_col], train_test_index, task)
    model = 'MolCLR'
    if model not in model_perf:
        model_perf[model] = model_molclr.run_molclr(data_df[smiles_col], data_df[label_col], train_valid_test_index, task_name, task)
    #
    robertas = [
        ('chemberta',  'ChemBERTa-10M-MLM', smiles_col), 
        ('chemberta',  'ChemBERTa-77M-MLM', smiles_col), 
        ('morganbert', 'MorganBERT_base_full_r_0_s_0',                    'morgan_sentence_r_0_s_0'), 
        ('morganbert', 'MorganBERT_base_full_r_1_s_0_atomFirst_f_300',    'morgan_sentence_r_1_s_0_atomFirst'), 
        ('morganbert', 'MorganBERT_base_full_r_1_s_0_radiusFirst_f_300',  'morgan_sentence_r_1_s_0_radiusFirst'), 
        ('morganbert', 'MorganBERT_base_full_r_2_s_0_atomFirst_f_2300',   'morgan_sentence_r_2_s_0_atomFirst'), 
        ('morganbert', 'MorganBERT_base_full_r_2_s_0_radiusFirst_f_2300', 'morgan_sentence_r_2_s_0_radiusFirst'), 
    ]
    for group, model, text_col in robertas:
        if model not in model_perf:
            model_perf[model] = model_roberta.run_roberta_like(group, model, data_df[text_col], data_df[label_col], train_test_index[:1], task_name, task)
    return model_perf


def run_benchmark(task_name, model_perf, data_dir='./benchmarks'):
    """"""
    file_name, task, label_col = TASK_NAME_TO_INFO[task_name]
    data_path = join(data_dir, file_name)
    data_df = pd.read_csv(data_path)
    print('Number of compounds:', data_df.shape[0])
    model_perf = test_model_performance(data_df, task_name, task, label_col, 
                                        scaffold_split_outpath=os.path.splitext(data_path)[0] + '_splitindex.json', 
                                        model_perf=model_perf)
    save_model_perf(model_perf, os.path.splitext(data_path)[0] + '_model_perf.json')
    return model_perf
