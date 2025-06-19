import numpy as np
import pandas as pd
from tqdm import tqdm

import pubchempy
import chembl_structure_pipeline
from rdkit.Chem import AllChem, DataStructs, MACCSkeys
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit import Chem, RDLogger
from PyBioMed.PyMolecule import fingerprint

from mol2vec.features import mol2alt_sentence, MolSentence
from gensim.models import word2vec

RDLogger.DisableLog('rdApp.*')


def get_morgan_sentence(mol, radius, radiusFirst, returnsList=False):
    """
    Radius first: atom0 radius0 a0 r1 a1 r0 a1 r1 ...
    Atom first: a0 r0 a1 r0 ... aN r0 [EOR0] a0 r1 a1 r1 ... aN r1 [EOR1]

    Parameters
    ----------
    radius : int
        Radius used.
    radiusFirst : bool
        The divisor.
    returnsList : bool
        Whether to return a list or a string.

    Returns
    -------
    list
        Substructure identifier list in the canonical sequence.
    """
    bi = {}
    # bi: identifier -> tuples of (atom index, radius)
    # This substructure is located at this radius of this atom
    _ = AllChem.GetMorganFingerprint(mol, radius, bitInfo=bi)
    atom_idx = [a.GetIdx() for a in mol.GetAtoms()]
    if radius == 0 or radiusFirst:
        ai_to_r_to_idf = {ai: {} for ai in atom_idx} # atom idx -> radius -> identifier
        for idf, ai_r in bi.items():
            for ai, r in ai_r:
                ai_to_r_to_idf[ai][r] = idf
        sentence = [str(ai_to_r_to_idf[ai][r]) for ai in atom_idx for r in sorted(ai_to_r_to_idf[ai])]
    else:
        r_to_ai_to_idf = {r: {} for r in range(0, radius+1)} # radius -> atom idx -> identifier
        for idf, ai_r in bi.items():
            for ai, r in ai_r:
                r_to_ai_to_idf[r][ai] = idf
        sentence = []
        for r in range(0, radius+1):
            sentence.extend(str(r_to_ai_to_idf[r][ai]) for ai in atom_idx if ai in r_to_ai_to_idf[r])
            sentence.append('[EOR%d]' % r)
    if returnsList:
        return sentence
    return ' '.join(sentence)


def append_morgan_sentence(df, mol_col='mol'):
    """"""
    df['morgan_sentence_r_0_s_0'] = df[mol_col].apply(lambda x: get_morgan_sentence(x, 0, True))
    df['morgan_sentence_r_1_s_0_radiusFirst'] = df[mol_col].apply(lambda x: get_morgan_sentence(x, 1, True))
    df['morgan_sentence_r_1_s_0_atomFirst']   = df[mol_col].apply(lambda x: get_morgan_sentence(x, 1, False))
    df['morgan_sentence_r_2_s_0_radiusFirst'] = df[mol_col].apply(lambda x: get_morgan_sentence(x, 2, True))
    df['morgan_sentence_r_2_s_0_atomFirst']   = df[mol_col].apply(lambda x: get_morgan_sentence(x, 2, False))


# featurization
def featurize_maccs(df, mol_col='mol'):
    """
    166-bit MACCS feature matrix X

    Returns
    -------
    Pandas DataFrame
    """
    X = []
    for mol in df[mol_col]:
        a = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), a)
        X.append(a[1:])
    return pd.DataFrame(np.stack(X), index=df.index)


def featurize_pubchem_fp(df, mol_col='mol'):
    """PubChem 881-d fingerprint featurization"""
    X = []
    for mol in df[mol_col]:
        a = np.array(fingerprint.CalculatePubChemFingerprint(mol))
        X.append(a)
    return pd.DataFrame(np.stack(X), index=df.index)


def featurize_pubchem_fp_pcp(df, smiles_col='canonical_smiles', cid_col='cid'):
    """
    Parameters
    ----------
    cid_col : str
        Not None: column name for PubChem CID
    """
    cids = []
    X = []
    for smiles in tqdm(df[smiles_col], desc='PubChem fingerprint'):
        compound = pubchempy.get_compounds(smiles, 'smiles')[0]
        cids.append(compound.cid)
        a = np.array([int(bit) for bit in compound.cactvs_fingerprint])
        X.append(a)
    if cid_col is not None:
        df[cid_col] = cids
    return pd.DataFrame(np.stack(X), index=df.index)


def featurize_ecfp(df, mol_col='mol', radius=1, nbits=1024):
    """ECFP featurization"""
    X = []
    for mol in df[mol_col]:
        a = np.zeros((0,), dtype=np.int8)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        DataStructs.ConvertToNumpyArray(fp, a)
        X.append(a)
    return pd.DataFrame(np.stack(X), index=df.index)


def featurize_rdkit_fp(df, mol_col='mol'):
    """2048-bit RDKit fingerprint featurization"""
    X = []
    fpgen = AllChem.GetRDKitFPGenerator()
    for mol in df[mol_col]:
        a = np.zeros((0,), dtype=np.int8)
        fp = fpgen.GetFingerprint(mol)
        DataStructs.ConvertToNumpyArray(fp, a)
        X.append(a)
    return pd.DataFrame(np.stack(X), index=df.index)


def featurize_daylight_fp(df, mol_col='mol'):
    """2048-bit Daylight-like featurization"""
    d = 2048
    X = []
    for mol in df[mol_col]:
        a = np.zeros((0,), dtype=np.int8)
        fp = FingerprintMols.FingerprintMol(mol)
        DataStructs.ConvertToNumpyArray(fp, a)
        n = d - len(a)
        if n > 0:
            a = np.concatenate([a, np.zeros(n)])
        X.append(a)
    return pd.DataFrame(np.stack(X), index=df.index)


def _mol2vec_sentences2vec(sentence, model, unseen=None):
    """Adapted from Mol2vec.
    
    Generate vectors for each sentence (list) in a list of sentences. Vector is simply a
    sum of vectors for individual words.
    
    Parameters
    ----------
    sentences : list, array
        List with sentences
    model : word2vec.Word2Vec
        Gensim word2vec model
    unseen : None, str
        Keyword for unseen words. If None, those words are skipped.
        https://stats.stackexchange.com/questions/163005/how-to-set-the-dictionary-for-text-analysis-using-neural-networks/163032#163032

    Returns
    -------
    np.array
    """
    keys = set(model.wv.key_to_index.keys())
    if unseen:
        unseen_vec = model.wv.get_vector(unseen)
        return sum([model.wv.get_vector(y) if y in set(sentence) & keys else unseen_vec for y in sentence])
    return sum([model.wv.get_vector(y) for y in sentence if y in set(sentence) & keys])


def featurize_mol2vec(df, mol_col='mol'):
    """"""
    X = []
    model = word2vec.Word2Vec.load('./mol2vec/examples/models/model_300dim.pkl')
    for mol in df[mol_col]:
        sent = MolSentence(mol2alt_sentence(mol, 1))
        a = _mol2vec_sentences2vec(sent, model, unseen='UNK')
        X.append(a)
    return pd.DataFrame(np.stack(X), index=df.index)
