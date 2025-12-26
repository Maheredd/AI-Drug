import sqlite3
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class MolecularFeaturizer:
    @staticmethod
    def smiles_to_features(smiles, fp_size=2048):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_size)
        fp_array = np.array(fp)
        descriptors = np.array([
            Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol), Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol), Descriptors.FractionCSP3(mol)
        ])
        descriptors /= np.array([500, 5, 10, 10, 150, 10, 5, 1])
        return np.concatenate([fp_array, descriptors]).astype(np.float32)

class ProteinSequenceEncoder:
    AA_ENCODE = {
        'A': [1.8, 0, 1], 'R': [-4.5, 1, 3], 'N': [-3.5, 0, 2], 'D': [-3.5, -1, 2],
        'C': [2.5, 0, 1], 'Q': [-3.5, 0, 2], 'E': [-3.5, -1, 2], 'G': [-0.4, 0, 1],
        'H': [-3.2, 0.5, 2], 'I': [4.5, 0, 2], 'L': [3.8, 0, 2], 'K': [-3.9, 1, 2],
        'M': [1.9, 0, 2], 'F': [2.8, 0, 3], 'P': [-1.6, 0, 1], 'S': [-0.8, 0, 1],
        'T': [-0.7, 0, 1], 'W': [-0.9, 0, 3], 'Y': [-1.3, 0, 3], 'V': [4.2, 0, 2],
        'X': [0, 0, 0]
    }

    @classmethod
    def encode(cls, sequence, max_len=1000):
        sequence = sequence[:max_len]
        encoded = []
        for aa in sequence:
            encoded.extend(cls.AA_ENCODE.get(aa.upper(), [0, 0, 0]))
        encoded += [0] * (max_len * 3 - len(encoded))
        return np.array(encoded, dtype=np.float32)

class DrugTargetDataset(Dataset):
    def __init__(self, drug_feats, protein_feats, labels):
        self.drug_feats = torch.FloatTensor(drug_feats)
        self.protein_feats = torch.FloatTensor(protein_feats)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.drug_feats[idx], self.protein_feats[idx], self.labels[idx]

def batch_query(db_path, batch_size=2000, limit=50000):
    conn = sqlite3.connect(db_path)
    offset = 0
    dfs = []
    query_base = """
    SELECT cs.canonical_smiles, a.standard_value, a.standard_type, t.chembl_id AS target_id
    FROM activities a
    JOIN molecule_dictionary md ON a.molregno=md.molregno
    JOIN compound_structures cs ON md.molregno=cs.molregno
    JOIN assays assay ON a.assay_id=assay.assay_id
    JOIN target_dictionary t ON assay.tid=t.tid
    WHERE a.standard_type='IC50' AND a.standard_units='nM'
      AND a.standard_relation='=' AND a.standard_value IS NOT NULL
      AND a.data_validity_comment IS NULL AND a.potential_duplicate = 0
      AND t.target_type='SINGLE PROTEIN' 
    """

    while offset < limit:
        query = query_base + f" LIMIT {batch_size} OFFSET {offset}"
        df = pd.read_sql_query(query, conn)
        if df.empty:
            break
        dfs.append(df)
        offset += batch_size
    
    conn.close()
    return pd.concat(dfs, ignore_index=True)

def fetch_protein_sequence(target_id):
    # Implement your own method to get sequence (from local DB or API)
    return "MKVLWALLVTFLAGCQAKVE" # Placeholder sequence

def create_dataloaders(db_path, sample_limit=10000, batch_size=128):
    df = batch_query(db_path, limit=sample_limit)
    drug_features, protein_features, labels = [], [], []

    for _, row in df.iterrows():
        drug_feat = MolecularFeaturizer.smiles_to_features(row['canonical_smiles'])
        prot_seq = fetch_protein_sequence(row['target_id'])
        prot_feat = ProteinSequenceEncoder.encode(prot_seq) if prot_seq else None

        if drug_feat is None or prot_feat is None:
            continue
        
        drug_features.append(drug_feat)
        protein_features.append(prot_feat)

        pActivity = -np.log10(row['standard_value'] * 1e-9)
        label = 1 if pActivity > 6 else 0
        labels.append(label)

    drug_features = np.array(drug_features)
    protein_features = np.array(protein_features)
    labels = np.array(labels)

    from sklearn.model_selection import train_test_split
    X_drug_train, X_drug_temp, X_prot_train, X_prot_temp, y_train, y_temp = train_test_split(
        drug_features, protein_features, labels, test_size=0.3, random_state=42, stratify=labels)

    X_drug_val, X_drug_test, X_prot_val, X_prot_test, y_val, y_test = train_test_split(
        X_drug_temp, X_prot_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    train_dataset = DrugTargetDataset(X_drug_train, X_prot_train, y_train)
    val_dataset = DrugTargetDataset(X_drug_val, X_prot_val, y_val)
    test_dataset = DrugTargetDataset(X_drug_test, X_prot_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    metadata = {
        'drug_feat_dim': drug_features.shape[1],
        'protein_feat_dim': protein_features.shape[1],
        'train_size': len(y_train),
        'val_size': len(y_val),
        'test_size': len(y_test),
    }
    return train_loader, val_loader, test_loader, metadata
