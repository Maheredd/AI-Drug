import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from model import DrugTargetInteractionNN
import json

class DrugTargetPredictor:
    def __init__(self, model_path='models/best_model.pt', config_path='models/model_config.json'):
        with open(config_path) as f:
            config = json.load(f)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DrugTargetInteractionNN(**config).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Model loaded from {model_path}")

    def smiles_to_features(self, smiles, fp_size=2048):
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

    def encode_protein(self, sequence, max_len=1000):
        AA_ENCODE = {
            'A': [1.8, 0, 1], 'R': [-4.5, 1, 3], 'N': [-3.5, 0, 2], 'D': [-3.5, -1, 2],
            'C': [2.5, 0, 1], 'Q': [-3.5, 0, 2], 'E': [-3.5, -1, 2], 'G': [-0.4, 0, 1],
            'H': [-3.2, 0.5, 2], 'I': [4.5, 0, 2], 'L': [3.8, 0, 2], 'K': [-3.9, 1, 2],
            'M': [1.9, 0, 2], 'F': [2.8, 0, 3], 'P': [-1.6, 0, 1], 'S': [-0.8, 0, 1],
            'T': [-0.7, 0, 1], 'W': [-0.9, 0, 3], 'Y': [-1.3, 0, 3], 'V': [4.2, 0, 2],
            'X': [0, 0, 0]
        }
        sequence = sequence[:max_len]
        encoded = []
        for aa in sequence:
            encoded.extend(AA_ENCODE.get(aa.upper(), [0, 0, 0]))
        encoded += [0] * (max_len * 3 - len(encoded))
        return np.array(encoded[:max_len * 3], dtype=np.float32)

    def predict(self, smiles, protein_sequence):
        drug_feat = self.smiles_to_features(smiles)
        prot_feat = self.encode_protein(protein_sequence)
        if drug_feat is None:
            return {'error': 'Invalid SMILES'}
        drug_tensor = torch.FloatTensor(drug_feat).unsqueeze(0).to(self.device)
        prot_tensor = torch.FloatTensor(prot_feat).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(drug_tensor, prot_tensor)
            prob = output.item()
        return {
            'probability': float(prob),
            'binding': 'Yes' if prob > 0.5 else 'No',
            'confidence': float(max(prob, 1 - prob))
        }
