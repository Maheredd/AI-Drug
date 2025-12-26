import torch
import torch.nn as nn

class DrugTargetInteractionNN(nn.Module):
    def __init__(self, drug_input_dim=2056, protein_input_dim=3000,
                 hidden_dims=[1024, 512, 256], dropout=0.3):
        super().__init__()

        self.drug_encoder = nn.Sequential(
            nn.Linear(drug_input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU()
        )

        self.protein_encoder = nn.Sequential(
            nn.Linear(protein_input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU()
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dims[2] * 2, hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dims[1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, drug_features, protein_features):
        drug_encoded = self.drug_encoder(drug_features)
        protein_encoded = self.protein_encoder(protein_features)
        combined = torch.cat([drug_encoded, protein_encoded], dim=1)
        output = self.fusion(combined)
        return output.squeeze()
