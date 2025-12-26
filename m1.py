import sqlite3
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, DataStructs
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Connect to the ChEMBL SQLite database
conn = sqlite3.connect('chembl_36.db')

# SQL query to fetch canonical SMILES and IC50 bioactivity data
sql_query = """
SELECT
    cs.canonical_smiles,
    a.standard_value,
    a.standard_type,
    t.chembl_id AS target_id
FROM activities a
JOIN molecule_dictionary md ON a.molregno = md.molregno
JOIN compound_structures cs ON md.molregno = cs.molregno
JOIN assays assay ON a.assay_id = assay.assay_id
JOIN target_dictionary t ON assay.tid = t.tid
WHERE
    a.standard_type = 'IC50'
    AND a.standard_units = 'nM'
    AND a.standard_relation = '='
    AND a.standard_value IS NOT NULL
    AND a.data_validity_comment IS NULL
    AND a.potential_duplicate = 0
    AND t.target_type = 'SINGLE PROTEIN'
LIMIT 20000
"""

# Load the data into a pandas DataFrame
df = pd.read_sql(sql_query, conn)

# Convert IC50 (nM) to pIC50 (-log10 molar)
df['standard_value'] = df['standard_value'].astype(float)
df['pIC50'] = -np.log10(df['standard_value'] * 1e-9)

# Generate Morgan fingerprints using GetMorganFingerprintAsBitVect
def smiles_to_fingerprint(smiles, radius=2, nBits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        arr = np.zeros((nBits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    else:
        return np.zeros((nBits,), dtype=np.int8)

df['fingerprint'] = df['canonical_smiles'].apply(smiles_to_fingerprint)

# Prepare feature matrix and target vector
X = np.array(df['fingerprint'].tolist())
y = df['pIC50'].values

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Random Forest regression model
model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

# Predict on test set and evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Test RMSE: {rmse:.3f}")
print(f"Test R2 score: {r2:.3f}")

# Save the trained model
joblib.dump(model, 'chembl_36_rf_model.pkl')

# Close database connection
conn.close()
