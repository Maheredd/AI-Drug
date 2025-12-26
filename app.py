from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from predict import DrugTargetPredictor

app = Flask(__name__)
CORS(app)

predictor = DrugTargetPredictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    smiles = data.get('smiles')
    protein_seq = data.get('protein_sequence')
    if not smiles or not protein_seq:
        return jsonify({'error': 'SMILES and protein sequence required'}), 400
    result = predictor.predict(smiles, protein_seq)
    return jsonify(result)

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'OK', 'device': str(predictor.device)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
