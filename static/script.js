const predictBtn = document.getElementById('predictBtn');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const errorMsg = document.getElementById('errorMsg');

// Example data
function loadExample1() {
  document.getElementById('smiles').value = 'CC(=O)Oc1ccccc1C(=O)O';
  document.getElementById('protein').value = 'MAGEKIVKFKELLEQKAETSNGVLLDLACQEPQQHYLKLDRRLENSEAYVAKKQSDMAGWSLYDCLDYDELPTQVDYQWRRAARDAAKTTLAQMTFAAI';
}

function loadExample2() {
  document.getElementById('smiles').value = 'CC1C(=O)NC(=O)NC1=O';
  document.getElementById('protein').value = 'PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMNLPGKWKYVGRGGPSVGEVLERLAKLAGNV';
}

function loadExample3() {
  document.getElementById('smiles').value = 'CC1=CNc2c1c(OC3CCNCC3)nc(n2)N';
  document.getElementById('protein').value = 'MSDNGPQNQRNAPRITFGGPSDSTGSNQNGERSGARSKQRRPQGLPNNTASWFTALTQHGKEDLKFPRGQGVPI';
}

predictBtn.addEventListener('click', async () => {
  const smiles = document.getElementById('smiles').value.trim();
  const protein = document.getElementById('protein').value.trim();

  errorMsg.classList.add('hidden');
  results.classList.add('hidden');

  if (!smiles || !protein) {
    showError('Both SMILES and protein sequence are required.');
    return;
  }

  loading.classList.remove('hidden');
  predictBtn.disabled = true;

  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ smiles, protein_sequence: protein })
    });

    const data = await response.json();

    if (data.error) {
      showError(data.error);
    } else {
      displayResults(data);
    }
  } catch (err) {
    showError('Failed to connect to the server. Please try again.');
  } finally {
    loading.classList.add('hidden');
    predictBtn.disabled = false;
  }
});

function displayResults(data) {
  results.classList.remove('hidden');

  const prob = data.probability;
  const binding = data.binding;
  const confidence = data.confidence;

  // Gauge animation
  const gaugeFill = document.getElementById('gauge-fill');
  const gaugeValue = document.getElementById('probabilityValue');
  const circumference = 251.2;
  const offset = circumference - (prob * circumference);
  gaugeFill.style.strokeDashoffset = offset;
  gaugeValue.textContent = (prob * 100).toFixed(1) + '%';

  // Status badge
  const badge = document.getElementById('bindingBadge');
  const status = document.getElementById('bindingStatus');
  status.textContent = binding;
  badge.className = 'status-badge ' + (binding === 'Yes' ? 'positive' : 'negative');

  // Confidence bar
  const confidenceFill = document.getElementById('confidenceFill');
  const confidenceText = document.getElementById('confidenceText');
  confidenceFill.style.width = (confidence * 100) + '%';
  confidenceText.textContent = (confidence * 100).toFixed(1) + '%';

  // Detailed analysis
  document.getElementById('predScore').textContent = prob.toFixed(4);
  const pActivity = -Math.log10(prob * 1e-6);
  document.getElementById('pActivity').textContent = pActivity.toFixed(2);
  const ic50 = Math.pow(10, -pActivity) * 1e9;
  document.getElementById('ic50').textContent = ic50.toFixed(2) + ' nM';
  document.getElementById('suitability').textContent = binding === 'Yes' ? 'High' : 'Low';

  // Interpretation
  let interpretation = '';
  if (prob > 0.7) {
    interpretation = '✅ <strong>Strong Binding Predicted:</strong> The model predicts a high likelihood of interaction between this drug and target protein. This suggests potential therapeutic efficacy. Consider further experimental validation.';
  } else if (prob > 0.5) {
    interpretation = '⚠️ <strong>Moderate Binding Predicted:</strong> The drug shows moderate binding potential. Additional analysis and optimization may improve interaction strength.';
  } else {
    interpretation = '❌ <strong>Weak/No Binding Predicted:</strong> The model suggests low probability of meaningful interaction. This drug-target pair may not be suitable for further development.';
  }
  document.getElementById('interpretation').innerHTML = interpretation;
}

function showError(message) {
  errorMsg.classList.remove('hidden');
  document.getElementById('errorText').textContent = message;
}
