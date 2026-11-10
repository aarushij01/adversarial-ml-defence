"""
demo.py
-------
Flask web demo for the Adversarial ML Defence project.
"""

from flask import Flask, request, jsonify
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import base64
import io
from PIL import Image

from src.model import load_model, DEVICE
from src.attacks import fgsm_attack, pgd_attack, to_input_eps

app = Flask(__name__)
model = None

def get_model():
    global model
    if model is None:
        model = load_model()
        model.eval()
    return model

DIGIT_NAMES = ["Zero","One","Two","Three","Four","Five","Six","Seven","Eight","Nine"]

def tensor_to_b64(tensor):
    mean, std = 0.1307, 0.3081
    img = tensor.cpu().squeeze()
    img = img * std + mean
    img = (img.clamp(0, 1).numpy() * 255).astype(np.uint8)
    pil_img = Image.fromarray(img, mode='L').resize((140, 140), Image.NEAREST)
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()

HTML = """<!DOCTYPE html>
<html>
<head>
<title>Adversarial ML Demo</title>
<style>
body{font-family:system-ui,sans-serif;background:#f5f5f5;padding:24px;margin:0}
.wrap{max-width:920px;margin:0 auto}
h1{font-size:22px;margin-bottom:4px}
.sub{color:#666;font-size:13px;margin-bottom:20px}
.card{background:white;border:1px solid #ddd;border-radius:8px;padding:24px;margin-bottom:16px}
label{font-size:13px;font-weight:500;display:block;margin:12px 0 4px}
select,input[type=number]{padding:8px;border:1px solid #ddd;border-radius:6px;font-size:13px}
input[type=range]{width:100%}
button{margin-top:12px;padding:10px 24px;background:#111;color:white;border:none;border-radius:6px;cursor:pointer;font-size:13px}
button:hover{background:#333}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin-top:16px}
.img-card{text-align:center;padding:12px;background:#f9f9f9;border-radius:8px;border:1px solid #eee}
.img-card img{border:2px solid #ddd;border-radius:4px;display:block;margin:8px auto}
.correct{color:#1a7a56;font-weight:500;font-size:13px}
.wrong{color:#c0392b;font-weight:500;font-size:13px}
.stat{font-size:12px;color:#888;margin-top:4px}
.loader{display:none;text-align:center;padding:20px;color:#666;font-size:13px}
.badge{display:inline-block;padding:3px 8px;border-radius:12px;font-size:11px;font-weight:500;margin-bottom:8px}
.badge-safe{background:#E1F5EE;color:#0F6E56}
.badge-attack{background:#FAECE7;color:#712B13}
.badge-perturb{background:#EEEDFE;color:#3C3489}
.summary{margin-top:16px;font-size:13px;padding:10px 14px;border-radius:6px}
.fooled{background:#FEECEC;color:#c0392b}
.survived{background:#EDFAF4;color:#1a7a56}
</style>
</head>
<body>
<div class="wrap">
<h1>Adversarial ML Attack &amp; Defence Demo</h1>
<p class="sub">See how invisible perturbations fool a 99% accurate neural network.</p>

<div class="card">
  <h3 style="font-size:15px;margin-bottom:12px">Attack Configuration</h3>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;align-items:end">
    <div>
      <label>Attack Type</label>
      <select id="attack">
        <option value="fgsm">FGSM (Fast, 1-step)</option>
        <option value="pgd">PGD (Strong, 40-step)</option>
      </select>
    </div>
    <div>
      <label>Epsilon (perturbation strength): <span id="eps-val">0.3</span></label>
      <input type="range" id="epsilon" min="0.05" max="0.5" step="0.05" value="0.3">
    </div>
    <div>
      <label>Test Image Index (0-9999)</label>
      <input type="number" id="img-idx" value="0" min="0" max="9999" style="width:100px">
    </div>
  </div>
  <button id="run-btn">Run Attack</button>
  <div class="loader" id="loader">Running attack... (PGD takes ~5 seconds)</div>
</div>

<div class="card" id="results-card" style="display:none">
  <h3 style="font-size:15px;margin-bottom:4px">Results</h3>
  <div class="grid" id="results-grid"></div>
  <div class="summary" id="summary"></div>
</div>
</div>

<script>
document.getElementById('epsilon').addEventListener('input', function() {
  document.getElementById('eps-val').innerText = this.value;
});

document.getElementById('run-btn').addEventListener('click', function() {
  var attack = document.getElementById('attack').value;
  var epsilon = parseFloat(document.getElementById('epsilon').value);
  var idx = parseInt(document.getElementById('img-idx').value);

  document.getElementById('loader').style.display = 'block';
  document.getElementById('results-card').style.display = 'none';

  var payload = JSON.stringify({attack: attack, epsilon: epsilon, idx: idx});

  fetch('/api/attack', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: payload
  })
  .then(function(r) { return r.json(); })
  .then(function(d) {
    document.getElementById('loader').style.display = 'none';
    document.getElementById('results-card').style.display = 'block';

    var adv_class = d.adv_correct ? 'correct' : 'wrong';
    var grid = document.getElementById('results-grid');
    grid.innerHTML =
      '<div class="img-card">' +
        '<span class="badge badge-safe">Original</span>' +
        '<img src="data:image/png;base64,' + d.clean_img + '" width="140" height="140">' +
        '<div class="correct">Predicted: ' + d.clean_pred + ' (' + d.clean_conf + '%)</div>' +
        '<div class="stat">True label: ' + d.true_label + '</div>' +
      '</div>' +
      '<div class="img-card">' +
        '<span class="badge badge-attack">Adversarial (' + d.attack_name + ')</span>' +
        '<img src="data:image/png;base64,' + d.adv_img + '" width="140" height="140">' +
        '<div class="' + adv_class + '">Predicted: ' + d.adv_pred + ' (' + d.adv_conf + '%)</div>' +
        '<div class="stat">Perturbation L&infin;=' + d.linf_norm + '</div>' +
      '</div>' +
      '<div class="img-card">' +
        '<span class="badge badge-perturb">Perturbation &times;10</span>' +
        '<img src="data:image/png;base64,' + d.perturb_img + '" width="140" height="140">' +
        '<div class="stat">Invisible to humans</div>' +
        '<div class="stat">Max pixel change: ' + Math.round(d.linf_norm * 255 * 10) / 10 + '/255</div>' +
      '</div>';

    var summary = document.getElementById('summary');
    if (d.fooled) {
      summary.className = 'summary fooled';
      summary.innerText = 'Attack succeeded — model fooled! The perturbation is imperceptible but completely changes the prediction.';
    } else {
      summary.className = 'summary survived';
      summary.innerText = 'Attack failed — model still correct. Try increasing epsilon or switching to PGD.';
    }
  })
  .catch(function(e) {
    document.getElementById('loader').style.display = 'none';
    alert('Error: ' + e);
  });
});
</script>
</body>
</html>"""


@app.route("/")
def index():
    return HTML


@app.route("/api/attack", methods=["POST"])
def api_attack():
    data = request.json
    attack_type = data.get("attack", "fgsm")
    epsilon = float(data.get("epsilon", 0.3))
    idx = int(data.get("idx", 0))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_set = torchvision.datasets.MNIST(root="data", train=False, download=True, transform=transform)
    image, label = test_set[idx]
    image = image.unsqueeze(0).to(DEVICE)
    label_tensor = torch.tensor([label]).to(DEVICE)

    mdl = get_model()

    eps_in = to_input_eps(epsilon)  # slider is in [0,1] pixel units
    if attack_type == "fgsm":
        adv_image = fgsm_attack(mdl, image, label_tensor, epsilon=eps_in)
        attack_name = "FGSM e=" + str(epsilon)
    else:
        adv_image = pgd_attack(mdl, image, label_tensor, epsilon=eps_in,
                               alpha=2.5 * eps_in / 40, num_steps=40)
        attack_name = "PGD e=" + str(epsilon)

    with torch.no_grad():
        clean_out = torch.softmax(mdl(image), dim=1)
        adv_out = torch.softmax(mdl(adv_image), dim=1)

    clean_pred = clean_out.argmax().item()
    adv_pred = adv_out.argmax().item()
    clean_conf = round(clean_out.max().item() * 100, 1)
    adv_conf = round(adv_out.max().item() * 100, 1)

    perturbation = (adv_image - image)
    linf = perturbation.abs().max().item() * 0.3081  # back to [0,1] pixel units
    perturb_display = (perturbation * 10 + 0.5).clamp(0, 1)

    return jsonify({
        "clean_img": tensor_to_b64(image),
        "adv_img": tensor_to_b64(adv_image),
        "perturb_img": tensor_to_b64(perturb_display),
        "clean_pred": DIGIT_NAMES[clean_pred],
        "adv_pred": DIGIT_NAMES[adv_pred],
        "true_label": DIGIT_NAMES[label],
        "clean_conf": clean_conf,
        "adv_conf": adv_conf,
        "linf_norm": round(linf, 3),
        "fooled": adv_pred != label,
        "adv_correct": adv_pred == label,
        "attack_name": attack_name,
    })


if __name__ == "__main__":
    print("\nAdversarial ML Demo running at: http://127.0.0.1:5003")
    app.run(debug=False, port=5003)
