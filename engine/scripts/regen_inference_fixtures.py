"""Regenerate inference-fixtures.json policy/value against a current model.

Keeps the existing 20 position tensors (diverse game phases, already vetted);
only recomputes the reference outputs. Mirrors compute_inference() in
engine/scripts/gen_inference_fixtures.py: policy = raw logits list, value
rounded to 8 places.
"""
import json
import sys

import torch

sys.path.insert(0, "engine")
from razzle.ai.network import create_network  # noqa: E402

MODEL_NAME = "pegasus_iter_250.pt"
MODEL_PATH = f"engine/output/models/{MODEL_NAME}"  # run from repo root
FIXTURE_PATH = "webapp/src/engine/__tests__/inference-fixtures.json"

d = json.load(open(FIXTURE_PATH))
print(f"loaded {d['num_positions']} positions, old model: {d['model']}, preset: {d['preset']}")

net = create_network(preset=d["preset"])
ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
if "conv_input.weight" in ckpt:
    sd = ckpt
net.load_state_dict(sd)
net.eval()
print(f"network loaded from {MODEL_PATH}")

max_delta = 0.0
for p in d["positions"]:
    x = torch.tensor(p["tensor"], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        policy_logits, value, _difficulty = net(x)
    new_policy = policy_logits.squeeze(0).tolist()
    old_policy = p["policy"]
    max_delta = max(max_delta, max(abs(a - b) for a, b in zip(new_policy, old_policy)))
    p["policy"] = new_policy
    p["value"] = round(value.item(), 8)

d["model"] = MODEL_NAME
json.dump(d, open(FIXTURE_PATH, "w"))
print(f"wrote {FIXTURE_PATH}")
print(f"sanity: max policy delta vs old fixtures = {max_delta:.4f} (expect >0, different weights)")
print(f"values now: {[p['value'] for p in d['positions'][:5]]} ...")
