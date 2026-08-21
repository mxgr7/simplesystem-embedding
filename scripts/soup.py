"""Uniform model soup: average the state_dicts of >=2 SPLADE checkpoints (same
backbone/arch) into one. python3 scripts/soup.py OUT.ckpt CKPT1 CKPT2 ..."""
import sys, glob, torch
out, ckpts = sys.argv[1], [glob.glob(p)[0] for p in sys.argv[2:]]
base = torch.load(ckpts[0], map_location="cpu", weights_only=False)
sds = [torch.load(c, map_location="cpu", weights_only=False)["state_dict"] for c in ckpts]
avg = {}
for k in sds[0]:
    if sds[0][k].is_floating_point():
        avg[k] = (sum(sd[k].float() for sd in sds) / len(sds)).to(sds[0][k].dtype)
    else:
        avg[k] = sds[0][k]
base["state_dict"] = avg
torch.save(base, out)
print(f"souped {len(ckpts)} ckpts -> {out}")
