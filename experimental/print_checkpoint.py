import torch, pprint

ckpt = torch.load("../checkpoints/dpfm_n2_500demos/flow_matching_policy.pt", map_location="cpu", weights_only=False)
pprint.pprint({k: v for k, v in ckpt.items() if k != "model_state_dict"})

for k, v in ckpt.items():
  if "norm" in k.lower():
      print(f"{k}: {v}")