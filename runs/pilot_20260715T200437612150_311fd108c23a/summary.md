# Run pilot_20260715T200437612150_311fd108c23a

- run dir: /content/drive/MyDrive/jacobian-lens-gemma/runs/pilot_20260715T200437612150_311fd108c23a
- mode: pilot
- config: configs/gemma_text_pilot.yaml (fingerprint sha256:311fd108c23a1bd8869c8bbb3e19013660a41fcbdb249710707d9edd65f4dc22)
- model: google/gemma-4-E4B-it @ fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd
- lens: /content/drive/MyDrive/jacobian-lens-gemma/runs/pilot_20260715T200437612150_311fd108c23a/artifacts/lens.pt (100 prompts fitted)
- checkpoint: /content/drive/MyDrive/jacobian-lens-gemma/runs/pilot_20260715T200437612150_311fd108c23a/checkpoints/ckpt.pt

## Controls (top-k overlap with model output / rank of model top-1)

- layer 3: J-lens=0.00/r23422, logit-lens=0.00/r25671, permuted=0.00/r93091, random=0.00/r198331, wrong_layer=0.00/r1762
- layer 7: J-lens=0.00/r824, logit-lens=0.00/r54024, permuted=0.00/r53437, random=0.00/r4080, wrong_layer=0.00/r7947
- layer 14: J-lens=0.05/r35572, logit-lens=0.00/r70932, permuted=0.00/r204397, random=0.00/r25885, wrong_layer=0.05/r32569
- layer 21: J-lens=0.05/r18842, logit-lens=0.00/r112694, permuted=0.00/r209056, random=0.00/r240277, wrong_layer=0.15/r1131
- layer 28: J-lens=0.10/r5205, logit-lens=0.00/r85662, permuted=0.00/r170479, random=0.00/r131744, wrong_layer=0.10/r78113
- layer 35: J-lens=0.20/r510, logit-lens=0.20/r1344, permuted=0.00/r132405, random=0.00/r15936, wrong_layer=0.20/r175
- layer 38: J-lens=0.20/r12, logit-lens=0.20/r54, permuted=0.00/r64015, random=0.00/r22715, wrong_layer=0.00/r260183