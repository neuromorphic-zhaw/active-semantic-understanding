# Paper Checkpoints

These are the exact best model weights used for the ADE20K-Object paper
experiments. Cluster paths and other machine-specific fields were removed from
the checkpoint metadata; model tensors are unchanged.

| File | Role | SHA-256 |
|---|---|---|
| `baseline.pt` | Full-image 512x512 Lite R-ASPP baseline | `d5ad468e21fdade8cc639b5d2e9a08520ed086f4845346bb6e8464fb4b78944c` |
| `fovea_only.pt` | 96x96 foveal model without context | `6fb7604226cc7b64b42908749e27395c87d6e13365c194e0f3e40297bad69db6` |
| `fovea_context.pt` | 96x96 foveal model with 128x128 global context | `43abe50aa9168d5f93460c20b51aeb6b4b37fbdf495251cd6a1081cd0f138c70` |
| `full_object.pt` | Full-object crop training control, evaluated on foveal crops | `3924c0b25f02a46da1280a94121fa1b0bb3850dac9c79e73d894b5ef7a193e51` |

`metadata.json` records file sizes, training settings, checkpoint epochs, and
the validation metrics stored with each original run.
