"""
model.py — RoadVision Nepal Phase 2: IDD baseline segmentation model.

Locked spec (Day 2, Task 1):
- smp.Unet(encoder_name="resnet50", encoder_weights="imagenet", classes=2)
- 2-channel output (background=0, drivable=1) to match CrossEntropyLoss(ignore_index=255)
- 512x512 input, matching Day 1's loader/test_loader.py output
- Encoder freeze/unfreeze toggle for a warmup-then-finetune training schedule
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def build_model(encoder_name: str = "resnet50", encoder_weights: str = "imagenet", classes: int = 2) -> nn.Module:
    """
    Builds the ResNet50 encoder + U-Net decoder segmentation model.

    Output: (B, classes, H, W) raw logits -- NOT softmaxed here. Softmax/argmax is
    applied inside the loss function or at inference time, not in the model itself,
    so this stays compatible with nn.CrossEntropyLoss (which expects raw logits).
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=classes,
    )
    return model


def freeze_encoder(model: nn.Module) -> None:
    """Freezes all encoder (ResNet50 backbone) parameters. Decoder stays trainable."""
    for param in model.encoder.parameters():
        param.requires_grad = False


def unfreeze_encoder(model: nn.Module) -> None:
    """Unfreezes all encoder parameters, allowing full fine-tuning."""
    for param in model.encoder.parameters():
        param.requires_grad = True


def set_encoder_frozen(model: nn.Module, epoch: int, freeze_until_epoch: int) -> bool:
    """
    Call once per epoch, before that epoch's training starts.
    Freezes the encoder for epoch < freeze_until_epoch, unfreezes from
    freeze_until_epoch onward.

    Returns True if the encoder is frozen after this call, False otherwise --
    useful for logging the freeze state into metrics.csv per epoch.
    """
    if epoch < freeze_until_epoch:
        freeze_encoder(model)
        return True
    else:
        unfreeze_encoder(model)
        return False


def count_trainable_params(model: nn.Module) -> int:
    """Sanity-check helper: counts trainable params, useful for confirming freeze/unfreeze actually worked."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick smoke test -- no training, just shape/behavior verification on random data.
    model = build_model()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params:                             {total_params:,}")
    print(f"Trainable params (encoder unfrozen):      {count_trainable_params(model):,}")

    freeze_encoder(model)
    print(f"Trainable params (encoder frozen):        {count_trainable_params(model):,}")

    unfreeze_encoder(model)
    print(f"Trainable params (re-unfrozen, sanity):   {count_trainable_params(model):,}")

    dummy_input = torch.randn(2, 3, 512, 512)
    output = model(dummy_input)
    print(f"\nInput shape:  {tuple(dummy_input.shape)}")
    print(f"Output shape: {tuple(output.shape)}  (expected: (2, 2, 512, 512))")
    assert output.shape == (2, 2, 512, 512), "Output shape mismatch!"
    print("model.py smoke test passed.")
