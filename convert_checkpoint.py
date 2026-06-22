"""Convert an original FREST checkpoint to the released (renamed) module layout.

The original training code used legacy attribute names; this release renames a
few modules for clarity. State-dict keys therefore need remapping:

    backbone.adapter{1..4}.*    ->  backbone.strainer{1..4}.*
    m_backbone.adapter{1..4}.*  ->  m_backbone.strainer{1..4}.*
    WeatherPassFilter.*         ->  condition_projector.*

The source checkpoint is only read; a new file is written. Usage:

    python convert_checkpoint.py SRC.ckpt DST.ckpt
"""
import sys
import torch


def remap_key(k):
    for i in range(1, 5):
        k = k.replace(f'backbone.adapter{i}.', f'backbone.strainer{i}.')
    k = k.replace('WeatherPassFilter.', 'condition_projector.')
    return k


def main(src, dst):
    ckpt = torch.load(src, map_location='cpu', weights_only=False)
    sd = ckpt.get('state_dict', ckpt)
    # drop the legacy stand-alone contrastive_loss module (training-only queue;
    # the release keeps only contrastive_loss_out / ConditionContrastiveSampler)
    sd = {k: v for k, v in sd.items() if not k.startswith('contrastive_loss.')}
    new_sd = {remap_key(k): v for k, v in sd.items()}
    changed = sum(1 for k in sd if remap_key(k) != k)
    if 'state_dict' in ckpt:
        ckpt['state_dict'] = new_sd
        # keep hyper-parameter names in sync with the renamed __init__ args
        hp = ckpt.get('hyper_parameters', {})
        if 'mutual_weight' in hp:
            hp['discriminator_weight'] = hp.pop('mutual_weight')
        if 'lr_wpf' in hp:
            hp['projector_lr'] = hp.pop('lr_wpf')
        out = ckpt
    else:
        out = new_sd
    torch.save(out, dst)
    print(f'remapped {changed} keys -> {dst}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
