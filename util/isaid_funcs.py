import torch

def load_weights_from_defromDETR(ckpt_path, model, remove_weight=[]):
    def key_check(k, wstr):
        if wstr in k:
            return False
        return True

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    pretrain_dict = checkpoint['model']

    for rwgt  in remove_weight:
        pretrain_dict = {k: v for k, v in pretrain_dict.items()
                        if key_check(k, rwgt)}

    model.load_state_dict(pretrain_dict, strict=False)
    print(f"Pretrain Loaded ...{ckpt_path}  After removing class_embed tokens")
    return model
