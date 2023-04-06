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

    ret = model.load_state_dict(pretrain_dict, strict=False)
    print(ret)
    print(f"Pretrain Loaded ...{ckpt_path}  After removing {remove_weight}")
    return model
