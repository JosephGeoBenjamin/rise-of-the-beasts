import torch

def load_weights_from_defromDETR(ckpt_path, model):
    def key_check(k):
        if "class_embed" in k:
            return False
        return True

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    pretrain_dict = checkpoint['model']

    #TODO:JGB fix below
    # pretrain_dict = {k: v for k, v in pretrain_dict.items() if key_check(k)}
    ## OR
    # del checkpoint["model"]["class_embed.weight"]
    # del checkpoint["model"]["class_embed.bias"]

    model.load_state_dict(pretrain_dict, strict=False)
    print(f"Pretrain Loaded ...{ckpt_path}  After removing class_embed tokens")
    return model
