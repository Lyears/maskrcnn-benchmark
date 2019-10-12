import torch


def delete_net_weights_for_finetune(
        model_file,
        out_file,
        rpn_final_convs=False,
        bbox_final_fcs=True,
        mask_final_conv=True
):
    del_keys = []
    checkpoint = torch.load(model_file)
    print("keys: {}".format(checkpoint.keys()))
    m = checkpoint['model']

    if rpn_final_convs:
        # 'module.rpn.anchor_generator.cell_anchors.0',
        # 'module.rpn.anchor_generator.cell_anchors.1',
        # 'module.rpn.anchor_generator.cell_anchors.2',
        # 'module.rpn.anchor_generator.cell_anchors.3',
        # 'module.rpn.anchor_generator.cell_anchors.4'
        # 'module.rpn.head.cls_logits.weight',
        # 'module.rpn.head.cls_logits.bias',
        # 'module.rpn.head.bbox_pred.weight',
        # 'module.rpn.head.bbox_pred.bias',
        del_keys.extend([
            k for k in m.keys() if k.find("rpn.anchor_generator") is not -1
        ])
        del_keys.extend([
            k for k in m.keys() if k.find("rpn.head.cls_logits") is not -1
        ])
        del_keys.extend([
            k for k in m.keys() if k.find("rpn.head.bbox_pred") is not -1
        ])

    if bbox_final_fcs:
        # 'module.roi_heads.box.predictor.cls_score.weight',
        # 'module.roi_heads.box.predictor.cls_score.bias',
        # 'module.roi_heads.box.predictor.bbox_pred.weight',
        # 'module.roi_heads.box.predictor.bbox_pred.bias',
        del_keys.extend([
            k for k in m.keys() if k.find(
                "roi_heads.box.predictor.cls_score"
            ) is not -1
        ])
        del_keys.extend([
            k for k in m.keys() if k.find(
                "roi_heads.box.predictor.bbox_pred"
            ) is not -1
        ])

    if mask_final_conv:
        # 'module.roi_heads.mask.predictor.mask_fcn_logits.weight',
        # 'module.roi_heads.mask.predictor.mask_fcn_logits.bias',
        del_keys.extend([
            k for k in m.keys() if k.find(
                "roi_heads.mask.predictor.mask_fcn_logits"
            ) is not -1
        ])

    for k in del_keys:
        print("del k: {}".format(k))
        del m[k]

    # checkpoint['model'] = m
    print("f: {}\nout_file: {}".format(f, out_file))
    recursively_mkdirs(os.path.dirname(out_file))
    torch.save({"model": m}, out_file)