MaskRcnn测试代码的修改说明

这个文档主要目的是修改MaskRcnn的测试代码能汇报出对每一个类别的详细测试结果。

修改部分位于line55-74, 当line61 eval_overall设为True时，只显示总体结果。否则，根据line68 cat_dict的类别字典（需匹配模型输出类别）显示每个类别的测试结果。

新增了一个函数def evaluate_predictions_on_coco_by_category(
    coco_gt, coco_results, json_result_file, cat_dict, iou_type="bbox"
)位于line338-364

