from .detector3d_template import Detector3DTemplate


class CenterPoint(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict, dann_loss = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            # Kept SEPARATE from `loss`, as second_net.py does: train_st_utils sums it with the
            # target half and PCGrad (SELF_TRAIN.USE_TORCHJD) projects it against the detection
            # gradients. None unless the head has a discriminator AND the batch was stamped with
            # `domain_label` by the self-training loop.
            if dann_loss is not None:
                ret_dict['dann_loss'] = dann_loss
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        # In-head DANN (pcdet/models/model_utils/dann_utils.py); (None, {}) for every config
        # without `dann_weight`. Excluded from `loss` and from the rpn_loss logging, exactly as
        # AnchorHeadTemplate.get_loss excludes it.
        dann_loss, tb_dict_domain = self.dense_head.get_domain_adversarial_loss()
        tb_dict.update(tb_dict_domain)

        loss = loss_rpn
        return loss, tb_dict, disp_dict, dann_loss

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict
