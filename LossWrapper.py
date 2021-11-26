import torch
import utils as utils
from rewards import init_scorer, get_self_critical_reward

class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        if opt.label_smoothing > 0:
            self.crit = utils.LabelSmoothing(smoothing=opt.label_smoothing)
        else:
            self.crit = utils.LanguageModelCriterion()
        self.rl_crit = utils.RewardCriterion()

    def forward(self, fc_feats, att_feats, labels, box_logits, att_relscore, masks, att_masks, gts, data, sc_flag):
        out = {}
        if not sc_flag:
            loss = self.crit(self.model(fc_feats, att_feats, labels, box_logits, att_relscore, att_masks), labels[:,1:], masks[:,1:])
            out['reward'] = 0
        else:
            gen_result, sample_logprobs = self.model(fc_feats, att_feats, box_logits, att_relscore, att_masks, opt={'sample_max':0}, mode='sample')
            reward = get_self_critical_reward(self.model, fc_feats, att_feats, box_logits, att_relscore, att_masks, data, gen_result, self.opt)
            loss = self.rl_crit(sample_logprobs, gen_result.data, torch.from_numpy(reward).float().cuda())
            out['reward'] = reward[:,0].mean()
            
        out['loss'] = loss
        return out
