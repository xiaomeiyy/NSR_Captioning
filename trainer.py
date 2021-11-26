from six.moves import range
import os
import time
from six.moves import cPickle

import torch.backends.cudnn as cudnn
import torch
from torch.autograd import Variable
import torch.optim as optim

import utils
import eval_utils_transf as eval_utils_cap
from LossWrapper import LossWrapper

def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)

def Trainer(loader, cfg):
    
    if cfg.overfit_train:
        log_interval = 10
        evl_interval = 10
    else:
        log_interval = 100
        evl_interval = 3000
    
    ## ------------- prepare for captioning model ----------------
    cfg.vocab_size = loader.vocab_size
    cfg.vocab_glove = torch.from_numpy(loader.vocab_dict_glove).cuda()
    cfg.obj_glove = torch.from_numpy(loader.obj_glove).cuda()
    cfg.seq_length = loader.seq_length
    cfg.rel_dic = torch.from_numpy(loader.rel_dic).cuda() 

    if cfg.id == 'deeplstm':
        from AttModel import DeepLSTMModel as model_cap
    elif cfg.id == 'topdown':
        from AttModel_transformer3 import TopDownModel as model_cap
    elif cfg.id == 'NSM':
        from AttModel_transformer1 import NSMModel as model_cap 
    # Transformer
    elif cfg.id == 'transformer':
        from Transformer_nsm_fusion import TransformerModel as model_cap 
    print('cap_model is {} of {}'.format(cfg.id, cfg.id_model))
    model_c = model_cap(cfg).cuda()
    dp_model_c = torch.nn.DataParallel(model_c)
    lw_model = LossWrapper(dp_model_c, cfg)
    dp_lw_model = torch.nn.DataParallel(lw_model)
    
    update_lr_flag = True
    infos = {}
    lang_stats_total = []

    dp_lw_model.train()
    crit_i2t = utils.LanguageModelCriterion()
    if cfg.noamopt:
        assert opt.caption_model == 'transformer', 'noamopt can only work with transformer'
        optimizer = utils.get_std_opt(model, factor=opt.noamopt_factor, warmup=opt.noamopt_warmup)
        optimizer._step = iteration
    else:
        optimizer = utils.build_optimizer(dp_model_c.parameters(), cfg)

    ####################################################################################
    best_val_score_cap = 0
    epoch = 1
    iter = 0
    while True:
        ## update image to caption learning_rate
        if update_lr_flag:
            if epoch > cfg.learning_rate_decay_start and cfg.learning_rate_decay_start >=0:
                frac = (epoch - cfg.learning_rate_decay_start) // cfg.learning_rate_decay_every
                decay_factor = cfg.learning_rate_decay_rate ** frac
                current_lr = cfg.learning_rate * decay_factor
#                 current_lr = cfg.learning_rate * 0.5
               
            else:
                current_lr = cfg.learning_rate
#                 current_lr = min(cfg.learning_rate * epoch, cfg.learning_rate *3)
            utils.set_lr(optimizer, current_lr)

            if epoch > cfg.scheduled_sampling_start and cfg.scheduled_sampling_start >= 0:
                frac = (epoch - cfg.scheduled_sampling_start) // cfg.scheduled_sampling_increase_every
                ss_prob = min(cfg.scheduled_sampling_increase_prob * frac, cfg.scheduled_sampling_max_prob)
                model_c.ss_prob = ss_prob
                
             ## if start self critical training
            if cfg.self_critical_after != -1 and epoch >= cfg.self_critical_after:
#             if iter > 50:
                sc_flag = True
            else:
                sc_flag = False  
                
            update_lr_flag_i2t = False

        #####################################################################################################################
        torch.cuda.synchronize()
        start_t = time.time()
        data = loader.get_batch('train')
        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks'], data['box_logits'], data['att_relscore']]
        tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
        fc_feats, att_feats, labels, masks, att_masks, box_logits, att_relscore = tmp

        ######################################################
        # training of  image to captioning
        ######################################################
        optimizer.zero_grad()
        model_out = dp_lw_model(fc_feats, att_feats, labels, box_logits, att_relscore, masks, att_masks, data['gts'], data, sc_flag)

        loss = model_out['loss'].mean()
        loss.backward()

        optimizer.step()
        train_loss_i2t = loss.item()

        torch.cuda.synchronize()
        end_t = time.time()

        iter = iter + 1
        # Update the iteration and epoch
        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag_i2t = True
        if iter % log_interval == 0 and iter != 0:
            print('[{}][{}/{}]: loss_cap={:.3f}, rewards={:.3f}, batch_time={:.3f}, learning_rate={:.9f}'.format(iter,
                epoch, cfg.TRAIN.MAX_EPOCH, train_loss_i2t, model_out['reward'], (end_t - start_t), current_lr))
       
        ###################################image caption test#################################
        # test captionging
#         if iter % evl_interval == 0 and iter != 0:
        if iter % evl_interval == 0 and epoch >= 10:
            predictions, lang_stats = eval_utils_cap.eval_split(dp_model_c, loader, cfg)
            lang_stats['epoch'] = epoch
            lang_stats_total.append(lang_stats)

            # Save model if is improving on validation result
            current_score = lang_stats['CIDEr']

            best_flag = False
            if True:  # if true
                if best_val_score_cap is None or current_score > best_val_score_cap:
                    best_val_score_cap = current_score
                    best_flag = True

                if sc_flag: LogDir = cfg.log_dir_sc
                else: LogDir = cfg.log_dir

                checkpoint_path = os.path.join(LogDir, 'model.pth')
                torch.save(dp_model_c.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                optimizer_path = os.path.join(LogDir, 'optimizer.pth')
                torch.save(optimizer.state_dict(), optimizer_path)

                # Dump miscalleous informations
                infos['split_ix'] = 'test'
                infos['opt'] = cfg
                infos['vocab'] = loader.get_vocab()                
                infos['lang_stats'] = lang_stats
                infos['lang_stats_total'] = lang_stats_total

                with open(os.path.join(LogDir, 'infos_' + cfg.DATASET_NAME + '.pkl'), 'wb') as f:
                    cPickle.dump(infos, f)

                if best_flag:
                    checkpoint_path = os.path.join(LogDir, 'model-best.pth')
                    torch.save(dp_model_c.state_dict(), checkpoint_path)
                    print("model saved to {}".format(checkpoint_path))
                    with open(os.path.join(LogDir, 'infos_' + cfg.DATASET_NAME + '-best.pkl'), 'wb') as f:
                        cPickle.dump(infos, f)

        # Stop if reaching max epochs
        if epoch > cfg.TRAIN.MAX_EPOCH and cfg.TRAIN.MAX_EPOCH != -1:
            break

