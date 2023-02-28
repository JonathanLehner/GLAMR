import os, sys
sys.path.append(os.path.join(os.getcwd()))
import os.path as osp
import glob
import argparse
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from motion_infiller.vis.vis_smpl import SMPLVisualizer
from motion_infiller.data.amass_dataset import AMASSDataset
from traj_pred.utils.config import Config
from traj_pred.models import model_dict
from lib.utils.tools import worker_init_fn, find_last_version, get_checkpoint_path
from lib.utils.vis import hstack_videos, hstack_3videos
from lib.utils.torch_utils import tensor_to


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default='trajv2_s300_ddec_heading_aa')
parser.add_argument('--split', default='test')
parser.add_argument('--seq_len', type=int, default=800)
parser.add_argument('--num_seq', type=int, default=2)
parser.add_argument('--num_motion_samp', type=int, default=5)
parser.add_argument('--multi_step', action='store_true', default=False)
parser.add_argument('--ntime', type=float, default=2e6)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--version', type=int, default=None)
parser.add_argument('--cp', default='best')
parser.add_argument('--noise_amount', type=float, default=0.01)
parser.add_argument('--no_vis', type=bool, default=True)
args = parser.parse_args()
args.no_vis = False

cfg = Config(args.cfg, training=False)
seed_everything(args.seed, workers=False)
device = torch.device('cuda', index=args.gpu) if torch.cuda.is_available() and args.gpu >= 0 else torch.device('cpu')
torch.torch.set_grad_enabled(False)

cfg.seq_sampling_method = 'length'
seq_len = cfg.seq_len if args.seq_len == -1 else args.seq_len

# val datasets
# J: make it equal
test_dataset = AMASSDataset(cfg.amass_dir, args.split, None, training=False, seq_len=seq_len, ntime_per_epoch=int(args.ntime))
#test_dataset = AMASSDataset(cfg.amass_dir, args.split, cfg, training=False, seq_len=seq_len, ntime_per_epoch=int(args.ntime))
test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)

# checkpoint
version = find_last_version(cfg.cfg_dir) if args.version is None else args.version
checkpoint_dir = f'{cfg.cfg_dir}/version_{version}/checkpoints'
model_cp, cp_name = get_checkpoint_path(checkpoint_dir, args.cp, return_name=True)
print(f'loading check point {model_cp}')

# model
traj_predictor = model_dict[cfg.model_name].load_from_checkpoint(model_cp, cfg=cfg)
traj_predictor.to(device)
traj_predictor.eval()

visualizer = SMPLVisualizer(generator_func=None, distance=17, device=device, sample_visible_alltime=True, show_smpl='all', verbose=False)

print("modified version only works if MOJO-cap has previously been run")
for i, batch in enumerate(test_dataloader):
    if i >= args.num_seq:
        print("maximum number of samples reached")
        break

    # num_drop_fr = batch["num_drop_fr"][0]
    print(f'{i}/{args.num_seq} seq_name: {batch["seq_name"][0]}, fr_start: {batch["fr_start"][0]}')
    
    batch = tensor_to(batch, device)
    seq_name = batch["seq_name"][0]

    if(args.no_vis == False): # J: we check here first so we don't need to do inference if file for visualisation is missing
        try: #J: make sure we already generated sequences
            mojoResults = torch.load(f'../MOJO-cap/results/{seq_name}_{args.noise_amount}_10.pt')
            print('loaded ')
            for key in mojoResults.keys():
                mojoResults[key] = mojoResults[key].float().cuda()

        except Exception as e:
            print(e)
            continue

    # J: compute noisy body feature
    batch['pose'][0,:,3:] = (batch['pose'][0,:,3:] + torch.randn(batch['pose'][0,:,3:].shape).cuda()*args.noise_amount).contiguous()
    #print(batch['pose'].shape)
    #exit()
    output = traj_predictor.inference(batch, sample_num=args.num_motion_samp, recon=True, multi_step=args.multi_step)
    
    # J: need to save for evaluation
    torch.save(output, f'out/vis_traj_pred/{seq_name}_{args.noise_amount}.pt')

    if(args.no_vis == False):
        # betas are always 0 in the visualiser
        mojoResults['shape'] = output['shape']
        mojoResults['frame_mask'] = output['frame_mask']
        mojoResults['seq_name'] = output['seq_name']

        prefix = 'multistep_' if args.multi_step else 'singlestep_'
        vid_name = f'out/vis_traj_pred/{cfg.id}/v{version}_{cp_name}/{args.split}/{prefix}seq_len_{seq_len}/sd{args.seed}_s{i}__{seq_name}_%s.mp4'

        # save MOJO-cap
        visualizer.save_animation_as_video(
            vid_name % 'MOJO-cap', init_args={'smpl_seq': mojoResults, 'mode': 'sample'}, window_size=(1500, 1500), cleanup=True
        )

        # save GT
        visualizer.save_animation_as_video(
            vid_name % 'gt', init_args={'smpl_seq': output, 'mode': 'gt'}, window_size=(1500, 1500), cleanup=True
        )

        # save sample
        visualizer.save_animation_as_video(
            vid_name % 'GLAMR-sample', init_args={'smpl_seq': output, 'mode': 'sample'}, window_size=(1500, 1500), cleanup=True
        )

        #hstack_videos(vid_name % 'recon', vid_name % 'gt', vid_name % 'sbs', verbose=False, text1='Recon', text2='GT', text_color='black')
        hstack_3videos(vid_name % 'GLAMR-sample', vid_name % 'gt', vid_name % 'MOJO-cap', vid_name % 'combined', verbose=False, text1='GLAMR', text2='GT', text3='MOJO-cap', text_color='black')

        if False: # J: currently not needed
            if traj_predictor.stochastic:
                # save sample
                visualizer.save_animation_as_video(
                    vid_name % 'sample', init_args={'smpl_seq': output, 'mode': 'sample'}, window_size=(1500, 1500), cleanup=True
                )
                #hstack_videos(vid_name % 'sample', vid_name % 'gt', vid_name % 'sample_sbs', verbose=False, text1='Sample', text2='GT', text_color='black')
                hstack_3videos(vid_name % 'GLAMR', vid_name % 'gt', vid_name % 'MOJO-cap', vid_name % 'combined', verbose=False, text1='Sample', text2='GT', text_color='black')
                os.remove(vid_name % 'sample')

        os.remove(vid_name % 'gt')
        os.remove(vid_name % 'GLAMR-sample')
        os.remove(vid_name % 'MOJO-cap')
    
    




