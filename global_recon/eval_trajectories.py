import os, sys
sys.path.append(os.path.join(os.getcwd()))
sys.path.append(os.path.join(os.getcwd(), 'kama'))
import os.path as osp
import pickle
import argparse
import numpy as np
import torch
from global_recon.utils.evaluator_traj import Evaluator
import glob
from torch.utils.data import DataLoader
from motion_infiller.data.amass_dataset import AMASSDataset
from lib.utils.tools import worker_init_fn, find_last_version, get_checkpoint_path

results_dir = './out/vis_traj_pred'
seeds = range(1)
multi_seeds = len(seeds) > 1

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
torch.torch.set_grad_enabled(False)

models = ["MOJO", "GLAMR"]
models = ["GLAMR"]
models = ["MOJO"]

test_dataset = AMASSDataset('../GLAMR/datasets/amass_processed/v1', 'test', None, training=False, seq_len=800, ntime_per_epoch=int(2e6))
test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)

for model in models:

    evaluator = Evaluator(results_dir, 'amass', device=device, log_file=f'{results_dir}/log_traj_{model}.txt', compute_sample=multi_seeds)
    seed_evaluator = Evaluator(results_dir, 'amass', device=device, log_file=f'{results_dir}/log_traj_seed_{model}.txt', compute_sample=multi_seeds)

    for sind, batch in enumerate(test_dataloader):

        seq_name = batch["seq_name"][0]
        metrics_dict_arr = []

        try: #J: make sure we already generated sequences
            if(model == "GLAMR"):
                results = torch.load(f'out/vis_traj_pred/{seq_name}.pt')
            else:
                results = torch.load(f'../MOJO-cap/results/{seq_name}.pt')
                for key in results.keys():
                    results[key] = results[key].float().cuda()
        except Exception as e:
            # print(e)
            continue

        evaluator.log.info(f'{sind}/{len(test_dataloader)} evaluating trajectory prediction for {seq_name}')
        
        batch['visible'] = torch.ones(800)
        batch['visible_orig'] = batch['visible'].clone()
        batch['root_trans'] = batch['trans'][0].clone()
        batch['pose'] = batch['pose'][0]
        batch['trans'] = batch['trans'][0]

        ## for predicted data
        batch['smpl_orient_world'] = batch['pose'][:,:3]
        batch['smpl_pose'] = batch['pose'][:,3:]
        batch['smpl_beta'] = batch['shape'][0]
        batch['root_trans_world'] = batch['trans']
        batch['scale'] = None

        # print(batch['smpl_orient_world'].shape, batch['smpl_pose'].shape, batch['root_trans_world'].shape)

        batch['shape'] = batch['shape'][0, 0] # J: this should be 1,10 in AMASS but somehow it is 1,800,10

        data = {}
        data["seq_len"] = 800
        data["gt"] = {0: batch}
        data_file = f'{results_dir}/{seq_name}.pt'

        #print(batch['smpl_pose'].shape)

        try:
            data_all_ours = torch.load(open(data_file, 'rb'))
        except:
            continue

        for seed in seeds:
            print("seed ", seed)

            batch['smpl_orient_world'] = results['infer_out_orient'][0, seed]
            batch['smpl_pose'] = results['infer_out_pose'][0, seed, :, 3:]
            batch['root_trans_world'] = results['infer_out_trans'][0, seed]

            data["person_data"] = {0: batch}
            metrics_dict = seed_evaluator.compute_sequence_metrics(data, seq_name, accumulate=False)
            metrics_dict_arr.append(metrics_dict)

        metrics_dict_allseeds = evaluator.metrics_from_multiple_seeds(metrics_dict_arr)
        evaluator.update_accumulated_metrics(metrics_dict_allseeds, seq_name)
        evaluator.print_metrics(metrics_dict_allseeds, prefix=f'{sind}/{len(test_dataloader)} --- All seeds {seq_name} --- ', print_accum=False)

    evaluator.print_metrics(prefix=f'Total ------- ', print_accum=True)