import os, sys
sys.path.append(os.path.join(os.getcwd()))
sys.path.append(os.path.join(os.getcwd(), 'kama'))
import os.path as osp
import pickle
import argparse
import numpy as np
import torch
import pickle as pkl
from global_recon.utils.evaluator_egobody import EvaluatorEgobody


# test_sequences = {
#     '3dpw': ['downtown_arguing_00', 'downtown_bar_00', 'downtown_bus_00', 'downtown_cafe_00', 'downtown_car_00', 'downtown_crossStreets_00', 'downtown_downstairs_00',
#              'downtown_enterShop_00', 'downtown_rampAndStairs_00', 'downtown_runForBus_00', 'downtown_runForBus_01', 'downtown_sitOnStairs_00', 'downtown_stairs_00',
#              'downtown_upstairs_00', 'downtown_walkBridge_01', 'downtown_walkUphill_00', 'downtown_walking_00', 'downtown_warmWelcome_00', 'downtown_weeklyMarket_00',
#              'downtown_windowShopping_00', 'flat_guitar_01', 'flat_packBags_00', 'office_phoneCall_00', 'outdoors_fencing_01']
# }

parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', default='3dpw')
parser.add_argument('--results_dir', default='/mnt/ssd/glamr_egobody_test')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--init_type', default='prohmr')  # hybrik/prohmr
# parser.add_argument('--seeds', default="1")
args = parser.parse_args()

results_dir = args.results_dir

egobody_test_clip_names = pkl.load(open('../egobody_test_data/egobody_test_clip_names.pkl', 'rb'))
sequences = list(egobody_test_clip_names.keys())

# sequences = ['recording_20220225_S24_S25_01_clip_01']

# seeds = [int(x) for x in args.seeds.split(',')]
# multi_seeds = len(seeds) > 1

if torch.cuda.is_available() and args.gpu >= 0:
    device = torch.device('cuda', index=args.gpu)
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device('cpu')
torch.torch.set_grad_enabled(False)

evaluator = EvaluatorEgobody(results_dir, device=device)
seed_evaluator = EvaluatorEgobody(results_dir, device=device)



final_g_mpjpe, final_g_pve, final_pa_mpjpe, final_accel_err = 0, 0, 0, 0
frame_cnt = 0
frame_accel_cnt = 0
for sind, seq_name in enumerate(sequences):
    metrics_dict_arr = []
    # evaluator.log.info(f'{sind}/{len(sequences)} evaluating global reconstruction for {seq_name}')

    pred_data_file = f'{results_dir}/{args.init_type}/{seq_name}/grecon/{seq_name}_seed1.pkl'
    gt_data_file = '../egobody_test_data/egobody_test_gt/{}.pkl'.format(seq_name)
    pred_data = pickle.load(open(pred_data_file, 'rb'))
    gt_pose_dict = pickle.load(open(gt_data_file, 'rb'))

    metrics_dict = seed_evaluator.compute_sequence_metrics(pred_data, gt_pose_dict, seq_name, accumulate=False)
    print('seq_name:{}, seq_len: {}, mean_g_mpjpe: {}, mean_g_pve: {}, mean_pa_mpjpe: {}, mean_accel_err: {}'.
          format(metrics_dict['seq_name'], metrics_dict['seq_len'], metrics_dict['mean_g_mpjpe'],
                 metrics_dict['mean_g_pve'], metrics_dict['mean_pa_mpjpe'], metrics_dict['mean_accel_err']))

    final_g_mpjpe += metrics_dict['mean_g_mpjpe'] * metrics_dict['seq_len']
    final_g_pve += metrics_dict['mean_g_pve'] * metrics_dict['seq_len']
    final_pa_mpjpe += metrics_dict['mean_pa_mpjpe'] * metrics_dict['seq_len']
    final_accel_err += metrics_dict['mean_accel_err'] * (metrics_dict['seq_len'] - 2)
    frame_cnt += metrics_dict['seq_len']
    frame_accel_cnt += metrics_dict['seq_len'] - 2

    # metrics_dict_arr.append(metrics_dict)
    # metrics_dict_allseeds = evaluator.metrics_from_multiple_seeds(metrics_dict_arr)
    # evaluator.update_accumulated_metrics(metrics_dict_allseeds, seq_name)
    # evaluator.print_metrics(metrics_dict_allseeds, prefix=f'{sind}/{len(sequences)} --- All seeds {seq_name} --- ', print_accum=False)

# evaluator.print_metrics(prefix=f'Total ------- ', print_accum=True)
print('-----------------------------')
print('[FINAL results]')
print('global_mpjpe:', final_g_mpjpe / frame_cnt)
print('global_pve:', final_g_pve / frame_cnt)
print('pa_mpjpe:', final_pa_mpjpe / frame_cnt)
print('accel_err:', final_accel_err / frame_accel_cnt)
