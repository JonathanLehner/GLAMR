import subprocess
import os, sys
sys.path.append(os.path.join(os.getcwd()))
import os.path as osp
import argparse
import torch
import numpy as np
import pickle
import glob

from lib.utils.log_utils import create_logger
from lib.utils.vis import get_video_num_fr, get_video_fps, hstack_video_arr, get_video_width_height, video_to_images
from global_recon.utils.config import Config
from global_recon.models import model_dict
from global_recon.vis.vis_grecon import GReconVisualizer
from global_recon.vis.vis_cfg import demo_seq_render_specs as seq_render_specs
from pose_est.run_pose_est_demo import run_pose_est_on_video

test_sequences = {
    '3dpw': ['downtown_arguing_00', 'downtown_bar_00', 'downtown_bus_00', 'downtown_cafe_00', 'downtown_car_00', 'downtown_crossStreets_00', 'downtown_downstairs_00', 
             'downtown_enterShop_00', 'downtown_rampAndStairs_00', 'downtown_runForBus_00', 'downtown_runForBus_01', 'downtown_sitOnStairs_00', 'downtown_stairs_00',
             'downtown_upstairs_00', 'downtown_walkBridge_01', 'downtown_walkUphill_00', 'downtown_walking_00', 'downtown_warmWelcome_00', 'downtown_weeklyMarket_00',
             'downtown_windowShopping_00', 'flat_guitar_01', 'flat_packBags_00', 'office_phoneCall_00', 'outdoors_fencing_01'],
    'h36m': list(sorted(glob.glob('datasets/H36M/processed_v1/pose/s_09*.pkl')) + sorted(glob.glob('datasets/H36M/processed_v1/pose/s_11*.pkl'))),
    'h36m_dyn': list(sorted(glob.glob('datasets/H36M/processed_v1/pose/s_09*.pkl')) + sorted(glob.glob('datasets/H36M/processed_v1/pose/s_11*.pkl'))),
}

test_sequences = {
    '3dpw': ['downtown_arguing_00', 'downtown_bar_00', 'downtown_bus_00', 'downtown_cafe_00', 'downtown_car_00', 'downtown_crossStreets_00', 'downtown_downstairs_00', 
             'downtown_enterShop_00', 'downtown_rampAndStairs_00', 'downtown_runForBus_00', 'downtown_runForBus_01', 'downtown_sitOnStairs_00', 'downtown_stairs_00',
             'downtown_upstairs_00', 'downtown_walkBridge_01', 'downtown_walkUphill_00', 'downtown_walking_00', 'downtown_warmWelcome_00', 'downtown_weeklyMarket_00',
             'downtown_windowShopping_00', 'flat_guitar_01', 'flat_packBags_00', 'office_phoneCall_00', 'outdoors_fencing_01'],
    'h36m': list(sorted(glob.glob('datasets/H36M/processed_v1/pose/s_09*.pkl')) + sorted(glob.glob('datasets/H36M/processed_v1/pose/s_11*.pkl')))
}

dataset_paths_dict = {
    '3dpw': {
        'image': 'datasets/3DPW/imageFiles',
        'bbox': 'datasets/3DPW/processed_v1/bbox',
        'gt_pose': 'datasets/3DPW/processed_v1/pose'
    },
    'h36m': {
        'image': 'datasets/H36M/images_25fps',
        'bbox': 'datasets/H36M/processed_v1/bbox',
        'gt_pose': 'datasets/H36M/processed_v1/pose'
    },
    'h36m_dyn': {
        'image': 'datasets/H36M/occluded_v2/images',
        'bbox': 'datasets/H36M/occluded_v2/bbox',
        'gt_pose': 'datasets/H36M/occluded_v2/pose'
    }

}


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='3dpw')
parser.add_argument('--cfg', default='glamr_3dpw')
parser.add_argument('--out_dir', default='out/3dpw')
parser.add_argument('--seeds', default="1")
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--cached', type=int, default=1)
parser.add_argument('--save_video', action='store_true', default=True)
args = parser.parse_args()


cached = int(args.cached)
cfg = Config(args.cfg, out_dir=args.out_dir)
if torch.cuda.is_available() and args.gpu >= 0:
    device = torch.device('cuda', index=args.gpu)
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device('cpu')

seeds = [int(x) for x in args.seeds.split(',')]
sequences = test_sequences[args.dataset]
dataset_paths = dataset_paths_dict[args.dataset]

# global recon model
grecon_model = model_dict[cfg.grecon_model_name](cfg, device, None)

for i, seq_name in enumerate(sequences[:]):
    for seed in seeds:
        print(f'{i}/{len(sequences)} seed {seed} processing {seq_name} for {args.dataset}..')

        if(args.dataset == "h36m" or args.dataset == "h36m_dyn"):
            seq_name = os.path.basename(seq_name).split(".")[0]
        seq_image_dir = f"{dataset_paths['image']}/{seq_name}"

        seq_out_dir = f"{args.out_dir}/{seq_name}"
        seq_bbox_file = f"{dataset_paths['bbox']}/{seq_name}.pkl"
        seq_gt_pose_file = f"{dataset_paths['gt_pose']}/{seq_name}.pkl"
        if seq_gt_pose_file is None:
            gt_dict = dict()
        else:
            gt_dict = pickle.load(open(seq_gt_pose_file, 'rb'))

        # J: need absolute path for other script
        seq_gt_pose_file = os.path.abspath(seq_gt_pose_file)

        cfg.save_yml_file(f'{seq_out_dir}/config.yml')
        grecon_model.log = log = create_logger(f'{cfg.cfg_dir}/log.txt')
        grecon_path = f'{seq_out_dir}/grecon'
        render_path = f'{seq_out_dir}/grecon_videos'
        os.makedirs(grecon_path, exist_ok=True)
        os.makedirs(render_path, exist_ok=True)

        pose_est_dir = f'{seq_out_dir}/pose_est'
        log.info(f"running {cfg.grecon_model_specs['est_type']} pose estimation on {seq_image_dir}...")

        run_pose_est_on_video(None, pose_est_dir, cfg.grecon_model_specs['est_type'], image_dir=seq_image_dir, bbox_file=seq_bbox_file, cached_pose=cached, gpu_index=args.gpu, dataset_path=seq_gt_pose_file)
        print("J: run pose est finished")
        pose_est_model_name = {'hybrik': 'HybrIK'}[cfg.grecon_model_specs['est_type']]

        np.random.seed(seed)
        torch.manual_seed(seed)

        pose_est_file = f'{pose_est_dir}/pose.pkl'
        log.info(f'running global reconstruction on {seq_image_dir}, seed: {seed}')
        seq_name = osp.basename(seq_image_dir)

        # main
        out_file = f'{grecon_path}/{seq_name}_seed{seed}.pkl'

        est_dict = pickle.load(open(pose_est_file, 'rb'))
        
        in_dict = {'est': est_dict, 'gt': gt_dict['person_data'], 'gt_meta': gt_dict['meta'], 'seq_name': seq_name}
            
        #print("only generating data - skipping optimisation")
        #continue

        # global optimization
        out_dict = grecon_model.optimize(in_dict)
        pickle.dump(out_dict, open(out_file, 'wb'))

        # save video
        if args.save_video:
            pose_est_video = f'{seq_out_dir}/pose_est/render.mp4'
            img_w, img_h = get_video_width_height(pose_est_video)

            render_specs = seq_render_specs.get(seq_name, seq_render_specs['default'])
            video_world = f'{render_path}/{seq_name}_seed{seed}_world.mp4'
            video_cam = f'{render_path}/{seq_name}_seed{seed}_cam.mp4'
            video_sbs = f'{render_path}/{seq_name}_seed{seed}_sbs_all.mp4'

            log.info(f'saving world animation for {seq_name}')
            visualizer = GReconVisualizer(out_dict, coord='world', verbose=False, show_camera=False,
                                        render_cam_pos=render_specs.get('cam_pos', None), render_cam_focus=render_specs.get('cam_focus', None))
            #J: gives not divisible by 2 error in H36M, visualizer.save_animation_as_video(video_world, window_size=render_specs.get('wsize', (int(1.5 * img_h), img_h)), cleanup=True, crf=5)
            visualizer.save_animation_as_video(video_world, window_size=render_specs.get('wsize', (img_w, img_h)), cleanup=True, crf=5)

            log.info(f'saving cam animation for {seq_name}')
            visualizer = GReconVisualizer(out_dict, coord='cam_in_world', verbose=False, background_img_dir=frame_dir)
            visualizer.save_animation_as_video(video_cam, window_size=(img_w, img_h), cleanup=True)

            log.info(f'saving side-by-side animation for {seq_name}')
            hstack_video_arr([pose_est_video, video_cam, video_world], video_sbs, text_arr=["Hybrik", 'GLAMR (Cam)', 'GLAMR (World)'], text_color='blue', text_size=img_h // 16, verbose=False)


