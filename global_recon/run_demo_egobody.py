import os, sys
sys.path.append(os.path.join(os.getcwd()))

import os.path as osp
import glob
import torch
import numpy as np
import pickle
import cv2 as cv
import shutil
import argparse
from lib.utils.log_utils import create_logger
from lib.utils.vis import get_video_num_fr, get_video_fps, hstack_video_arr, get_video_width_height, video_to_images
from global_recon.utils.config import Config
from global_recon.models import model_dict
from global_recon.vis.vis_grecon import GReconVisualizer
from global_recon.vis.vis_cfg import demo_seq_render_specs as seq_render_specs
from pose_est.run_pose_est_demo import run_pose_est_on_video


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default='glamr_dynamic')  # glamr_dynamic
parser.add_argument('--est_type', default='hybrik')  # hybrik/prohmr
# parser.add_argument('--video_path', default='/local/home/szhang/MOJO-cap-multiperson/downtown_enterShop_00.mp4')   # ../assets/dynamic/running.mp4
# parser.add_argument('--out_dir', default='../out/glamr_dynamic/downtown_enterShop_00')  # glamr_dynamic/running
parser.add_argument('--video_path', default='recording_20210907_S03_S04_01_clip_01.mp4')
# parser.add_argument('--out_dir', default='/mnt/ssd/glamr_egobody_test/hybrik/recording_20210907_S03_S04_01_clip_01')
parser.add_argument('--out_dir', default='./out/egobody_test_prohmr/recording_20210911_S03_S08_01_clip_01')
parser.add_argument('--pose_est_dir', default='./out/egobody_test/recording_20210911_S03_S08_01_clip_01/pose_est')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--cached', type=int, default=0)
parser.add_argument('--multi', action='store_true', default=False)
parser.add_argument('--vis', action='store_true', default=False)
parser.add_argument('--vis_cam', action='store_true', default=False)
# parser.add_argument('--save_video', action='store_true', default=False)
parser.add_argument('--save_video', default='True', type=lambda x: x.lower() in ['true', '1'])
args = parser.parse_args()


egobody_video_root = '/local/home/szhang/GLAMR-main/egobody_test_data/egobody_test_videos'
video_path = os.path.join(egobody_video_root, args.video_path)
# video_path = args.video_path




cached = int(args.cached)
cfg = Config(args.cfg, out_dir=args.out_dir)
if torch.cuda.is_available() and args.gpu >= 0:
    device = torch.device('cuda', index=args.gpu)
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device('cpu')

cfg.grecon_model_specs['est_type']= args.est_type

cfg.save_yml_file(f'{args.out_dir}/config.yml')
log = create_logger(f'{cfg.log_dir}/log.txt')
grecon_path = f'{args.out_dir}/grecon'
render_path = f'{args.out_dir}/grecon_videos'
os.makedirs(grecon_path, exist_ok=True)
os.makedirs(render_path, exist_ok=True)


####################### run hybrik for stage 1 initialization
if args.est_type == 'hybrik':
    if args.pose_est_dir is None:
        pose_est_dir = f'{args.out_dir}/pose_est'
        log.info(f"running {cfg.grecon_model_specs['est_type']} pose estimation on {video_path}...")
        run_pose_est_on_video(video_path, pose_est_dir, cfg.grecon_model_specs['est_type'], 1, args.gpu, args.multi)
    else:
        pose_est_dir = args.pose_est_dir
    pose_est_model_name = {'hybrik': 'HybrIK'}[cfg.grecon_model_specs['est_type']]
elif args.est_type == 'prohmr':
    assert("_prohmr" in args.out_dir)
    pose_est_dir = f'{args.out_dir}/pose_est'
    pose_est_model_name = 'prohmr'



################### global recon model
grecon_model = model_dict[cfg.grecon_model_name](cfg, device, log)

seed = args.seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)

log.info(f'running global reconstruction on {video_path}, seed: {seed}')
seq_name = osp.splitext(osp.basename(video_path))[0]
pose_est_file = f'{pose_est_dir}/pose.pkl'  # read stage 1 init results from hybrik
pose_est_video = f'{pose_est_dir}/render.mp4'  # read stage 1 init results from hybrik
img_w, img_h = get_video_width_height(video_path)
num_fr = get_video_num_fr(video_path)
fps = get_video_fps(video_path)

################# global optimization
# seq_name, person_data (intrinsics, smpl params, vis_frame, root_trans_world, person2cam, ...)
out_file = f'{grecon_path}/{seq_name}_seed{seed}.pkl'
if cached and osp.exists(out_file):
    out_dict = pickle.load(open(out_file, 'rb'))
else:
    est_dict = pickle.load(open(pose_est_file, 'rb'))
    temp_dict = {}
    if args.multi:
        for person_id in range(len(est_dict)):
            temp_dict[person_id] = est_dict[person_id]
        est_dict = temp_dict
    in_dict = {'est': est_dict, 'gt': dict(), 'gt_meta': dict(), 'seq_name': seq_name}
    # global optimization
    out_dict = grecon_model.optimize(in_dict)
    pickle.dump(out_dict, open(out_file, 'wb'))  # save glamr results

if (args.vis and args.vis_cam) or args.save_video:
    frame_dir = f'{pose_est_dir}/frames'
    if len(glob.glob(f'{frame_dir}/*.jpg')) != out_dict['meta']['num_fr']:
        log.info(f'generating frames from {video_path}...')
        video_to_images(video_path, frame_dir, fps=30, verbose=False)

# # visualization
# if args.vis:
#     render_specs = seq_render_specs.get(seq_name, dict())
#     if args.vis_cam:
#         visualizer = GReconVisualizer(out_dict, coord='cam_in_world', verbose=False, background_img_dir=frame_dir)
#         visualizer.show_animation(window_size=(img_w, img_h), show_axes=False)
#     else:
#         render_specs = seq_render_specs.get(seq_name, seq_render_specs['default'])
#         visualizer = GReconVisualizer(out_dict, coord='world', verbose=False, show_camera=True,
#                                       render_cam_pos=render_specs.get('cam_pos', None), render_cam_focus=render_specs.get('cam_focus', None))
#         visualizer.show_animation(window_size=(1920, 1080))

# save video
if args.save_video:
    render_specs = seq_render_specs.get(seq_name, seq_render_specs['default'])
    video_world = f'{render_path}/{seq_name}_seed{seed}_world.mp4'
    video_cam = f'{render_path}/{seq_name}_seed{seed}_cam.mp4'
    video_sbs = f'{render_path}/{seq_name}_seed{seed}_sbs_all.mp4'


    log.info(f'saving cam animation for Seq {seq_name}')
    visualizer = GReconVisualizer(out_dict, coord='cam_in_world', verbose=False, background_img_dir=frame_dir)
    visualizer.save_animation_as_video(video_cam, window_size=(img_w, img_h), cleanup=True)

    log.info(f'saving world animation for Seq {seq_name} ')
    visualizer = GReconVisualizer(out_dict, coord='world', verbose=False, show_camera=False,
                                  render_cam_pos=render_specs.get('cam_pos', None), render_cam_focus=render_specs.get('cam_focus', None))
    visualizer.save_animation_as_video(video_world, window_size=render_specs.get('wsize', (int(1.5 * img_h), img_h)), cleanup=True, crf=5)

    log.info(f'saving side-by-side animation for Seq {seq_name}')
    hstack_video_arr([video_cam, video_world], video_sbs, text_arr=['GLAMR (Cam)', 'GLAMR (World)'], text_color='blue', text_size=img_h // 16, verbose=False)
    hstack_video_arr([pose_est_video, video_cam, video_world], video_sbs, text_arr=[pose_est_model_name, 'GLAMR (Cam)', 'GLAMR (World)'], text_color='blue', text_size=img_h // 16, verbose=False)
