import argparse
import sys
import cv2
import numpy as np
from tqdm import tqdm
from getyaw import GetYaw
import os
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name", type=str, help="experiment folder name"
    )
    parser.add_argument(
        "--iters", type=str, help="ckpt iters"
    )
    parser.add_argument('--num_test', type=int, default=1000)
    args = parser.parse_args()
    args.dir = '../exps/' + args.exp_name + '/images_for_id'
    if os.path.exists(args.dir + '/simi.npy'):
        all_simi = np.load(args.dir + '/simi.npy')
    else:
        args.imagedir = args.dir + '/images'
        sys.path.append('./simi')
        from id_feature import IDFeatureNet
        from PIL import Image
        import torch
        device="cuda"
        id_feat = IDFeatureNet().to(device).eval()
        num_s = 5
        all_simi = np.zeros([args.num_test, num_s - 1])
        for i in tqdm(range(args.num_test)):
            img_rgb_batch = np.zeros([num_s, 3, 1024, 1024])
            for j in range(num_s):
                path = args.imagedir + '/' + str(i).zfill(6) + '_' + str(j) + '.png'
                img_data = np.asarray(Image.open(path))
                img_data = (img_data.astype(np.float32) - 127.5) / 127.5
                img_data = np.transpose(img_data, (2, 0, 1))
                img_rgb_batch[j] = img_data
            img_rgb_batch = torch.tensor(img_rgb_batch, dtype=torch.float32).cuda()
            res = id_feat(img_rgb_batch)
            simi = torch.sum(torch.mul(res[0], res[1:]), dim=1)
            all_simi[i] = simi.detach().cpu().numpy()
        np.save(args.dir + '/simi.npy', all_simi)

    if os.path.exists(args.dir + '/pose.npy'):
        all_yaws = np.load(args.dir + '/pose.npy')
    else:
        args.imagedir = args.dir + '/images'
        predyaw = GetYaw()
        num_s = 5
        all_yaws = np.zeros([args.num_test, num_s])
        for i in tqdm(range(args.num_test)):
            img_rgb_batch = np.zeros([num_s, 224, 224, 3])
            for j in range(num_s):
                path = args.imagedir + '/' + str(i).zfill(6) + '_' + str(j) + '.png'
                all_yaws[i][j] = predyaw.path_to_yaw(path)
        np.save(args.dir + '/pose.npy', all_yaws)

    angles = [20, 30, 40, 60, 75]
    num_angles = len(angles)
    ans = np.zeros([num_angles - 1, 2])
    for i in range(num_angles - 1):
        ag = angles[i]
        ag2 = angles[i+1]
        tst = (all_yaws - all_yaws[:,0:1])[:,1:]
        mask = (abs(tst) < ag2) & (abs(tst) > ag)
        cnt = mask[:args.num_test, ].sum()
        simi = (all_simi[:args.num_test] * mask[:args.num_test,]).sum()
        ans[i,0] = simi / cnt
        ans[i,1] = cnt
    print(ans)
    np.savetxt(args.dir + '/eval_id.txt', ans)
