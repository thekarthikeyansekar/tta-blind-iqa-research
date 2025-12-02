from __future__ import print_function
import argparse
import random
import torchvision
import torchvision.transforms as T
from scipy import stats
from tqdm import tqdm
import copy
import torch.utils.data as data
import os.path
from utils import *
from rotation import *
from models import Net
import os
import time
import csv
from datetime import datetime

# -----------------------------
# Adaptive Margin Ranking Loss
# -----------------------------
class AdaptiveMarginRankLoss(nn.Module):
    """
    Implements:
    L_AMRL = (1/|P|) * sum_{(i,j) in P} max(0, m_ij - (dist_high - dist_low))
    where m_ij = gamma * sigmoid(|q_high - q_low|)
    """

    def __init__(self, gamma=0.5, eps=1e-8):
        print("AdaptiveMarginRankLoss initialized")
        super(AdaptiveMarginRankLoss, self).__init__()
        self.gamma = float(gamma)
        self.eps = eps

    def forward(self, dist_high, dist_low, q_high, q_low):
        """
        dist_high, dist_low: tensors shape (batch,) or (batch,1)
        q_high, q_low: predicted quality scores tensors shape (batch,) or (batch,1)
        Returns:
            loss scalar (tensor)
        """
        # Flatten
        dist_high = dist_high.view(-1)
        dist_low = dist_low.view(-1)
        q_high = q_high.view(-1)
        q_low = q_low.view(-1)

        # adaptive margin per pair
        abs_diff = torch.abs(q_high - q_low)
        m_ij = self.gamma * torch.sigmoid(abs_diff)

        # margin violation
        margin_violation = m_ij - (dist_high - dist_low)
        loss_per_pair = torch.clamp(margin_violation, min=0.0)

        if loss_per_pair.numel() == 0:
            return dist_high.sum() * 0.0
            # torch.tensor(0.0, device=dist_high.device, requires_grad=True)
        loss = loss_per_pair.mean()
        return loss

# -----------------------------
# DataLoader & Folder
# -----------------------------
class DataLoader(object):
    """
    Dataset class for IQA databases
    """

    def __init__(self,config, path, img_indx, patch_size, patch_num, batch_size=1):

        self.batch_size = batch_size
        self.config=config

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(size=patch_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])

        self.data = Folder(self.config,root=path, index=img_indx, transform=transforms, patch_num=patch_num)


    def get_data(self):
        dataloader = torch.utils.data.DataLoader(self.data, batch_size=self.batch_size, shuffle=True)
        return dataloader

class Folder(data.Dataset):
    def __init__(self, config, root, index, transform, patch_num):

        csv_path = os.path.join(root, 'MOS.csv')
        df = pd.read_csv(csv_path)

        # df['0'] = df['0'].apply(lambda x: root + '/' + x)
        # dataset = df['0'].tolist()
        # labels = df['1'].tolist()

        df['image_name'] = df['image_name'].apply(lambda x: root + '/' + x)
        dataset = df['image_name'].tolist()
        labels = df['MOS'].tolist()
        sample = []
        self.root = root
        self.config = config
        for item, i in enumerate(index):
            for aug in range(patch_num):
                sample.append((dataset[i], labels[i]))
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        data_dict = {}

        if self.transform is not None:
            data_dict['image'] = self.transform(sample)

        path, target = self.samples[index]
        sample = pil_loader(path)
        data_dict = {}

        if self.transform is not None:
            data_dict['image'] = self.transform(sample)

        data_dict = processing(data_dict, sample, self.transform, self.root, path, self.config)

        return data_dict, target

    def __len__(self):
        length = len(self.samples)
        return length

class Model(object):

    def __init__(self, config, device, Net):
        super(Model, self).__init__()

        if not config.tta_tf:   #if we want to adapt transformer also
            from SSHead import head_on_layer2, ExtractorHead, extractor_from_layer2
        else:
            from SSHead_tf import head_on_layer2, ExtractorHead, extractor_from_layer2

        self.device = device
        self.test_patch_num = config.test_patch_num
        self.l1_loss = torch.nn.L1Loss()
        self.lr = config.lr
        self.rank_loss = nn.BCELoss()

        # Adaptive Margin Rank - Start
        self.amrl = None
        self.adaptive_gamma = getattr(config, 'adaptive_gamma', 0.5)
        if config.adaptive_margin_rank:
            self.amrl = AdaptiveMarginRankLoss(gamma=self.adaptive_gamma)
        self.global_step = 0
        # Adaptive Margin Rank - End

        self.net = Net(config, device).to(device)
        self.head = head_on_layer2(config)
        self.ext = extractor_from_layer2(self.net)
        self.ssh = ExtractorHead(self.ext, self.head).cuda()

        self.config = config
        self.clsloss = nn.CrossEntropyLoss()

        self.optimizer_ssh = torch.optim.Adam(self.ext.parameters(), lr=self.lr)
        if not config.fix_ssh:
            self.optimizer_ssh = torch.optim.Adam(self.ssh.parameters(), lr=self.lr)


    def test(self, data, pretrained=0):
        if pretrained:
            self.net.load_state_dict(torch.load(self.config.svpath))

        self.net.eval()

        pred_scores = []
        gt_scores = []

        srcc=np.zeros(len(data))
        plcc=np.zeros(len(data))


        with torch.no_grad():
            steps2 = 0

            for data_dict, label in tqdm(data, leave=False):

                img = data_dict['image']
                img = torch.as_tensor(img.to(self.device))
                label = torch.as_tensor(label.to(self.device))
                pred, _ = self.net(img)

                pred_scores = pred_scores + pred.cpu().tolist()
                gt_scores = gt_scores + label.cpu().tolist()

                steps2 += 1

                try:
                    pred_scores4 = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
                    gt_scores4 = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)
                except:
                    pred_scores4 = np.mean(np.reshape(np.array(pred_scores), (-1, 1)), axis=1)
                    gt_scores4 = np.mean(np.reshape(np.array(gt_scores), (-1, 1)), axis=1)

                test_srcc4, _ = stats.spearmanr(pred_scores4, gt_scores4)
                test_plcc4, _ = stats.pearsonr(pred_scores4, gt_scores4)

                srcc[steps2 - 1]=test_srcc4
                plcc[steps2 - 1]=test_plcc4


                if steps2%50==0:

                    print('After {} images test_srcc : {} \n test_plcc:{}'.format(steps2, test_srcc4, test_plcc4))

        try:
            pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
            gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)
        except:
            pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, 1)), axis=1)
            gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, 1)), axis=1)

        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)

        return test_srcc, test_plcc,srcc,plcc

    def adapt(self, data_dict, config, old_net):

        inputs = data_dict['image']

        f_low = []
        f_high = []
        q_high = None
        q_low = None
        dist_high = None
        dist_low = None

        print(f"[adapt][config.adaptive_margin_rank] = {config.adaptive_margin_rank}")

        with torch.no_grad():
            pred0, _ = old_net(data_dict['image'].cuda())

            if config.rank:

                sigma1 = 40 + np.random.random() * 20
                sigma2 = 5 + np.random.random() * 15

                data_dict['blur_high'] = T.GaussianBlur(kernel_size=(5, 5), sigma=(sigma1))(inputs).cuda()
                data_dict['blur_low'] = T.GaussianBlur(kernel_size=(5, 5), sigma=(sigma2))(inputs).cuda()

                id_dict = {0: data_dict['comp_high'], 1: data_dict['comp_low'], 2: data_dict['nos_high'],
                           3: data_dict['nos_low'], 4: data_dict['blur_high'], 5: data_dict['blur_low']}

                pred1, _ = old_net(data_dict['comp_high'].cuda())
                pred2, _ = old_net(data_dict['comp_low'].cuda())

                pred3, _ = old_net(data_dict['nos_high'].cuda())
                pred4, _ = old_net(data_dict['nos_low'].cuda())

                pred5, _ = old_net(data_dict['blur_high'].cuda())
                pred6, _ = old_net(data_dict['blur_low'].cuda())

                try:
                    comp = torch.unsqueeze(torch.abs(pred2 - pred1), dim=1)
                except:
                    comp = (torch.ones(1, 1) * (torch.abs(pred2 - pred1)).item()).cuda()

                try:
                    nos = torch.unsqueeze(torch.abs(pred4 - pred3), dim=1)
                except:
                    nos = (torch.ones(1, 1) * (torch.abs(pred4 - pred3)).item()).cuda()

                try:
                    blur = torch.unsqueeze(torch.abs(pred6 - pred5), dim=1)
                except:
                    blur = (torch.ones(1, 1) * (torch.abs(pred6 - pred5)).item()).cuda()


                all_diff = torch.cat([comp, nos, blur], dim=1)

                for p in range(len(pred0)):
                    if all_diff[p].argmax().item() == 0:
                        f_low.append(id_dict[1][p].cuda())
                        f_high.append(id_dict[0][p].cuda())
                        # print('comp', end=" ")
                    if all_diff[p].argmax().item() == 1:
                        f_low.append(id_dict[3][p].cuda())
                        f_high.append(id_dict[2][p].cuda())
                        # print('nos', end=" ")
                    if all_diff[p].argmax().item() == 2:
                        f_low.append(id_dict[5][p].cuda())
                        f_high.append(id_dict[4][p].cuda())
                        # print('blur', end=" ")

                f_low = torch.squeeze(torch.stack(f_low), dim=1)
                f_high = torch.squeeze(torch.stack(f_high), dim=1)
            if config.adaptive_margin_rank:
                print(f"[AMRLConfig] step={self.global_step}")

                sigma1 = 40 + np.random.random() * 20
                sigma2 = 5 + np.random.random() * 15

                data_dict['blur_high'] = T.GaussianBlur(kernel_size=(5, 5), sigma=(sigma1))(inputs).cuda()
                data_dict['blur_low'] = T.GaussianBlur(kernel_size=(5, 5), sigma=(sigma2))(inputs).cuda()

                q_high, _ = old_net(data_dict['blur_high'].cuda())
                q_low, _ = old_net(data_dict['blur_low'].cuda())

                f_low = data_dict['blur_low']
                f_high = data_dict['blur_high']

                f_neg_feat = self.ssh(f_low)
                f_pos_feat = self.ssh(f_high)
                f_actual = self.ssh(inputs.cuda())

                dist_high = torch.nn.PairwiseDistance(p=2)(f_pos_feat, f_actual)
                dist_low = torch.nn.PairwiseDistance(p=2)(f_neg_feat, f_actual)
        if config.comp:
            f_low = data_dict['comp_low'].cuda()
            f_high = data_dict['comp_high'].cuda()
        if config.nos:
            f_low = data_dict['nos_low'].cuda()
            f_high = data_dict['nos_high'].cuda()
        if config.blur:

            sigma2 = 40 + np.random.random() * 20
            sigma1 = 5 + np.random.random() * 15

            f_low = T.GaussianBlur(kernel_size=(5, 5), sigma=(sigma1))(inputs).cuda()
            f_high = T.GaussianBlur(kernel_size=(5, 5), sigma=(sigma2))(inputs).cuda()


        if config.contrastive or  config.contrique:
            f_low = data_dict['image1'].cuda()
            f_high = data_dict['image2'].cuda()

        m = nn.Sigmoid()

        for param in self.ssh.parameters():
            param.requires_grad = False

        for layer in self.ssh.ext.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.requires_grad_(True)
            if config.tta_tf:
                if not config.bn:
                    if isinstance(layer, nn.BatchNorm2d):
                        layer.requires_grad_(False)
            if config.ln:
                if isinstance(layer, nn.LayerNorm):
                    layer.requires_grad_(True)

        if config.fix_ssh:
            self.ssh.eval()
            self.ssh.ext.train()
        else:
            self.ssh.train()

        loss_hist = []

        for iteration in range(config.niter):

            target = torch.ones(inputs.shape[0]).cuda()

            if config.rank or config.blur or config.comp or config.nos:
                f_neg_feat = self.ssh(f_low)
                f_pos_feat = self.ssh(f_high)
                f_actual = self.ssh(inputs.cuda())

                dist_high = torch.nn.PairwiseDistance(p=2)(f_pos_feat, f_actual)
                dist_low = torch.nn.PairwiseDistance(p=2)(f_neg_feat, f_actual)

                loss = self.rank_loss(m(dist_high - dist_low), target)
                tmp_loss = self.rank_loss(m(dist_high - dist_low), target)
                
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                with open(args.logs_csv_path, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, iteration, "rank", float(tmp_loss.detach().cpu().item())])
                    f.flush()
            
            # Adaptive Margin Rank Loss - Start
            print(f"[DEBUG] adaptive_margin_rank = {config.adaptive_margin_rank}, amrl = {self.amrl is not None}, q_high = {q_high is not None}, q_low = {q_low is not None}")
            if config.adaptive_margin_rank and (self.amrl is not None) and (q_high is not None) and (q_low is not None):
                # q_high and q_low are predicted scores from old_net (as tensors)
                print(f"[AMRLLoss] step={self.global_step}")
                q_high_vec = q_high.view(-1).to(dist_high.device)
                q_low_vec = q_low.view(-1).to(dist_low.device)

                ### Start
                # diagnostics BEFORE computing mean loss
                abs_diff = torch.abs(q_high_vec - q_low_vec)
                m_ij = self.adaptive_gamma * torch.sigmoid(abs_diff)
                margin_violation = m_ij - (dist_high - dist_low)

                # scalar stats
                amrl_mean_mij = float(m_ij.mean().detach().cpu().item())
                amrl_mean_abs_diff = float(abs_diff.mean().detach().cpu().item())
                amrl_mean_margin_violation = float(torch.clamp(margin_violation, min=0.0).mean().detach().cpu().item())
                amrl_mean_dist_high = float(dist_high.mean().detach().cpu().item())
                amrl_mean_dist_low = float(dist_low.mean().detach().cpu().item())
                num_pairs = int(dist_high.view(-1).shape[0])

                # compute loss (use the module)
                ### End
                loss = self.amrl(dist_high, dist_low, q_high_vec, q_low_vec)
                tmp_loss = self.amrl(dist_high, dist_low, q_high_vec, q_low_vec)
                loss_amrl_val = float(tmp_loss.detach().cpu().item())
                print(f"[loss] amrl_loss = {loss_amrl_val}")

                # debug print(s)
                print(f"[AMRL] step={self.global_step} iter={iteration} num_pairs={num_pairs} mean_mij={amrl_mean_mij:.6f} mean_abs_diff={amrl_mean_abs_diff:.6f} mean_margin_violation={amrl_mean_margin_violation:.6f} mean_dist_high={amrl_mean_dist_high:.6f} mean_dist_low={amrl_mean_dist_low:.6f}")
                
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                with open(args.logs_csv_path, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, iteration, "adaptive_margin_rank", loss_amrl_val])
                    f.flush()
            # Adaptive Margin Rank Loss - End

            if config.contrastive or config.contrique:
                f_neg_feat = self.ssh(f_low)
                f_pos_feat = self.ssh(f_high)

                loss_fn = ContrastiveLoss(f_pos_feat.shape[0], 0.1).cuda()

                loss = loss_fn(f_neg_feat, f_pos_feat)

            if config.group_contrastive:

                idx = np.argsort(pred0.cpu(), axis=0)

                f_feat = self.ssh(inputs.cuda())

                f_pos_feat = []
                f_neg_feat = []

                for n in range(max(2,int(config.batch_size*config.p))):
                    try:
                        f_pos_feat.append(f_feat[idx[n]])
                        f_neg_feat.append(f_feat[idx[-n - 1]])
                    except:
                        continue

                f_pos_feat = torch.squeeze(torch.stack(f_pos_feat), dim=1)
                f_neg_feat = torch.squeeze(torch.stack(f_neg_feat), dim=1)

                loss_fn = GroupContrastiveLoss(f_pos_feat.shape[0], 0.1).cuda()
                tmp_loss = loss_fn(f_neg_feat, f_pos_feat)

                if config.rank or config.blur or config.comp or config.nos:
                    loss +=  tmp_loss * config.weight
                else:
                    loss = tmp_loss

                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                with open(args.logs_csv_path, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, iteration, "group_contrastive", float(tmp_loss.detach().cpu().item())])
                    writer.writerow([timestamp, iteration, "group_contrastive_additive", loss])
                    f.flush()

            if config.rotation:
                inputs_ssh, labels1_ssh = rotate_batch(inputs.cuda(), 'rand')
                outputs_ssh = self.ssh(inputs_ssh.float())
                loss = nn.CrossEntropyLoss()(outputs_ssh, labels1_ssh.cuda())

            loss.backward()
            self.optimizer_ssh.step()
            loss_hist.append(loss.detach().cpu())

        # print(loss_hist)
        self.global_step += 1

        return loss_hist

    def new_ttt(self, data, config):

        if config.online:
            self.net.load_state_dict(torch.load(self.config.svpath ))

        old_net = copy.deepcopy(self.net)
        old_net.load_state_dict(torch.load(self.config.svpath))

        steps = 0

        pred_scores = []
        pred_scores_old = []

        gt_scores = []
        mse_all = []
        mse_all_old = []


        for data_dict, label in tqdm(data, leave=False):
            img = data_dict['image']

            if not config.online:
                self.net.load_state_dict(torch.load(self.config.svpath ))

            label = torch.as_tensor(label.to(self.device)).requires_grad_(False)

            old_net.load_state_dict(torch.load(self.config.svpath ))

            if config.group_contrastive:
                if len(img) > 3:
                    loss_hist = self.adapt(data_dict, config, old_net)
                else:
                    if config.rank or config.blur or config.comp or config.nos or config.contrastive or config.rotation or config.contrique or config.adaptive_margin_rank:
                        config.group_contrastive = False
                        loss_hist = self.adapt(data_dict, config, old_net)
            elif config.rank or config.blur or config.comp or config.nos or config.contrastive or config.rotation or config.contrique:
                loss_hist = self.adapt(data_dict, config, old_net)
            # Adaptive Margin 
            elif config.adaptive_margin_rank:
                loss_hist = self.adapt(data_dict, config, old_net)

            # if config.rank:
            #     print('done')

            mse, pred = self.test_single_iqa(self.net, img, label)

            old_net.load_state_dict(torch.load(self.config.svpath ))

            mse_old, pred_old = self.test_single_iqa(old_net, img, label)

            pred_scores = pred_scores + pred.cpu().tolist()
            pred_scores_old = pred_scores_old + pred_old.cpu().tolist()
            gt_scores = gt_scores + label.cpu().tolist()

            mse_all.append(mse.cpu())
            mse_all_old.append(mse_old.cpu())

            steps += 1

            if steps % 10 == 0:
                pred_scores1 = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
                pred_scores_old1 = np.mean(np.reshape(np.array(pred_scores_old), (-1, self.test_patch_num)), axis=1)
                gt_scores1 = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)

                test_srcc_old, _ = stats.spearmanr(pred_scores_old1, gt_scores1)
                test_plcc_old, _ = stats.pearsonr(pred_scores_old1, gt_scores1)

                test_srcc, _ = stats.spearmanr(pred_scores1, gt_scores1)
                test_plcc, _ = stats.pearsonr(pred_scores1, gt_scores1)
                
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                with open(args.plcc_csv_path, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, steps, test_srcc_old, test_srcc, test_plcc_old, test_plcc])
                    f.flush()

                print(
                    'After {} images test_srcc old : {}  new {} \n test_plcc old:{} new:{}'.format(steps, test_srcc_old,
                                                                                                   test_srcc,
                                                                                                   test_plcc_old,
                                                                                                   test_plcc))

        pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
        pred_scores_old = np.mean(np.reshape(np.array(pred_scores_old), (-1, self.test_patch_num)), axis=1)
        gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)

        test_srcc_old, _ = stats.spearmanr(pred_scores_old, gt_scores)
        test_plcc_old, _ = stats.pearsonr(pred_scores_old, gt_scores)

        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)

        return test_srcc_old, test_plcc_old, test_srcc, test_plcc

    def test_single_iqa(self, model, image, label):
        model.eval()
        with torch.no_grad():
            pred, _ = model(image.cuda())
            mse_loss = self.l1_loss(label, pred)
        return mse_loss, pred

parser = argparse.ArgumentParser()
parser.add_argument('--img_num',type=int, default='1000')
parser.add_argument('--datapath', default='..')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--ng', default=2, type=int)
parser.add_argument('--fix_ssh', action='store_true')
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--niter', default=3, type=int)
parser.add_argument('--online', action='store_true')
parser.add_argument('--weight', type=float, default=1,help='Weight for rank plus GC')
parser.add_argument('--p', type=float, default=0.25,help='p for GC loss')
parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default=1,
                    help='Number of sample patches from testing image')
parser.add_argument('--seed', dest='seed', type=int, default=2021,
                        help='for reproducing the results')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=224,
                    help='Crop size for training & testing image patches')
parser.add_argument('--svpath', dest='svpath', type=str,default='/home/user/Subhadeep/ttt_cifar_IQA/Save_RES/',
                        help='the path to save the info')
parser.add_argument('--gpunum', dest='gpunum', type=str, default='0',
                    help='the id for the gpu that will be used')
parser.add_argument('--contrastive', action='store_true')
parser.add_argument('--group_contrastive', action='store_true')
parser.add_argument('--rank', action='store_true')
parser.add_argument('--comp', action='store_true')
parser.add_argument('--contrique', action='store_true')
parser.add_argument('--nos', action='store_true')
parser.add_argument('--blur', action='store_true')
parser.add_argument('--sr', action='store_true')
parser.add_argument('--rotation', action='store_true')
parser.add_argument('--test_only', action='store_true')
parser.add_argument('--online_mse', action='store_true')
parser.add_argument('--tta_tf', action='store_true')
parser.add_argument('--bn', action='store_true')
parser.add_argument('--ln', action='store_true')
parser.add_argument('--run', dest='run', type=int, default=1,
                        help='for running at multiple seeds')

# Adaptive Margin Rank - Start
                        # new flags for Adaptive Margin Rank Loss
parser.add_argument('--adaptive_margin_rank', dest='adaptive_margin_rank', action='store_true',
                    help='Enable Adaptive Margin Ranking Loss (AMRL)')
parser.add_argument('--adaptive_gamma', dest='adaptive_gamma', type=float, default=0.5,
                    help='Gamma (max margin) used by AMRL; only used when --adaptive_margin_rank is set (default=0.5)')
# Adaptive Margin Rank - End

parser.add_argument("--exp-name",  dest='exp_name', help="experiment name", default="exp")
parser.add_argument("--exp-path",  dest='exp_path', help="experiment path")
parser.add_argument("--logs-csv-path",  dest='logs_csv_path', help="logs csv path")
parser.add_argument("--plcc-csv-path",  dest='plcc_csv_path', help="plcc csv path")

args = parser.parse_args()

if args.exp_name == "exp":
    args.exp_name = args.exp_name + "_" + str(datetime.now()).replace(" ","_").replace(":","_").split(".")[0]

args.exp_path = "/content/results/" + args.exp_name 
os.makedirs(args.exp_path, exist_ok=True)
csv_path = os.path.join(args.exp_path, "args_values.csv")

# convert argparse Namespace to dict and save as a single-row CSV
df = pd.DataFrame([vars(args)])
df.to_csv(csv_path, index=False)
# args.datapath='/media/user/New Volume/Subhadeep/datasets/'+args.datapath

args.logs_csv_path = os.path.join(args.exp_path,"training_loss_log.csv")
with open(args.logs_csv_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "iteration", "loss_type", "loss_value"])

args.plcc_csv_path = os.path.join(args.exp_path,"plcc_srcc_scores.csv")
with open(args.plcc_csv_path, mode="w", newline="") as f:
    b_writer = csv.writer(f)
    b_writer.writerow(["timestamp", "steps", "srcc", "srcc_new", "plcc", "plcc_new"])


if torch.cuda.is_available():
    if len(args.gpunum) == 1:
        device = torch.device("cuda", index=int(args.gpunum))
else:
    device = torch.device("cpu")

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

#----------------------------------------
folder_path = args.datapath
sel_num = list(range(0, args.img_num))


rho_s_list, rho_p_list=[],[]

for mul in range(args.run):

    # fix the seed if needed for reproducibility
    if args.seed == 0:
        pass
    else:
        if mul!=0:
            args.seed=args.seed+np.random.randint(1000)
        print('we are using the seed = {}'.format(args.seed))
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    # Randomly select 80% images for training and the rest for testing
    random.shuffle(sel_num)
    test_index = sel_num

    test_loader = DataLoader(args, folder_path,
                                         test_index, args.patch_size,
                                         args.test_patch_num,batch_size=args.batch_size)
    test_data = test_loader.get_data()

    solver = Model(args, device,Net)

    if args.test_only:
        srcc_computed, plcc_computed,_,_= solver.test(test_data,pretrained=1)
        print('srcc_computed_test {}, plcc_computed_test {}'.format(srcc_computed, plcc_computed))
    else:
        test_srcc_old,test_plcc_old,srcc_computed, plcc_computed =solver.new_ttt(test_data,args)
        print('srcc_computed {}, plcc_computed {}'.format(srcc_computed, plcc_computed))

    rho_s_list.append(srcc_computed)
    rho_p_list.append(plcc_computed)

final_rho_s=np.mean(np.array(rho_s_list))
final_rho_p=np.mean(np.array(rho_p_list))

if not args.test_only:
    print('final_srcc old {}, final_plcc old {}'.format(test_srcc_old, test_plcc_old))
print(' final_srcc new {}, final_plcc new:{}'.format(final_rho_s,final_rho_p))
