import os
import math
import torch.nn as nn
from decimal import Decimal
import time
from collections import OrderedDict
import utility
import torch.nn.functional as F
import cv2 as cv
import torch
import torch.nn.utils as utils
from tqdm import tqdm

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        self.optimizer.schedule()
        self.loss.step()
        #epoch = self.optimizer.get_last_epoch() + 1
        epoch = self.optimizer.get_last_epoch()
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
                
           
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, hr, _, idx_scale) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            grad = Gradient()
            gradhr = grad(hr)
            timer_data.hold()
            timer_model.tic()
             
            self.optimizer.zero_grad()
            sr = self.model(lr, idx_scale)
            gradsr = grad(sr)
            loss= self.loss(sr, hr)
            #print(0.01*loss1)
            
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gcl
                )
            self.optimizer.step()
            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        torch.set_grad_enabled(False)

        #epoch = self.optimizer.get_last_epoch() + 1
        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()
    
        start= torch.cuda.Event(enable_timing=True)
        end=torch.cuda.Event(enable_timing=True)
        test_results=OrderedDict()
        test_results['runtime']=[]

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename, _ in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)
                    #torch.cuda.synchronize()
                    start.record()
                    #print(lr.size())
                    sr = self.model(lr, idx_scale)
                    #torch.cuda.synchronize()
                    end.record()
                    torch.cuda.synchronize()
                    test_results['runtime'].append(start.elapsed_time(end))

                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')
        ave_runtime=sum(test_results['runtime'])/len(test_results['runtime'])/1000.0
        self.ckp.write_log('Averagetime is :{:.6f} seconds'.format(ave_runtime))
        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            #epoch = self.optimizer.get_last_epoch() + 1
            epoch = self.optimizer.get_last_epoch()
            return epoch >= self.args.epochs
class Gradient(nn.Module):
    def __init__(self):
        super(Gradient,self).__init__()
        kernel_x = [
            [-1,0,1],
            [-1,0,1],
            [-1,0,1]
        ]
        kernel_y = [
            [-1,-1,-1],
            [0,0,0],
            [1,1,1]
        ]
        kernel_lap = [
            [0,-1,0],
            [-1,4,-1],
            [0,-1,0]
        ]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0)
        self.weightx =nn.Parameter(data = kernel_x,requires_grad=True).cuda()
        self.weighty = nn.Parameter(data=kernel_y,requires_grad = True).cuda()
    def forward(self,x):
        x0 = x[:,0]
        #print(x.size())
        x1 = x[:,1]
        x2 = x[:,2]
        gradx0 = F.conv2d(x0.unsqueeze(1),self.weightx,padding=1)
        grady0 = F.conv2d(x0.unsqueeze(1),self.weighty,padding=1)
        gradx1 = F.conv2d(x1.unsqueeze(1),self.weightx,padding=1)
        grady1 = F.conv2d(x1.unsqueeze(1),self.weighty,padding=1)
        gradx2 = F.conv2d(x2.unsqueeze(1),self.weightx,padding=1)
        grady2 = F.conv2d(x2.unsqueeze(1),self.weighty,padding=1)
        gradient0 = torch.sqrt(torch.pow(gradx0,2)+torch.pow(grady0,2))
        gradient1 = torch.sqrt(torch.pow(gradx1,2)+torch.pow(grady1,2))
        gradient2 = torch.sqrt(torch.pow(gradx2,2)+torch.pow(grady2,2))
        x = torch.cat([gradient0,gradient1,gradient2],dim=1)
        return x 
def gradient(x):
    h_x = x.size()[2]
    w_x = x.size()[3]
    r = F.pad(x,[0,1,0,0])[:,:,:,1:]
    l = F.pad(x,[1,0,0,0])[:,:,:,:h_x]
    t = F.pad(x,[0,0,0,1])[:,:,:w_x,:]
    b = F.pad(x,[0,0,0,1])[:,:,1:,:]
    xgrad = torch.pow(torch.pow((r-l)*0.5,2)+torch.pow((b-t)*0.5,2),0.5)
    return xgrad

