import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            model = model.Model(args, checkpoint)
            loss_l = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, model, loss_l, checkpoint)
            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()

if __name__ == '__main__':
    main()
