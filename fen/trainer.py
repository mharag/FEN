import torch
from torch import nn
import time
from progress.bar import Bar


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
          self.avg = self.sum / self.count


class Trainer:
    def __init__(
        self,
        model,
        training_id='default',
        save_dir='./exp',
        lr=3e-5,
        use_cuda=True,
    ):
        super(Trainer, self).__init__()
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        self.lr = lr

        self.reg_loss = nn.MSELoss().to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.model = model.to(self.device)

    def run_sample(self, sample):
        g1, g2, sim = sample
        g1_embd = self.model(g1.to(self.device))
        g2_embd = self.model(g2.to(self.device))
        g1_output = g1_embd[g1.outputs]
        g2_output = g2_embd[g2.outputs]
        emb_dis = (torch.cosine_similarity(g1_output, g2_output, eps=1e-8) + 1) / 2
        func_loss = self.reg_loss(emb_dis, torch.tensor([sim], device=self.device)) * 10

        return func_loss

    def train(self, num_epoch, train_dataset, val_dataset):
        # AverageMeter
        batch_time = AverageMeter()
        loss_meter = AverageMeter()

        # Train
        print('[INFO] Start training, lr = {:.4f}'.format(self.optimizer.param_groups[0]['lr']))
        for epoch in range(1, num_epoch+1):
            self.setup_params(epoch)

            for phase in ['train']:
                if phase == 'train':
                    dataset = train_dataset
                    self.model.train()
                else:
                    dataset = val_dataset
                    self.model.eval()
                    torch.cuda.empty_cache()

                bar = Bar(f"{phase} {epoch}/{num_epoch}", max=len(dataset))
                for iter_id, sample in enumerate(dataset):
                    time_stamp = time.time()
                    loss = self.run_sample(sample)
                    if phase == 'train':
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    batch_time.update(time.time() - time_stamp)
                    loss_meter.update(loss.item())
                    Bar.suffix = f"[{iter_id}/{len(dataset)}]|Tot: {bar.elapsed_td:} |ETA: {bar.eta_td:} | Loss: {loss_meter.avg:.4f} | Net: {batch_time.avg:.2f}s"
                    bar.next()
                bar.finish()

    def save(self, path):
        data = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        torch.save(data, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        for param_group in self.optimizer.param_groups:
            self.lr = param_group["lr"]
        self.model.load(path)

    def setup_params(self, epoch):
        pass
        #self.model_epoch += 1
        #if self.lr_step > 0 and self.model_epoch % self.lr_step == 0:
        #    self.lr *= 0.1
        #    if self.local_rank == 0:
        #        print('[INFO] Learning rate decay to {}'.format(self.lr))
        #    for param_group in self.optimizer.param_groups:
        #        param_group['lr'] = self.lr
