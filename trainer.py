import torch
import argparse
import os, time
import copy
from pathlib import Path
import numpy as np
from sklearn import metrics
from utils import trainUtils

parser = argparse.ArgumentParser(description="optfeature trainer")
parser.add_argument("dataset", type=str, help="specify dataset")
parser.add_argument("model", type=str, help="specify model")

# dataset information
parser.add_argument("--feature", type=int, help="feature number", required=True)
parser.add_argument("--field", type=int, help="field number", required=True)
parser.add_argument("--data_dir", type=str, help="data directory", required=True)

# training hyperparameters
parser.add_argument("--lr", type=float, help="learning rate" , default=1e-4)
parser.add_argument("--l2", type=float, help="L2 regularization", default=1e-5)
parser.add_argument("--bsize", type=int, help="batchsize", default=4096)
parser.add_argument("--optim", type=str, default="Adam", help="optimizer type")
parser.add_argument("--max_epoch", type=int, default=20, help="maxmium epochs")
parser.add_argument("--save_dir", type=Path, help="model save directory")
parser.add_argument("--store_dict", action="store_true", help="store model dictionary or not")

# neural network hyperparameters
parser.add_argument("--dim", type=int, help="embedding dimension", default=16)
parser.add_argument("--mlp_dims", type=int, nargs='+', default=[1024, 512, 256], help="mlp layer size")
parser.add_argument("--mlp_dropout", type=float, default=0.0, help="mlp dropout rate (default:0.0)")
parser.add_argument("--mlp_bn", action="store_true", help="mlp batch normalization")

# device information
parser.add_argument("--cuda", type=int, choices=range(-1, 8), default=-1, help="device info")
args = parser.parse_args()

my_seed = 2023
torch.manual_seed(my_seed)
torch.cuda.manual_seed_all(my_seed)
np.random.seed(my_seed)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
os.environ['NUMEXPR_MAX_THREADS'] = '8'

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.lr = args.lr
        self.l2 = args.l2
        self.bs = args.bsize
        self.model_dir = args.save_dir
        self.store_dict = args.store_dict

        self.dataloader = trainUtils.getDataLoader(args.dataset, args.data_dir)
        self.device = trainUtils.getDevice(args.cuda)
        self.network = trainUtils.getModel(args.model, args, self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optim = trainUtils.getOptimOne(self.network, args.optim, self.lr, self.l2)
    
    def train_on_batch(self, label, data):
        self.network.train()
        self.optim.zero_grad()
        data, label = data.to(self.device), label.to(self.device)
        logit = self.network(data)
        logloss = self.criterion(logit, label)
        loss = logloss
        loss.backward()
        self.optim.step()
        return logloss.item()

    def eval_on_batch(self, data):
        self.network.eval()
        with torch.no_grad():
            data = data.to(self.device)
            logit = self.network(data)
            prob = torch.sigmoid(logit).detach().cpu().numpy()
        return prob
    
    def train(self, epochs):
        step = 0
        cur_auc = 0.0
        for epoch_idx in range(int(epochs)):
            train_loss = .0
            step = 0
            for feature, label in self.dataloader.get_data("train", batch_size = self.bs):
                train_loss += self.train_on_batch(label, feature)
                step += 1
            train_loss /= step
            val_auc, val_loss = self.evaluate("val")
            print("[Epoch {epoch:d} | Train Loss:{loss:.6f} | Val AUC:{val_auc:.6f}, Val Loss:{val_loss:.6f}]".
                        format(epoch=epoch_idx,  loss=train_loss, val_auc=val_auc, val_loss=val_loss ))
            early_stop = False
            if val_auc > cur_auc:
                cur_auc = val_auc
                if self.store_dict:
                    torch.save(self.network.state_dict(), self.model_dir)
                else:
                    self.model_best_dict = copy.deepcopy(self.network.state_dict())
            else:
                if self.store_dict:
                    self.network.load_state_dict(torch.load(self.model_dir))
                else:
                    self.network.load_state_dict(self.model_best_dict)
                self.network.to(self.device)
                early_stop = True
                te_auc, te_loss = self.evaluate("test")
                print("Early stop at epoch {epoch:d}|Test AUC: {te_auc:.6f}, Test Loss:{te_loss:.6f}".
                        format(epoch=epoch_idx, te_auc = te_auc, te_loss = te_loss))
                break
        if not early_stop:
            te_auc, te_loss = self.evaluate("test")
            print("Final Test AUC:{te_auc:.6f}, Test Loss:{te_loss:.6f}".format(te_auc=te_auc, te_loss=te_loss))

    def evaluate(self, on:str):
        preds, trues = [], []
        for feature, label in self.dataloader.get_data(on, batch_size = self.bs):
            pred = self.eval_on_batch(feature)
            label = label.detach().cpu().numpy()
            preds.append(pred)
            trues.append(label)
        y_pred = np.concatenate(preds).astype("float64")
        y_true = np.concatenate(trues).astype("float64")
        auc = metrics.roc_auc_score(y_true, y_pred)
        loss = metrics.log_loss(y_true, y_pred)
        return auc, loss

def main():
    print(args)
    trainer = Trainer(args)
    trainer.train(args.max_epoch)

if __name__ == "__main__":
    """
    python trainer.py Criteo DeepFM --feature    
    """
    main()
