import pandas as pd
from SAFS.Models import SAFSModel
import re
import glob
import yaml


class ExpRunner():
    def __init__(self, loader_method, config_path):
        with open(config_path, 'r') as ymlfile:
            config = yaml.load(ymlfile)
        self.config = config
        self.name = config['name']
        self.save_path = config['save_path']
        self.n_fs = config['n_features']
        self.kernel = config['kernel']
        self.stride = config['stride']
        self.d_hidden = config['d_hidden']
        self.d_classifier = config['d_classifier']
        self.n_cls = config['n_cls']
        self.n_heads = config['n_heads']
        self.random_seeds = config['random_seeds']
        self.epochs = config['epochs']
        self.lr = config['lr']
        self.device = config['device']

        self.dataloader = loader_method(config['data_path'], batch_size=config['batch_size'],
                                        eval_size=config['eval_size'], shuffle=config['shuffle'])

    def train(self, no_log):
        for i in range(15, 296, 10):
            d_out = [i]
            n_sub = [int(i / 5)]
            name_ = self.name + '_out_dim_%d' % d_out[-1]
            model = SAFSModel(name_,
                              self.save_path + '/%s/models/' % name_,
                              self.save_path + '/%s/logs/' % name_,
                              d_features=self.n_fs, d_out_list=d_out, n_subset_list=n_sub, kernel=self.kernel,
                              stride=self.stride, d_k=self.d_hidden, d_v=self.d_hidden, d_classifier=self.d_classifier,
                              n_classes=self.n_cls, n_heads=self.n_heads, random_seeds=self.random_seeds, no_log=no_log)

            model.train(self.epochs, self.lr, self.dataloader.train_dataloader(), self.dataloader.eval_dataloader(),
                        self.device)

    def test(self, start_dim, end_dim, interval):
        results = pd.DataFrame(columns=['dataset', 'out_dim', 'sub_dim', 'accuracy', 'loss'])
        for i in range(start_dim, end_dim, interval):
            d_out = [i]
            n_sub = [int(i / 5)]
            name_ = self.name + '_out_dim_%d' % d_out[-1]
            model = SAFSModel(None, self.save_path, None, d_features=self.n_fs, d_out_list=d_out, n_subset_list=n_sub,
                              kernel=self.kernel, stride=self.stride, d_k=self.d_hidden, d_v=self.d_hidden,
                              d_classifier=self.d_classifier, n_classes=self.n_cls, n_heads=self.n_heads,
                              random_seeds=self.random_seeds, test=True)

            # Get minimum loss model
            models = glob.glob(model.model_path + '/%s/models/*loss-*' % (name_))
            minimum = min(map(lambda x: float(re.findall(".*loss-(.*)", x)[0]), models))
            model_path = glob.glob(model.model_path + '/%s/models/*loss-%f*' % (name_, minimum))
            model.load_model(model_path)
            pred, real, acc, loss, attn = model.predict_dataset(self.dataloader.test_dataloader(), 'cuda')
            # attn_map = end2end_attention(random_shuffle(d_features, n_heads, seeds=random_seeds), attn, d_features, n_heads, kernel, stride)
            result = pd.DataFrame([[name_, d_out, n_sub, acc, loss]],
                                  columns=['dataset', 'out_dim', 'sub_dim', 'accuracy', 'loss'])
            results = results.append(result, ignore_index=True)