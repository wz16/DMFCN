import os
import torch
import sys
import importlib
# from models import Autoformer, LMFCN_3layer, LMFCN_fourier_t1_singleblock, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
#     Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, LMFCN_1, TimesNet_autocorre,TimesNet_cross_corre, \
#     FCN_dilation7,FCN_no_dilation, OS_CNN, LMFCN_fourier_t1_singleblock,LMFCN_fourier_t2_singleblock_noembedding, \
#     LMFCN_3layer,LMFCN_1layer

directory_path = './models'
sys.path.append(directory_path)

file_names = os.listdir(directory_path)

# Filter the files to get only Python modules (files ending with .py)
module_names = [file_name[:-3] for file_name in file_names if file_name.endswith('.py')]


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {}
        for ind, module_name in enumerate(module_names):
            module = importlib.import_module(module_name)
            self.model_dict[module_names[ind]] = module

        # self.model_dict = {
        #     'TimesNet': TimesNet,
        #     'Autoformer': Autoformer,
        #     'Transformer': Transformer,
        #     'Nonstationary_Transformer': Nonstationary_Transformer,
        #     'DLinear': DLinear,
        #     'FEDformer': FEDformer,
        #     'Informer': Informer,
        #     'LightTS': LightTS,
        #     'Reformer': Reformer,
        #     'ETSformer': ETSformer,
        #     'PatchTST': PatchTST,
        #     'Pyraformer': Pyraformer,
        #     'MICN': MICN,
        #     'Crossformer': Crossformer,
        #     'FiLM': FiLM,
        #     'LMFCN_1':LMFCN_1,
        #     "FCN_dilation7":FCN_dilation7,
        #     "FCN_no_dilation":FCN_no_dilation,
        #     'LMFCN_fourier': LMFCN_3layer,
        #     "LMFCN_fourier_t1_singleblock":LMFCN_fourier_t1_singleblock,
        #     "LMFCN_fourier_t2_singleblock_noembedding":LMFCN_fourier_t2_singleblock_noembedding,
        #     'OS_CNN': OS_CNN,
        #     'TimesNet_autocorre': TimesNet_autocorre,
        #     'TimesNet_cross_corre': TimesNet_cross_corre,
        #     'LMFCN_3layer':LMFCN_3layer,
        #     'LMFCN_1layer':LMFCN_1layer
        # }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("model size:{}".format(count_parameters(self.model)))

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
