from ruamel.yaml import YAML
import click
import numpy as np

import sys
sys.path.append('.')
from src.funcs import set_seed

conf = YAML().load(open('params.yaml'))
set_seed(conf['seed'])
# до TextModelClass, где нейронка инициализируется
# from src.spec_nn_funcs import TextDFDataset, TextModelClass, train_eval_bert
from src.spec_nn_funcs import train_eval_bert

@click.command()
@click.argument('config_path_dop')
def main(config_path_dop):

    conf_dop = YAML().load(open(config_path_dop))
    conf_alt = YAML().load(open('dvc_pipes/ttp/params_ttp.yaml'))
    conf['feat_gen'] = conf_alt['feat_gen_ttp']
    conf['train_eval_model'] = conf_alt['train_eval_model_ttp']
    

    conf_dop['nn'] = conf_dop['nn_ttp']
    conf_dop['nn_bert'] = conf_dop['nn_bert_ttp']
    # np.arange(0.005, 1, 0.005))
    train_eval_bert(conf, conf_dop, target_col='ttp', fig_size1=(60,40), fig_size2=(20,12), thresh_space_l=np.arange(0.001, 1, 0.002))


if __name__=='__main__':

    main()