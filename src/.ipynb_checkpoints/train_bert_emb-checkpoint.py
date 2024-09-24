from ruamel.yaml import YAML
import click

import sys
sys.path.append('.')
from src.funcs import set_seed

conf = YAML().load(open('params.yaml'))
set_seed(conf['seed'])
# до TextModelClass, где нейронка инициализируется
from src.spec_nn_funcs import TextDFDataset, TextModelClass, train_eval_bert


@click.command()
@click.argument('config_path_dop')
def main(config_path_dop):

    conf_dop = YAML().load(open(config_path_dop))

    train_eval_bert(conf, conf_dop, target_col='labels', fig_size1=(15,8), fig_size2=(20,12), thresh_space_l=[])


if __name__=='__main__':

    main()