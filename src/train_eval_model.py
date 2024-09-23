
import click


from ruamel.yaml import YAML

import sys
sys.path.append('.')

from src.funcs import set_seed
from src.spec_funcs import train_eval_classic

@click.command()
@click.argument('config_path')
def main(config_path):
    
    conf = YAML().load(open(config_path))

    set_seed(conf['seed'])

    train_eval_classic(conf, target_col='labels', fig_size1=(15,8), fig_size2=(20,12), thresh_space_l=[])


if __name__=='__main__':

    main()