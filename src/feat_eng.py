
import click


from ruamel.yaml import YAML

import sys
sys.path.append('.')
from src.spec_funcs import calc_feat_eng_matr
from src.funcs import set_seed

@click.command()
@click.argument('config_path')
def main(config_path):
    
    conf = YAML().load(open(config_path))
    set_seed(conf['seed'])
    calc_feat_eng_matr(conf, target_col='labels')


if __name__=='__main__':

    main()