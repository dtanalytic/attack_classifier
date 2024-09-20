import click

from ruamel.yaml import YAML

import sys
sys.path.append('.')
from src.spec_funcs import calc_feat_eng_matr
from src.funcs import set_seed

@click.command()
@click.argument('config_path')
def main(config_path):
    
    conf = YAML().load(open('params.yaml'))
    set_seed(conf['seed'])

    conf_ttp = YAML().load(open(config_path))
        
    conf_ttp['seed'] = conf['seed']
    conf_ttp['prep_text'] = conf['prep_text']
    
    conf_ttp['feat_gen'] = conf_ttp['feat_gen_ttp']
    conf_ttp['feat_eng'] = conf_ttp['feat_eng_ttp']

    calc_feat_eng_matr(conf_ttp, target_col='ttp')


if __name__=='__main__':

    main()