
import click
from ruamel.yaml import YAML

import sys
sys.path.append('.')

from src.funcs import set_seed
from src.spec_funcs import train_eval_classic

@click.command()
@click.argument('config_path')
def main(config_path):
    
    conf = YAML().load(open('params.yaml'))
    conf_ttp = YAML().load(open(config_path))
    set_seed(conf['seed'])

    # так как в функцию передается conf с одноименными ключами
    # а такие ключи в разных params.yaml файлах dvc не дает задать приходится обходить    
    conf_ttp['seed'] = conf['seed']
    conf_ttp['prep_text'] = conf['prep_text']
    conf_ttp['train_eval_model'] = conf_ttp['train_eval_model_ttp']
    conf_ttp['feat_gen'] = conf_ttp['feat_gen_ttp']
    conf_ttp['feat_eng'] = conf_ttp['feat_eng_ttp']

    set_seed(conf['seed'])

    train_eval_classic(conf_ttp, target_col='ttp', fig_size1=(60,40), fig_size2=(20,12))


if __name__=='__main__':

    main()