import argparse
import copy
import json
import os
from types import SimpleNamespace

import optuna
import wandb

from train_mtcl import OneFoldTrainer
from utils import set_random_seed


def objective(trial, base_config, args):
    config = copy.deepcopy(base_config)
    hpo_config = config['hpo']
    search = hpo_config['search']

    config['feature_pyramid']['dim'] = trial.suggest_categorical(
        'feature_dim', search['feature_dim'])
    config['classifier']['model_dim'] = trial.suggest_categorical(
        'model_dim', search['model_dim'])
    config['classifier']['feedforward_dim'] = trial.suggest_categorical(
        'feedforward_dim', search['feedforward_dim'])
    config['classifier']['pool'] = trial.suggest_categorical(
        'pool', search['pool'])
    #config['classifier']['dropout'] = trial.suggest_categorical(
    #    'dropout', search['dropout'])
    #config['backbone']['dropout'] = config['classifier']['dropout']
    config['training_params']['lr'] = trial.suggest_float(
        'lr', search['lr'][0], search['lr'][1], log=True)
    config['training_params']['weight_decay'] = trial.suggest_float(
        'weight_decay', search['weight_decay'][0], search['weight_decay'][1], log=True)
    config['training_params']['batch_size'] = trial.suggest_categorical(
        'batch_size', search['batch_size'])
    config['training_params']['max_epochs'] = hpo_config['max_epochs']
    config['training_params']['val_period'] = hpo_config['val_period']
    config['training_params']['early_stopping']['patience'] = 10
    config['name'] = '{}_hpo_trial_{:04d}'.format(
        base_config['name'], trial.number)

    set_random_seed(hpo_config['seed'] + trial.number, use_cuda=True)
    run = wandb.init(
        entity='jschramm-technical-university-of-berlin',
        project='SleePyCo',
        name=config['name'],
        group='optuna-{}'.format(base_config['name']),
        job_type='hpo',
        config={
            'optuna_trial': trial.number,
            'optuna_params': trial.params,
            'training_config': config,
        },
    )
    try:
        trainer = OneFoldTrainer(args, hpo_config['fold'], config)
        score = trainer.run(evaluate_test=False)
        wandb.log({'best_val_acc': score}, step=trainer.train_iter)
        trial.set_user_attr('config', config)
        return score
    finally:
        run.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config) as config_file:
        config = json.load(config_file)
    config['name'] = os.path.basename(args.config).replace('.json', '')

    hpo_config = config['hpo']
    trainer_args = SimpleNamespace(gpu=args.gpu)
    study_name = '{}_hpo'.format(config['name'])
    # Use a per-config sqlite file to avoid conflicts when search spaces change
    storage_path = f"sqlite:///{config['name']}_optuna.db"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_path,
        load_if_exists=True,
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=hpo_config['seed']))
    study.optimize(
        lambda trial: objective(trial, config, trainer_args),
        n_trials=hpo_config['n_trials'])

    best = study.best_trial
    output_path = '{}_best.json'.format(config['name'])
    with open(output_path, 'w') as output_file:
        json.dump(best.user_attrs['config'], output_file, indent=4)

    print('[INFO] Best validation accuracy: {:.3f}'.format(best.value))
    print('[INFO] Best parameters: {}'.format(best.params))
    print('[INFO] Best config written to {}'.format(output_path))


if __name__ == '__main__':
    main()