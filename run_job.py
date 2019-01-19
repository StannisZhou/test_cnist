import os
# from db import db
import argparse
from utils import logger
from utils import py_utils
from config import Config
from ops import model_tools


def get_fine_tune_params(out_dir, reduction=False):
    """Parameters for fine-tuning, to e.g. test transfer/forgetting."""
    if out_dir is not None and reduction:
        return {
            'reduction': reduction,
            'out_dir': out_dir,
            'max_steps': 20000,
            'test_iters': 1000
        }
    elif out_dir is not None and not reduction:
        return {
            'out_dir': out_dir,
            'max_steps': 1000,
            'test_iters': 50
        }
    else:
        raise NotImplementedError


def main(
        experiment,
        model,
        train,
        val,
        checkpoint,
        use_db=True,
        reduction=0,
        random=True,
        gpu_device='/gpu:0',
        cpu_device='/cpu:0',
        transfer=False,
        out_dir=None):
    """Interpret and run a model."""
    main_config = Config()
    dt_string = py_utils.get_dt_stamp()
    log = logger.get(
        os.path.join(main_config.log_dir, '%s_%s' % (experiment, dt_string)))
    if use_db:
        exp_params = db.get_parameters(
            log=log,
            experiment=experiment,
            random=random)[0]
    else:
        exp = py_utils.import_module(experiment, pre_path='experiments')
        exp_params = exp.experiment_params()
        exp_params['experiment'] = experiment
        exp_params['train_dataset'] = train
        exp_params['val_dataset'] = val
        exp_params['model'] = model
    if reduction or out_dir is not None or transfer:
        raise NotImplementedError
        # fine_tune = get_fine_tune_params(
        #     out_dir=out_dir, reduction=reduction)
    else:
        pass
        # fine_tune = False
    model_tools.build_model(
        exp_params=exp_params,
        dt_string=dt_string,
        log=log,
        config=main_config,
        gpu_device=gpu_device,
        cpu_device=cpu_device,
        checkpoint=checkpoint)
    log.info('Finished.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment',
        dest='experiment',
        type=str,
        default=None,
        help='Model script.')
    parser.add_argument(
        '--model',
        dest='model',
        type=str,
        default=None,
        help='Model script.')
    parser.add_argument(
        '--train',
        dest='train',
        type=str,
        default=None,
        help='Train data.')
    parser.add_argument(
        '--val',
        dest='val',
        type=str,
        default=None,
        help='Validation dataset.')
    parser.add_argument(
        '--ckpt',
        dest='checkpoint',
        type=str,
        default=None,
        help='Path to model ckpt for finetuning.')
    parser.add_argument(
        '--reduction',
        dest='reduction',
        type=int,
        default=None,
        help='Dataset reduction factor.')
    parser.add_argument(
        '--out_dir',
        dest='out_dir',
        type=str,
        default=None,
        help='Customized output directory for finetuned model.')
    parser.add_argument(
        '--gpu',
        dest='gpu_device',
        type=str,
        default='/gpu:0',
        help='GPU device.')
    parser.add_argument(
        '--cpu',
        dest='cpu_device',
        type=str,
        default='/cpu:0',
        help='CPU device.')
    parser.add_argument(
        '--transfer',
        dest='transfer',
        action='store_true',
        help='Enable the transfer learning routine.')
    parser.add_argument(
        '--no_db',
        dest='use_db',
        action='store_false',
        help='Do not use the db.')
    args = parser.parse_args()
    main(**vars(args))
