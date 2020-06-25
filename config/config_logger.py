#import os 
#import logging
#import argparse
import logging
import logging.config
import operator
import os
import platform
import shutil
import sys
import time
import pkg_resources
from git import Repo, InvalidGitRepositoryError
import numpy as np
import torch
import tabulate
import distiller
from .tbbackend import TBBackend
import summary
import utility as utl
try:
    import lsb_release
    HAVE_LSB = True
except ImportError:
    HAVE_LSB = False
#density, sparsity, sparsity_2D, size_to_str, to_np, norm_filters
__all__ = ['log_execution_env_state', 'config_pylogger', 
           'PythonLogger', 'TensorBoardLogger', 'CsvLogger', 'NullLogger']

logger = logging.getLogger('app_cfg')

def log_execution_env_state(config_paths=None, logdir=None):
    """Log information about the execution environment.
    Files in 'config_paths' will be copied to directory 'logdir'. A common use-case
    is passing the path to a (compression) schedule YAML file. Storing a copy
    of the schedule file, with the experiment logs, is useful in order to
    reproduce experiments.
    Args:
        config_paths: path(s) to config file(s), used only when logdir is set
        logdir: log directory
        git_root: the path to the .git root directory
    """

    ""
    def log_git_state():
        """Log the state of the git repository.
        It is useful to know what git tag we're using, and if we have outstanding code.
        """
        try:
            repo = Repo(os.path.join(os.path.dirname(__file__), '..', '..'))
            assert not repo.bare
        except InvalidGitRepositoryError:
            logger.debug("Cannot find a Git repository.  You probably downloaded an archive of Distiller.")
            return

        if repo.is_dirty():
            logger.debug("Git is dirty")
        try:
            branch_name = repo.active_branch.name
        except TypeError:
            branch_name = "None, Git is in 'detached HEAD' state"
        logger.debug("Active Git branch: %s", branch_name)
        logger.debug("Git commit: %s" % repo.head.commit.hexsha)

    try:
        num_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        num_cpus = os.cpu_count()
    logger.debug("Number of CPUs: %d", num_cpus)
    logger.debug("Number of GPUs: %d", torch.cuda.device_count())
    logger.debug("CUDA version: %s", torch.version.cuda)
    logger.debug("CUDNN version: %s", torch.backends.cudnn.version())
    logger.debug("Kernel: %s", platform.release())
    if HAVE_LSB:
        logger.debug("OS: %s", lsb_release.get_lsb_information()['DESCRIPTION'])
    logger.debug("Python: %s", sys.version)
    try:
        logger.debug("PYTHONPATH: %s", os.environ['PYTHONPATH'])
    except KeyError:
        pass
    def _pip_freeze():
        return {x.key:x.version for x in sorted(pkg_resources.working_set,
                                                key=operator.attrgetter('key'))}
    logger.debug("pip freeze: {}".format(_pip_freeze()))
    log_git_state()
    logger.debug("Command line: %s", " ".join(sys.argv))
    if (logdir is None) or (config_paths is None):
        return

    # clone configuration files to output directory
    configs_dest = os.path.join(logdir, 'configs')

    if isinstance(config_paths, str) or not hasattr(config_paths, '__iter__'):
        config_paths = [config_paths]
    for cpath in config_paths:
        os.makedirs(configs_dest, exist_ok=True)

        if os.path.exists(os.path.join(configs_dest, os.path.basename(cpath))):
            logger.debug('{} already exists in logdir'.format(
                os.path.basename(cpath) or cpath))
        else:
            try:
                shutil.copy(cpath, configs_dest)
            except OSError as e:
                logger.debug('Failed to copy of config file: {}'.format(str(e)))

        
def config_pylogger(log_cfg_file, experiment_name, output_dir='logs', verbose=False):
    """Configure the Python logger.
    For each execution of the application, we'd like to create a unique log directory.
    By default this directory is named using the date and time of day, so that directories
    can be sorted by recency.  You can also name your experiments and prefix the log
    directory with this name.  This can be useful when accessing experiment data from
    TensorBoard, for example.
    """
    timestr = time.strftime("%Y.%m.%d-%H%M%S")
    exp_full_name = timestr if experiment_name is None else experiment_name + '_' + timestr
    logdir = os.path.join(output_dir, exp_full_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    log_filename = os.path.join(logdir, exp_full_name + '.log')
    if os.path.isfile(log_cfg_file):
        logging.config.fileConfig(log_cfg_file, defaults={'logfilename': log_filename})
    else:
        print("Could not find the logger configuration file (%s) - using default logger configuration" % log_cfg_file)
        apply_default_logger_cfg(log_filename)
    msglogger = logging.getLogger()
    msglogger.logdir = logdir
    msglogger.log_filename = log_filename
    if verbose:
        msglogger.setLevel(logging.DEBUG)
    msglogger.info('Log file for this run: ' + os.path.realpath(log_filename))

    # Create a symbollic link to the last log file created (for easier access)
    try:
        os.unlink("latest_log_file")
    except FileNotFoundError:
        pass
    try:
        os.unlink("latest_log_dir")
    except FileNotFoundError:
        pass
    try:
        os.symlink(logdir, "latest_log_dir")
        os.symlink(log_filename, "latest_log_file")
    except OSError:
        msglogger.debug("Failed to create symlinks to latest logs")
    return msglogger

def apply_default_logger_cfg(log_filename):
    d = {
        'version': 1,
        'formatters': {
            'simple': {
                'class': 'logging.Formatter',
                'format': '%(asctime)s - %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
            },
            'file': {
                'class': 'logging.FileHandler',
                'filename': log_filename,
                'mode': 'w',
                'formatter': 'simple',
            },
        },
        'loggers': {
            '': {  # root logger
                'level': 'DEBUG',
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'app_cfg': {
                'level': 'DEBUG',
                'handlers': ['file'],
                'propagate': False
            },
        }
    }
    logging.config.dictConfig(d)


class DataLogger(object):
    """This is an abstract interface for data loggers
    Data loggers log the progress of the training process to some backend.
    This backend can be a file, a web service, or some other means to collect and/or
    display the training
    """
    def __init__(self):
        pass

    def log_training_progress(self, stats_dict, epoch, completed, total, freq):
        pass

    def log_activation_statistic(self, phase, stat_name, activation_stats, epoch):
        pass

    def log_weights_sparsity(self, model, epoch):
        pass

    def log_weights_distribution(self, named_params, steps_completed):
        pass

    def log_model_buffers(self, model, buffer_names, tag_prefix, epoch, completed, total, freq):
        pass


# Log to null-space
NullLogger = DataLogger
class PythonLogger(DataLogger):
    def __init__(self, logger):
        super(PythonLogger, self).__init__()
        self.pylogger = logger

    def log_training_progress(self, stats_dict, epoch, completed, total, freq):
        stats_dict = stats_dict[1]
        if epoch > -1:
            log = 'Epoch: [{}][{:5d}/{:5d}]    '.format(epoch, completed, int(total))
        else:
            log = 'Test: [{:5d}/{:5d}]    '.format(completed, int(total))

        for name, val in stats_dict.items():
            if isinstance(val, int):
                log = log + '{name} {val}    '.format(name=name, val=distiller.pretty_int(val))
            else:
                log = log + '{name} {val:.6f}    '.format(name=name, val=val)
        self.pylogger.info(log)

    def log_activation_statistic(self, phase, stat_name, activation_stats, epoch):
        data = []
        for layer, statistic in activation_stats.items():
            data.append([layer, statistic])
        t = tabulate.tabulate(data, headers=['Layer', stat_name], tablefmt='psql', floatfmt=".2f")
        self.pylogger.info('\n' + t)

    def log_weights_sparsity(self, model, epoch):
        t, total = summary.weights_sparsity_tbl_summary(model, return_total_sparsity=True)
        self.pylogger.info("\nParameters:\n" + str(t))
        self.pylogger.info('Total sparsity: {:0.2f}\n'.format(total))

    def log_model_buffers(self, model, buffer_names, tag_prefix, epoch, completed, total, freq):
        """Logs values of model buffers.
        Notes:
            1. Each buffer provided in 'buffer_names' is displayed in a separate table.
            2. Within each table, each value is displayed in a separate column.
        """
        datas = {name: [] for name in buffer_names}
        maxlens = {name: 0 for name in buffer_names}
        for n, m in model.named_modules():
            for buffer_name in buffer_names:
                try:
                    p = getattr(m, buffer_name)
                except AttributeError:
                    continue
                data = datas[buffer_name]
                values = p if isinstance(p, (list, torch.nn.ParameterList)) else p.view(-1).tolist()
                data.append([distiller.normalize_module_name(n) + '.' + buffer_name, *values])
                maxlens[buffer_name] = max(maxlens[buffer_name], len(values))

        for name in buffer_names:
            if datas[name]:
                headers = ['Layer'] + ['Val_' + str(i) for i in range(maxlens[name])]
                t = tabulate.tabulate(datas[name], headers=headers, tablefmt='psql', floatfmt='.4f')
                self.pylogger.info('\n' + name.upper() + ': (Epoch {0}, Step {1})\n'.format(epoch, completed) + t)


class TensorBoardLogger(DataLogger):
    def __init__(self, logdir):
        super(TensorBoardLogger, self).__init__()
        # Set the tensorboard logger
        self.tblogger = TBBackend(logdir)
        print('\n--------------------------------------------------------')
        print('Logging to TensorBoard - remember to execute the server:')
        print('> tensorboard --logdir=\'./logs\'\n')

        # Hard-code these preferences for now
        self.log_gradients = False      # True
        self.logged_params = ['weight'] # ['weight', 'bias']

    def log_training_progress(self, stats_dict, epoch, completed, total, freq):
        def total_steps(total, epoch, completed):
            return total*epoch + completed

        prefix = stats_dict[0]
        stats_dict = stats_dict[1]

        for tag, value in stats_dict.items():
            self.tblogger.scalar_summary(prefix+tag, value, total_steps(total, epoch, completed))
        self.tblogger.sync_to_file()

    def log_activation_statistic(self, phase, stat_name, activation_stats, epoch):
        group = stat_name + '/activations/' + phase + "/"
        for tag, value in activation_stats.items():
            self.tblogger.scalar_summary(group+tag, value, epoch)
        self.tblogger.sync_to_file()

    def log_weights_sparsity(self, model, epoch):
        params_size = 0
        sparse_params_size = 0

        for name, param in model.state_dict().items():
            if param.dim() in [2, 4]:
                _density = utl.density(param)
                params_size += torch.numel(param)
                sparse_params_size += param.numel() * _density
                self.tblogger.scalar_summary('sparsity/weights/' + name,
                                             utl.sparsity(param)*100, epoch)
                self.tblogger.scalar_summary('sparsity-2D/weights/' + name,
                                             utl.sparsity_2D(param)*100, epoch)

        self.tblogger.scalar_summary("sparsity/weights/total", 100*(1 - sparse_params_size/params_size), epoch)
        self.tblogger.sync_to_file()

    def log_weights_filter_magnitude(self, model, epoch, multi_graphs=False):
        """Log the L1-magnitude of the weights tensors.
        """
        for name, param in model.state_dict().items():
            if param.dim() in [4]:
                self.tblogger.list_summary('magnitude/filters/' + name,
                                           list(utl.to_np(utl.norm_filters(param))), epoch, multi_graphs)
        self.tblogger.sync_to_file()

    def log_weights_distribution(self, named_params, steps_completed):
        if named_params is None:
            return
        for tag, value in named_params:
            tag = tag.replace('.', '/')
            if any(substring in tag for substring in self.logged_params):
                self.tblogger.histogram_summary(tag, utl.to_np(value), steps_completed)
            if self.log_gradients:
                self.tblogger.histogram_summary(tag+'/grad', utl.to_np(value.grad), steps_completed)
        self.tblogger.sync_to_file()

    def log_model_buffers(self, model, buffer_names, tag_prefix, epoch, completed, total, freq):
        """Logs values of model buffers.
        Notes:
            1. Buffers are logged separately per-layer (i.e. module) within model
            2. All values in a single buffer are logged such that they will be displayed on the same graph in
               TensorBoard
            3. Similarly, if multiple buffers are provided in buffer_names, all are presented on the same graph.
               If this is un-desirable, call the function separately for each buffer
            4. USE WITH CAUTION: While sometimes desirable, displaying multiple distinct values in a single
               graph isn't well supported in TensorBoard. It is achieved using a work-around, which slows
               down TensorBoard loading time considerably as the number of distinct values increases.
               Therefore, while not limited, this function is only meant for use with a very limited number of
               buffers and/or values, e.g. 2-5.
        """
        for module_name, module in model.named_modules():
            if distiller.has_children(module):
                continue

            sd = module.state_dict()
            values = []
            for buf_name in buffer_names:
                try:
                    values += sd[buf_name].view(-1).tolist()
                except KeyError:
                    continue

            if values:
                tag = '/'.join([tag_prefix, module_name])
                self.tblogger.list_summary(tag, values, total * epoch + completed, len(values) > 1)
        self.tblogger.sync_to_file()


class CsvLogger(DataLogger):
    def __init__(self, fname_prefix='', logdir=''):
        super(CsvLogger, self).__init__()
        self.logdir = logdir
        self.fname_prefix = fname_prefix

    def get_fname(self, postfix):
        fname = postfix + '.csv'
        if self.fname_prefix:
            fname = self.fname_prefix + '_' + fname
        return os.path.join(self.logdir, fname)

    def log_weights_sparsity(self, model, epoch):
        fname = self.get_fname('weights_sparsity')
        with open(fname, 'w') as csv_file:
            params_size = 0
            sparse_params_size = 0

            writer = csv.writer(csv_file)
            # write the header
            writer.writerow(['parameter', 'shape', 'volume', 'sparse volume', 'sparsity level'])

            for name, param in model.state_dict().items():
                if param.dim() in [2, 4]:
                    _density = utl.density(param)
                    params_size += torch.numel(param)
                    sparse_params_size += param.numel() * _density
                    writer.writerow([name, utl.size_to_str(param.size()),
                                     torch.numel(param),
                                     int(_density * param.numel()),
                                     (1-_density)*100])

    def log_model_buffers(self, model, buffer_names, tag_prefix, epoch, completed, total, freq):
        """Logs values of model buffers.
        Notes:
            1. Each buffer provided is logged in a separate CSV file
            2. Each CSV file is continuously updated during the run.
            3. In each call, a line is appended for each layer (i.e. module) containing the named buffers.
        """
        with ExitStack() as stack:
            files = {}
            writers = {}
            for buf_name in buffer_names:
                fname = self.get_fname(buf_name)
                new = not os.path.isfile(fname)
                files[buf_name] = stack.enter_context(open(fname, 'a'))
                writer = csv.writer(files[buf_name])
                if new:
                    writer.writerow(['Layer', 'Epoch', 'Step', 'Total', 'Values'])
                writers[buf_name] = writer

            for n, m in model.named_modules():
                for buffer_name in buffer_names:
                    try:
                        p = getattr(m, buffer_name)
                    except AttributeError:
                        continue
                    writer = writers[buffer_name]
                    if isinstance(p, (list, torch.nn.ParameterList)):
                        values = []
                        for v in p:
                            values += v.view(-1).tolist()
                    else:
                        values = p.view(-1).tolist()
                    writer.writerow([distiller.normalize_module_name(n) + '.' + buffer_name,
                                     epoch, completed, int(total)] + values)



"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test logger file.')
    parser.add_argument('--out_dir', default='test', type=str, help='Directory to save output log file.')
    parser.add_argument('--learning-rate-decay', '--lrd', default=0.7, type=floa
"""


