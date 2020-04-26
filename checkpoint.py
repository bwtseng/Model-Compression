import shutil
import os 
import torch 
import distiller
import logging 
from errno import ENOENT
from tabulate import tabulate 
from distiller.utils import normalize_module_name
from tabulate import tabulate # Like LeTex Tabkle.
from numbers import Number  # what's this package used for?
msglogger = logging.getLogger()
def save_checkpoint(epoch, arch, model, optimizer=None, scheduler=None,
                    extras=None, is_best=False, name=None, dir='.'):
    """Save a pytorch training checkpoint
    Args:
        epoch: current epoch number
        arch: name of the network architecture/topology
        model: a pytorch model
        optimizer: the optimizer used in the training session
        scheduler: the CompressionScheduler instance used for training, if any
        extras: optional dict with additional user-defined data to be saved in the checkpoint.
            Will be saved under the key 'extras'
        is_best: If true, will save a copy of the checkpoint with the suffix 'best'
        name: the name of the checkpoint file
        dir: directory in which to save the checkpoint
    """
    if not os.path.isdir(dir):
        raise IOError(ENOENT, 'Checkpoint directory does not exist at', os.path.abspath(dir))

    if extras is None:
        extras = {}

    if not isinstance(extras, dict):
        raise TypeError('extras must be either a dict or None')

    filename = 'checkpoint.pth.tar' if name is None else name + '_checkpoint.pth.tar'
    fullpath = os.path.join(dir, filename)
    # msglogger.info("Saving checkpoint to: %s" % fullpath)
    print("Saving checkpoint to: {}".format(fullpath))
    filename_best = 'best.pth.tar' if name is None else name + '_best.pth.tar'
    fullpath_best = os.path.join(dir, filename_best)

    checkpoint = {'epoch': epoch, 'state_dict': model.state_dict(), 'arch': arch}
    try:
        checkpoint['is_parallel'] = model.is_parallel
        checkpoint['dataset'] = model.dataset
        if not arch:
            checkpoint['arch'] = model.arch
    except AttributeError:
        pass

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        checkpoint['optimizer_type'] = type(optimizer)
    if scheduler is not None:
        checkpoint['compression_sched'] = scheduler.state_dict()

    if hasattr(model, 'thinning_recipes'):
        checkpoint['thinning_recipes'] = model.thinning_recipes
    if hasattr(model, 'quantizer_metadata'):
        checkpoint['quantizer_metadata'] = model.quantizer_metadata

    checkpoint['extras'] = extras
    torch.save(checkpoint, fullpath)
    if is_best:
        # Pytorch love using shutil to manage file..
        shutil.copyfile(fullpath, fullpath_best)

# Draw the value table, tabulate is also the ploting table function in latex
def get_contents_table(d):
    def inspect_val(val):
        if isinstance(val, (Number, str)):
            return val
        elif isinstance(val, type):
            return val.__name__
        return None

    contents = [[k, type(d[k]).__name__, inspect_val(d[k])] for k in d.keys()]
    contents = sorted(contents, key=lambda entry: entry[0])
    return tabulate(contents, headers=["Key", "Type", "Value"], tablefmt="psql")


        
def load_lean_checkpoint(model, chkpt_file, model_device=None):
    return load_checkpoint(model, chkpt_file, model_device=model_device,
                           lean_checkpoint=True)[0]

def load_checkpoint(model, chkpt_file, optimizer=None,
                    model_device=None, lean_checkpoint=False, strict=False):
    """Load a pytorch training checkpoint.
    Args:
        model: the pytorch model to which we will load the parameters.  You can
        specify model=None if the checkpoint contains enough metadata to infer
        the model.  The order of the arguments is misleading and clunky, and is
        kept this way for backward compatibility.
        chkpt_file: the checkpoint file
        lean_checkpoint: if set, read into model only 'state_dict' field
        optimizer: [deprecated argument]
        model_device [str]: if set, call model.to($model_device)
                This should be set to either 'cpu' or 'cuda'.
    :returns: updated model, compression_scheduler, optimizer, start_epoch
    """
    def _load_compression_scheduler():
        normalize_keys = False
        try:
            compression_scheduler.load_state_dict(checkpoint['compression_sched'], normalize_keys)
        except KeyError as e:
            # A very common source of this KeyError is loading a GPU model on the CPU.
            # We rename all of the DataParallel keys because DataParallel does not execute on the CPU.
            normalize_keys = True
            compression_scheduler.load_state_dict(checkpoint['compression_sched'], normalize_keys)
        # ******************** 
        # In Log file column 28.
        # ********************
        msglogger.info("Loaded compression schedule from checkpoint (epoch {})".format(
                                                                            checkpoint_epoch))
        #print("Loaded compression schedule from checkpoint (epoch {})".format(
        #        checkpoint_epoch))
        return normalize_keys

    def _load_and_execute_thinning_recipes():
        #print("Loaded a thinning recipe from the checkpoint")
        msglogger.info("Loaded a thinning recipe from the checkpoint")
        # Cache the recipes in case we need them later
        model.thinning_recipes = checkpoint['thinning_recipes']
        if normalize_dataparallel_keys:
            model.thinning_recipes = [distiller.get_normalized_recipe(recipe)
                                      for recipe in model.thinning_recipes]
        distiller.execute_thinning_recipes_list(model,
                                                compression_scheduler.zeros_mask_dict,
                                                model.thinning_recipes)

    def _load_optimizer():
        """Initialize optimizer with model parameters and load src_state_dict"""
        try:
            cls, src_state_dict = checkpoint['optimizer_type'], checkpoint['optimizer_state_dict']
            # Initialize the dest_optimizer with a dummy learning rate,
            # this is required to support SGD.__init__()
            dest_optimizer = cls(model.parameters(), lr=1)
            dest_optimizer.load_state_dict(src_state_dict)
            #print('Optimizer of type {} was loaded from checkpoint'.format(type(dest_optimizer)))
            msglogger.info('Optimizer of type {type} was loaded from checkpoint'.format(
                            type=type(dest_optimizer)))
            optimizer_param_groups = dest_optimizer.state_dict()['param_groups']
            #print('Optimizer Args: {}'.format(
            #                dict((k, v) for k, v in optimizer_param_groups[0].items()
            #                     if k != 'params')))
            msglogger.info('Optimizer Args: {}'.format(
                            dict((k, v) for k, v in optimizer_param_groups[0].items()
                                 if k != 'params')))
            return dest_optimizer
        except KeyError:
            # Older checkpoints do support optimizer loading: They either had an 'optimizer' field
            # (different name) which was not used during the load, or they didn't even checkpoint
            # the optimizer.
            #print('Optimizer could not be loaded from checkpoint.')
            msglogger.warning('Optimizer could not be loaded from checkpoint.')
            return None

    def _create_model_from_ckpt():
        try:
            return distiller.models.create_model(False, checkpoint['dataset'], checkpoint['arch'],
                                                 checkpoint['is_parallel'], device_ids=None)
        except KeyError:
            return None

    def _sanity_check():
        try:
            if model.arch != checkpoint["arch"]:
                raise ValueError("The model architecture does not match the checkpoint architecture")
        except (AttributeError, KeyError):
            # One of the values is missing so we can't perform the comparison
            pass

    chkpt_file = os.path.expanduser(chkpt_file)
    if not os.path.isfile(chkpt_file):
        raise IOError(ENOENT, 'Could not find a checkpoint file at', chkpt_file)
    assert optimizer == None, "argument optimizer is deprecated and must be set to None"

    #print("=> loading checkpoint {}.".format(chkpt_file))
    msglogger.info("= loading checkpoint %s", chkpt_file)
    checkpoint = torch.load(chkpt_file, map_location=lambda storage, loc: storage)
    # *********************
    # in log file column 15.
    # *********************
    #print("=> Checkpoint contents:\n {} \n".format(get_contents_table(checkpoint)))
    msglogger.info("=> Checkpoint contents:\n %s \n", get_contents_table(checkpoint))
    if 'extras' in checkpoint:
        #print("=> Checkpoint['extras'] contents:\n{}\n".format(get_contents_table(checkpoint['extras'])))
        msglogger.info("=> Checkpoint['extras'] contents:\n%s\n", get_contents_table(checkpoint['extras']))
    if 'state_dict' not in checkpoint:
        raise ValueError("Checkpoint must contain the model parameters under the key 'state_dict'")

    """
    if not model:
        model = _create_model_from_ckpt()
        if not model:
            raise ValueError("You didn't provide a model, and the checkpoint %s doesn't contain "
                             "enough information to create one", chkpt_file)
    """

    checkpoint_epoch = checkpoint.get('epoch', None)
    start_epoch = checkpoint_epoch + 1 if checkpoint_epoch is not None else 0
    compression_scheduler = None
    normalize_dataparallel_keys = False
    if 'compression_sched' in checkpoint:
        compression_scheduler = distiller.CompressionScheduler(model)
        normalize_dataparallel_keys = _load_compression_scheduler()
    else:
        #print("Warning: compression schedule data does not exist in the checkpoint")
        msglogger("Warning: compression schedule data does not exist in the checkpoint")

    if 'thinning_recipes' in checkpoint:
        if not compression_scheduler:
            print("Found thinning_recipes key, but missing key compression_scheduler")
            compression_scheduler = distiller.CompressionScheduler(model)
        _load_and_execute_thinning_recipes()

    if 'quantizer_metadata' in checkpoint:
        print('Loaded quantizer metadata from the checkpoint')
        qmd = checkpoint['quantizer_metadata']
        quantizer = qmd['type'](model, **qmd['params'])
        quantizer.prepare_model(qmd['dummy_input'])

        if qmd.get('pytorch_convert', False):
            #print('Converting Distiller PTQ model to PyTorch quantization API')
            msglogger.info('Converting Distiller PTQ model to PyTorch quantization API')
            model = quantizer.convert_to_pytorch(qmd['dummy_input'], backend=qmd.get('pytorch_convert_backend', None))

    if normalize_dataparallel_keys:
        checkpoint['state_dict'] = {normalize_module_name(k): v for k, v in checkpoint['state_dict'].items()}

    anomalous_keys = model.load_state_dict(checkpoint['state_dict'], strict)
    if anomalous_keys:
        # This is pytorch 1.1+
        missing_keys, unexpected_keys = anomalous_keys
        if unexpected_keys:
            msglogger.warning("Warning: the loaded checkpoint (%s) contains %d unexpected state keys" %
                              (chkpt_file, len(unexpected_keys)))
            #print("Warning: the loaded checkpoint {} contains {} unexpected state keys.".format(
            #                  (chkpt_file, len(unexpected_keys))))
        if missing_keys:
            raise ValueError("The loaded checkpoint (%s) is missing %d state keys" %
                             (chkpt_file, len(missing_keys)))

    if model_device is not None:
        model.to(model_device)

    if lean_checkpoint:
        print("=> loaded 'state_dict' from checkpoint '{}'".format(str(chkpt_file)))
        return model, None, None, 0

    optimizer = _load_optimizer()
    #print("=> loaded checkpoint '{f}' (epoch {e})".format(f=str(chkpt_file),
    #                                                               e=checkpoint_epoch))
    msglogger.info("=> loaded checkpoint '{f}' (epoch {e})".format(f=str(chkpt_file),
                                                                   e=checkpoint_epoch))
    _sanity_check()
    return model, compression_scheduler, optimizer, start_epoch


def create_activation_stats_collectors(model, *phases):
    """Create objects that collect activation statistics.
    This is a utility function that creates two collectors:
    1. Fine-grade sparsity levels of the activations
    2. L1-magnitude of each of the activation channels
    Args:
        model - the model on which we want to collect statistics
        phases - the statistics collection phases: train, valid, and/or test
    WARNING! Enabling activation statsitics collection will significantly slow down training!
    """
    class missingdict(dict):
        """This is a little trick to prevent KeyError"""
        def __missing__(self, key):
            return None  # note, does *not* set self[key] - we don't want defaultdict's behavior

    genCollectors = lambda: missingdict({
        "sparsity_ofm":      SummaryActivationStatsCollector(model, "sparsity_ofm",
            lambda t: 100 * distiller.utils.sparsity(t)),
        "l1_channels":   SummaryActivationStatsCollector(model, "l1_channels",
                                                         distiller.utils.activation_channels_l1),
        "apoz_channels": SummaryActivationStatsCollector(model, "apoz_channels",
                                                         distiller.utils.activation_channels_apoz),
        "mean_channels": SummaryActivationStatsCollector(model, "mean_channels",
                                                         distiller.utils.activation_channels_means),
        "records":       RecordsActivationStatsCollector(model, classes=[torch.nn.Conv2d])
    })

    return {k: (genCollectors() if k in phases else missingdict())
            for k in ('train', 'valid', 'test')}



'''
class SparsityAccuracyTracker(TrainingPerformanceTracker):
    """A performance tracker which prioritizes non-zero parameters.
    Sort the performance history using the count of non-zero parameters
    as main sort key, then sort by top1, top5 and and finally epoch number.
    Expects 'top1' and 'top5' to appear in the kwargs.
    """
    def step(self, model, epoch, **kwargs):
        assert all(score in kwargs.keys() for score in ('top1', 'top5'))
        model_sparsity, _, params_nnz_cnt = distiller.model_params_stats(model)
        self.perf_scores_history.append(distiller.MutableNamedTuple({
            'params_nnz_cnt': -params_nnz_cnt,
            'sparsity': model_sparsity,
            'top1': kwargs['top1'],
            'top5': kwargs['top5'],
            'epoch': epoch}))
        # Keep perf_scores_history sorted from best to worst
        self.perf_scores_history.sort(
            key=operator.attrgetter('params_nnz_cnt', 'top1', 'top5', 'epoch'),
            reverse=True)


def train_one_epoch(self, epoch, verbose=True):
    """Train for one epoch"""
    self.load_datasets()

    with collectors_context(self.activations_collectors["train"]) as collectors:
        top1, top5, loss = train(self.train_loader, self.model, self.criterion, self.optimizer, 
                                    epoch, self.compression_scheduler, 
                                    loggers=[self.tflogger, self.pylogger], args=self.args)
        if verbose:
            distiller.log_weights_sparsity(self.model, epoch, [self.tflogger, self.pylogger])
        distiller.log_activation_statistics(epoch, "train", loggers=[self.tflogger],
                                            collector=collectors["sparsity"])
        if self.args.masks_sparsity:
            msglogger.info(distiller.masks_sparsity_tbl_summary(self.model, 
                                                                self.compression_scheduler))
    return top1, top5, loss


# reference
def create_model(pretrained, dataset, arch, parallel=True, device_ids=None):
    """Create a pytorch model based on the model architecture and dataset
    Args:
        pretrained [boolean]: True is you wish to load a pretrained model.
            Some models do not have a pretrained version.
        dataset: dataset name (only 'imagenet' and 'cifar10' are supported)
        arch: architecture name
        parallel [boolean]: if set, use torch.nn.DataParallel
        device_ids: Devices on which model should be created -
            None - GPU if available, otherwise CPU
            -1 - CPU
            >=0 - GPU device IDs
    """
    dataset = dataset.lower()
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError('Dataset {} is not supported'.format(dataset))

    model = None
    cadene = False
    try:
        if dataset == 'imagenet':
            model, cadene = _create_imagenet_model(arch, pretrained)
        elif dataset == 'cifar10':
            model = _create_cifar10_model(arch, pretrained)
        elif dataset == 'mnist':
            model = _create_mnist_model(arch, pretrained)
    except ValueError:
        if _is_registered_extension(arch, dataset, pretrained):
            model = _create_extension_model(arch, dataset)
        else:
            raise ValueError('Could not recognize dataset {} and arch {} pair'.format(dataset, arch))

    msglogger.info("=> created a %s%s model with the %s dataset" % ('pretrained ' if pretrained else '',
                                                                     arch, dataset))
    if torch.cuda.is_available() and device_ids != -1:
        device = 'cuda'
        if parallel:
            if arch.startswith('alexnet') or arch.startswith('vgg'):
                model.features = torch.nn.DataParallel(model.features, device_ids=device_ids)
            else:
                model = torch.nn.DataParallel(model, device_ids=device_ids)
        model.is_parallel = parallel
    else:
        device = 'cpu'
        model.is_parallel = False

    # Cache some attributes which describe the model
    _set_model_input_shape_attr(model, arch, dataset, pretrained, cadene)
    model.arch = arch
    model.dataset = dataset
    return model.to(device)


'''

