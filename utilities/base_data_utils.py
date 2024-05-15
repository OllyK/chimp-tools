from enum import Enum
import logging
import sys
from inspect import currentframe, getframeinfo
from pathlib import Path
from types import SimpleNamespace
from collections import deque, defaultdict
import torch.distributed as dist
import datetime
import time
import pickle

import pandas as pd
import torch
import torchvision

import utilities.config as cfg
from utilities.settings_data import get_settings_data

class DetectorMethod(Enum):
    STANDARD = 1
    DETECTRON = 2
    PANOPTIC = 3

def create_enum_from_setting(setting_str, enum):
    if isinstance(setting_str, Enum):
        return setting_str
    try:
        output_enum = enum[setting_str.upper()]
    except KeyError:
        options = [k.name for k in enum]
        logging.error(
            f"{enum.__name__}: {setting_str} is not valid. Options are {options}."
        )
        sys.exit(1)
    return output_enum

def get_detector_method(settings: SimpleNamespace) -> Enum:
    detector_method = create_enum_from_setting(settings.detector_method, DetectorMethod)
    return detector_method

def check_csv_filepath(csv_filepath: str) -> Path:
    csv_filepath = Path(csv_filepath).resolve()
    if csv_filepath.exists():
        if csv_filepath.suffix != ".csv":
            logging.error("File needs to be a CSV file!")
            sys.exit(1)
        return csv_filepath
    else:
        logging.error("Settings file does not exist!")
        sys.exit(1)


def get_standard_training_data(csv_filepath, prepend_dir):
    # Read in CSV file and prepend directory if paths are relative
    csv_filepath = check_csv_filepath(csv_filepath)
    training_data = pd.read_csv(csv_filepath)
    if prepend_dir:
        prepend_dir = Path(prepend_dir).resolve()
        if prepend_dir.exists():
            training_data.iloc[:, 0] = training_data.iloc[:, 0].apply(
                lambda x: prepend_dir / x
            )
        else:
            logging.error(f"Prepend directory: {prepend_dir} does not exist!")
            sys.exit(1)
    return training_data

def get_classifier_settings(settings_file):
    if settings_file == cfg.CLF_TRAIN_SETTINGS_FN:
        filename = getframeinfo(currentframe()).filename
        # Select directory two up from this file
        parent = Path(filename).resolve().parents[1]
        settings_path = parent / Path(settings_file)
    else:
        settings_path = Path(settings_file)
    settings = get_settings_data(settings_path)
    return settings

def get_pre_split_data(csv_filepath, validation_filepath, prepend_dir):
    csv_filepath = check_csv_filepath(csv_filepath)
    training_data = pd.read_csv(csv_filepath)
    validation_data = pd.read_csv(validation_filepath)
    if prepend_dir:
        prepend_dir = Path(prepend_dir).resolve()
        if prepend_dir.exists():
            training_data.iloc[:, 0] = training_data.iloc[:, 0].apply(
                lambda x: prepend_dir / x
            )
            validation_data.iloc[:, 0] = validation_data.iloc[:, 0].apply(
                lambda x: prepend_dir / x
            )
        else:
            logging.error(f"Prepend directory: {prepend_dir} does not exist!")
            sys.exit(1)
    return training_data, validation_data

def get_detector_train_settings(settings_file):
    if settings_file == cfg.DET_TRAIN_SETTINGS_FN:
        filename = getframeinfo(currentframe()).filename
        # Select directory two up from this file
        parent = Path(filename).resolve().parents[1]
        settings_path = parent / Path(settings_file)
    else:
        settings_path = Path(settings_file)
    return get_settings_data(settings_path)

def get_available_device_type() -> str:
    if torch.cuda.is_available():
        return "cuda"
    else:
        try:
            mps_bool = torch.backends.mps.is_available()
        except Exception as e:
            logging.info("Pytorch version pre-dates MPS support.")
            return "cpu"
        if mps_bool:
            return "mps"
        else:
            return "cpu"

def get_batch_size(settings: SimpleNamespace, prediction: bool = False) -> int:
    device_type = get_available_device_type()
    if device_type == "cuda":
        cuda_device_num = settings.cuda_device
        total_gpu_mem = torch.cuda.get_device_properties(cuda_device_num).total_memory
        allocated_gpu_mem = torch.cuda.memory_allocated(cuda_device_num)
        free_gpu_mem = (total_gpu_mem - allocated_gpu_mem) / 1024**3
        logging.info(f"Free GPU memory is {free_gpu_mem:0.2f} GB.")
        if free_gpu_mem < cfg.BIG_CUDA_THRESHOLD:
            batch_size = cfg.SMALL_CUDA_BATCH
        elif not prediction:
            batch_size = cfg.BIG_CUDA_TRAIN_BATCH
        else:
            batch_size = cfg.BIG_CUDA_PRED_BATCH
    else:
        logging.info("MPS Device or CPU used.")
        batch_size = cfg.MPS_CPU_BATCH
    logging.info(f"Batch size will be {batch_size}.")
    return batch_size

def mask_r_cnn_collate_fn(batch):
    return tuple(zip(*batch))


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = self.delimiter.join([
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}',
            'max mem: {memory:.0f}'
        ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(log_msg.format(
                    i, len(iterable), eta=eta_string,
                    meters=str(self),
                    time=str(iter_time), data=str(data_time),
                    memory=torch.cuda.max_memory_allocated() / MB))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict

def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list
