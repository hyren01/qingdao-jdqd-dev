import os
import six
import codecs
import argparse
import logging
import logging.handlers
import numpy as np
from .bert4keras_v01.snippets import sequence_padding


def str2bool(v):
    """
    String to Boolean
    """
    # because argparse does not support to parse "true, False" as python
    # boolean directly
    return v.lower() in ("true", "t", "1")


class ArgumentGroup(object):
    """
    Argument Class
    """

    def __init__(self, parser, title, des):
        self._group = parser.add_argument_group(title=title, description=des)

    def add_arg(self, name, type, default, help, **kwargs):
        """
        Add argument
        """
        type = str2bool if type == bool else type
        self._group.add_argument(
            "--" + name,
            default=default,
            type=type,
            help=help + ' Default: %(default)s.',
            **kwargs)


class ArgConfig(object):
    def __init__(self):
        parser = argparse.ArgumentParser()

        model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
        model_g.add_arg("config_path", str, None, "Path to the json file for pretrained model config.")
        model_g.add_arg("init_checkpoint", str, None, "Init checkpoint to resume training from.")
        model_g.add_arg("vocab_path", str, None, "Vocabulary path.")
        model_g.add_arg("output_dir", str, None, "Directory path to save checkpoints")

        train_g = ArgumentGroup(parser, "training", "training options.")
        train_g.add_arg("epoch", int, 10, "Number of epoches for training.")
        train_g.add_arg("gelu", str, "tanh", "Model gelu, could be tanh or erf.")
        train_g.add_arg("save_best", bool, True, "If True, only save best model.")

        # log_g = ArgumentGroup(parser, "logging", "logging related")
        # log_g.add_arg("verbose_result", bool, True, "Whether to output verbose result.")

        data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
        data_g.add_arg("train_data_dir", str, None, "Directory path to training data.")
        data_g.add_arg("valid_data_dir", str, None, "Directory path to valid data.")
        data_g.add_arg("test_data_dir", str, None, "Directory path to testing data.")
        data_g.add_arg("batch_size", int, 32, "Total examples' number in batch for training.")
        data_g.add_arg("max_length", int, 128, "Max length for per example.")

        custom_g = ArgumentGroup(parser, "customize", "customized options.")
        self.custom_g = custom_g

        self.parser = parser

    def add_arg(self, name, dtype, default, descrip):
        self.custom_g.add_arg(name, dtype, default, descrip)

    def build_conf(self):
        return self.parser.parse_args()


def print_arguments(args):
    """
    Print Arguments
    """
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(six.iteritems(vars(args))):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def init_log(
        log_path,
        level=logging.INFO,
        when="D",
        backup=7,
        format="%(levelname)s: %(asctime)s - %(filename)s:%(lineno)d * %(thread)d %(message)s",
        datefmt=None):
    """
    init_log - initialize log module

    Args:
      log_path      - Log file path prefix.
                      Log data will go to two files: log_path.log and log_path.log.wf
                      Any non-exist parent directories will be created automatically
      level         - msg above the level will be displayed
                      DEBUG < INFO < WARNING < ERROR < CRITICAL
                      the default value is logging.INFO
      when          - how to split the log file by time interval
                      'S' : Seconds
                      'M' : Minutes
                      'H' : Hours
                      'D' : Days
                      'W' : Week day
                      default value: 'D'
      format        - format of the log
                      default format:
                      %(levelname)s: %(asctime)s: %(filename)s:%(lineno)d * %(thread)d %(message)s
                      INFO: 12-09 18:02:42: log.py:40 * 139814749787872 HELLO WORLD
      backup        - how many backup file to keep
                      default value: 7

    Raises:
        OSError: fail to create log directories
        IOError: fail to open log file
    """
    formatter = logging.Formatter(format, datefmt)
    logger = logging.getLogger()
    logger.setLevel(level)

    # console Handler
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)
    logger.addHandler(consoleHandler)

    dir = os.path.dirname(log_path)
    if not os.path.isdir(dir):
        os.makedirs(dir)

    handler = logging.handlers.TimedRotatingFileHandler(
        log_path + ".log", when=when, backupCount=backup)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    handler = logging.handlers.TimedRotatingFileHandler(
        log_path + ".log.wf", when=when, backupCount=backup)
    handler.setLevel(logging.WARNING)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def set_level(level):
    """
    Reak-time set log level
    """
    logger = logging.getLogger()
    logger.setLevel(level)
    logging.info('log level is set to : %d' % level)


def get_level():
    """
    get Real-time log level
    """
    logger = logging.getLogger()
    return logger.level


def load_data(filename):
    D = []
    with codecs.open(filename, encoding='utf-8') as f:
        for l in f:
            split = l.strip().split('\t')
            if len(split) != 3:
                print(filename, split)
                continue
            text1, text2, label = split
            D.append((text1, text2, int(label)))
    return D


class data_generator:
    """数据生成器
    """

    def __init__(self, data, tokenizer, max_length=128, batch_size=16):
        self.data = data
        self.tokenizer = tokenizer
        self.maxlen = max_length
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self, random=False):
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for i in idxs:
            text1, text2, label = self.data[i]
            token_ids, segment_ids = self.tokenizer.encode(text1, text2, max_length=self.maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

    def forfit(self):
        while True:
            for d in self.__iter__(True):
                yield d


if __name__ == '__main__':
    args = ArgConfig()
    args = args.build_conf()

    print_arguments(args)
