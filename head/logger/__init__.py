import time
import logging
import chainer


def setup_logger(path=None):
    log_format = "[%(asctime)s:%(levelname)s] %(message)s"
    log_formatter = logging.Formatter(log_format)
    root_logger = logging.getLogger()
    if path:
        file_handler = logging.FileHandler(path)
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.INFO)


#
# Trainer extensions
#

class ObservationBasedLearningRateController(chainer.training.extension.Extension):

    """
    largely based on Pylearn2
    https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/termination_criteria/__init__.py#L41
    """

    def __init__(self, optimizer, reduction_radio=0.1,
                 observation_key='validation/main/loss',
                 expected_decrease=0.01, n_tolerance_steps=5):
        self.optimizer = optimizer
        self.reduction_ratio = reduction_radio
        self.observation_key = observation_key
        self.expected_decrease = expected_decrease
        self.n_tolerance_steps = n_tolerance_steps

        self.best_value = np.inf
        self.countdown = self.n_tolerance_steps

    def __call__(self, trainer):
        assert self.observation_key in trainer.observation
        value = trainer.observation[self.observation_key]

        if value < (1.0 - self.expected_decrease) * self.best_value:
            self.countdown = self.n_tolerance_steps
        else:
            self.countdown -= 1

        if value < self.best_value:
            self.best_value = value

        if self.countdown <= 0:
            self.optimizer.lr *= self.reduction_ratio
            self.countdown = self.n_tolerance_steps
            logging.info('Learning rate changed!: {}'.format(self.optimizer.lr))


def observe_value(key, target_func):
    """Returns a trainer extension to continuously record a value.

    Args:
        key (str): Key of observation to record.
        target_func (function): Function that returns the value to record.
            It must take one argument: trainer object.
    Returns:
        The extension function.
    """
    @chainer.training.extension.make_extension(
        trigger=(1, 'epoch'), priority=chainer.training.extension.PRIORITY_WRITER)
    def _observe_value(trainer):
        trainer.observation[key] = target_func(trainer)
    return _observe_value


def observe_time(key='time'):
    """Returns a trainer extension to record the elapsed time.

    Args:
        key (str): Key of observation to record.

    Returns:
        The extension function.
    """
    start_time = time.time()
    return observe_value(key, lambda _: time.time() - start_time)


def observe_lr(optimizer, key='lr'):
    """Returns a trainer extension to record the learning rate.

    Args:
        optimizer: Optimizer object whose learning rate is recorded.
        key (str): Key of observation to record.

    Returns:
        The extension function.
    """
    return observe_value(key, lambda _: optimizer.lr)
