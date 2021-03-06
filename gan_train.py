import sys
import yaml
from data import build_datasets
from model import build_model, build_algorithm
from monitor import build_extensions
from blocks.model import Model
from blocks.main_loop import TrainingFinish

"""The event-based main loop of Blocks."""
import signal
import logging
import traceback

from blocks.config import config
from blocks.log import BACKENDS
from blocks.utils import reraise_as, unpack, change_recursion_limit
from blocks.utils.profile import Profile, Timer
from blocks.algorithms import DifferentiableCostMinimizer
from blocks.extensions import CallbackName
from blocks.model import Model

import numpy as np
from theano.compile.sharedvalue import SharedVariable
from theano import tensor
import theano
from blocks.graph import ComputationGraph
from blocks.bricks import MLP, Rectifier, Softmax, Maxout, Logistic, Identity, Tanh
from blocks.initialization import IsotropicGaussian, Constant
from blocks.algorithms import GradientDescent, Scale, RMSProp, Momentum
from blocks.extensions import Printing, Timing
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.utils import shared_like

logger = logging.getLogger(__name__)

error_message = """

Blocks will attempt to run `on_error` extensions, potentially saving data, \
before exiting and reraising the error. Note that the usual `after_training` \
extensions will *not* be run. The original error will be re-raised and also \
stored in the training log. Press CTRL + C to halt Blocks immediately."""

error_in_error_handling_message = """

Blocks will now exit. The remaining `on_error` extensions will not be run."""


epoch_interrupt_message = """

Blocks will complete this epoch of training and run extensions \
before exiting. If you do not want to complete this epoch, press CTRL + C \
again to stop training after the current batch."""

batch_interrupt_message = """

Blocks will complete the current batch and run extensions before exiting. If \
you do not want to complete this batch, press CTRL + C again. WARNING: Note \
that this will end training immediately, and extensions that e.g. save your \
training progress won't be run."""

no_model_message = """

A possible reason: one of your extensions requires the main loop to have \
a model. Check documentation of your extensions."""

class GANMainLoop(object):
    """The standard main loop for GAN.

    Parameters
    ----------
    algorithm_g : instance of :class:`~blocks.algorithms.TrainingAlgorithm`
        The training algorithm for the generator.
    algorithm_d : instance of :class:`~blocks.algorithms.TrainingAlgorithm`
        The training algorithm for the discriminator.
    data_stream : instance of :class:`.DataStream`.
        The data stream. Should support :class:`AbstractDataStream`
        interface from Fuel.
    generator : instance of :class:`.ComputationGraph`
    discriminator : instance of :class:`.ComputationGraph`
    log : instance of :class:`.TrainingLog`, optional
        The log. When not given, a :class:`.TrainingLog` is created.
    log_backend : str
        The backend to use for the log. Currently `python` and `sqlite` are
        available. If not given, `config.log_backend` will be used. Ignored
        if `log` is passed.
    extensions : list of :class:`.TrainingExtension` instances
        The training extensions. Will be called in the same order as given
        here.

    """
    def __init__(self, algorithm_g, g_out, algorithm_d, d_out, data_stream, 
                 false_generated, false_dataset, generator_errors,
                 generator=None, discriminator=None, noise_per_sample=10, k=1,
                 minibatches=1, log=None, log_backend=None, extensions=None, observables=[]):
        if log is None:
            if log_backend is None:
                log_backend = config.log_backend
            log = BACKENDS[log_backend]()
        if extensions is None:
            extensions = []

        self.data_stream = data_stream
        self.algorithm = algorithm_g
        self.algorithm_g = algorithm_g
        self.algorithm_d = algorithm_d
        self.log = log
        self.extensions = extensions
        self.g_out = g_out
        self.d_out = d_out
        self.k = k
        self.minibatches = minibatches
        self.noise_per_sample = noise_per_sample
        self.false_generated = false_generated
        self.false_dataset = false_dataset
        self.generator_errors = generator_errors
        self.observables = observables
        self.first_batch = None

        self.profile = Profile()

        self._generator = generator

        self._discriminator = discriminator

        self.status['training_started'] = False
        self.status['epoch_started'] = False
        self.status['epoch_interrupt_received'] = False
        self.status['batch_interrupt_received'] = False

    @property
    def generator(self):
        if not self._generator:
            raise AttributeError("no generator in this main loop" +
                                 no_model_message)
        return self._generator

    @property
    def discriminator(self):
        if not self._discriminator:
            raise AttributeError("no discriminator in this main loop" +
                                 no_model_message)
        return self._discriminator

    @property
    def iteration_state(self):
        """Quick access to the (data stream, epoch iterator) pair."""
        return (self.data_stream, self.epoch_iterator)

    @iteration_state.setter
    def iteration_state(self, value):
        (self.data_stream, self.epoch_iterator) = value

    @property
    def status(self):
        """A shortcut for `self.log.status`."""
        return self.log.status

    def run(self):
        """Starts the main loop.

        The main loop ends when a training extension makes
        a `training_finish_requested` record in the log.

        """
        # This should do nothing if the user has already configured
        # logging, and will it least enable error messages otherwise.
        logging.basicConfig()

        # If this is resumption from a checkpoint, it is crucial to
        # reset `profile.current`. Otherwise, it simply does not hurt.
        self.profile.current = []

        with change_recursion_limit(config.recursion_limit):
            self.original_sigint_handler = signal.signal(
                signal.SIGINT, self._handle_epoch_interrupt)
            self.original_sigterm_handler = signal.signal(
                signal.SIGTERM, self._handle_batch_interrupt)
            try:
                logger.info("Entered the main loop")
                if not self.status['training_started']:
                    for extension in self.extensions:
                        extension.main_loop = self
                    self._run_extensions('before_training')
                    with Timer('initialization', self.profile):
                        self.algorithm_g.initialize()
                        self.algorithm_d.initialize()
                    self.status['training_started'] = True
                # We can not write "else:" here because extensions
                # called "before_training" could have changed the status
                # of the main loop.
                if self.log.status['iterations_done'] > 0:
                    self.log.resume()
                    self._run_extensions('on_resumption')
                    self.status['epoch_interrupt_received'] = False
                    self.status['batch_interrupt_received'] = False
                with Timer('training', self.profile):
                    while self._run_epoch():
                        pass
            except TrainingFinish:
                self.log.current_row['training_finished'] = True
            except Exception as e:
                self._restore_signal_handlers()
                self.log.current_row['got_exception'] = traceback.format_exc()
                logger.error("Error occured during training." + error_message)
                try:
                    self._run_extensions('on_error')
                except Exception:
                    logger.error(traceback.format_exc())
                    logger.error("Error occured when running extensions." +
                                 error_in_error_handling_message)
                reraise_as(e)
            finally:
                self._restore_signal_handlers()
                if self.log.current_row.get('training_finished', False):
                    self._run_extensions('after_training')
                if config.profile:
                    self.profile.report()

    def find_extension(self, name):
        """Find an extension with a given name.

        Parameters
        ----------
        name : str
            The name of the extension looked for.

        Notes
        -----
        Will crash if there no or several extension found.

        """
        return unpack([extension for extension in self.extensions
                       if extension.name == name], singleton=True)

    def _run_epoch(self):
        if not self.status.get('epoch_started', False):
            try:
                self.log.status['received_first_batch'] = False
                self.epoch_iterator = (self.data_stream.
                                       get_epoch_iterator(as_dict=True))
            except StopIteration:
                return False
            self.status['epoch_started'] = True
            self._run_extensions('before_epoch')
        with Timer('epoch', self.profile):
            while self._run_iteration():
                pass
        self.status['epoch_started'] = False
        self.status['epochs_done'] += 1
        # Log might not allow mutating objects, so use += instead of append
        self.status['_epoch_ends'] += [self.status['iterations_done']]
        self._run_extensions('after_epoch')
        self._check_finish_training('epoch')
        return True

    def _run_iteration(self):
        ministeps_made = 0 # disabled D learning
        self.false_generated.set_value(0.)
        self.false_dataset.set_value(0.)
        while ministeps_made < self.k:
            try:
                with Timer('read_data', self.profile):
                    batch = next(self.epoch_iterator)
            except StopIteration:
                if not self.log.status['received_first_batch']:
                    reraise_as(ValueError("epoch iterator yielded zero batches"))
                return False
            self.log.status['received_first_batch'] = True
            self._run_extensions('before_batch', batch)
            batch = batch['features']
            noise = np.random.rand(self.minibatches, self.noise_per_sample).astype(np.float32)
            generated_batch = self._generator(noise)[0]
            bound_batch = np.zeros((batch.shape[0] * 2, batch.shape[1]), dtype=np.float32)
            bound_batch[:self.minibatches, :] = generated_batch
            bound_batch[self.minibatches:, :] = batch
            np.save('generated.npy', generated_batch)
            bound_batch = {'features': bound_batch}
            # if self.first_batch is None:
            #     self.first_batch = bound_batch
            with Timer('train', self.profile):
                # self.algorithm_d.process_batch(self.first_batch)
                self.algorithm_d.process_batch(bound_batch)
            ministeps_made += 1
        
        false_generated_perc = self.false_generated.get_value() / (self.k * self.minibatches)
        false_dataset_perc = self.false_dataset.get_value() / (self.k * self.minibatches)
        self.log[self.status['iterations_done'] + 1]['error_on_generated'] = false_generated_perc
        self.log[self.status['iterations_done'] + 1]['error_on_dataset'] = false_dataset_perc
        self.generator_errors.set_value(0.)
        noise = np.random.rand(self.minibatches, self.noise_per_sample).astype(np.float32)
        noise_batch = {'noise': noise}
        with Timer('train', self.profile):
            self.algorithm_g.process_batch(noise_batch)
        gen_errors = self.generator_errors.get_value() / self.minibatches
        self.log[self.status['iterations_done'] + 1]['generator_errors'] = gen_errors
        for o in self.observables:
            self.log[self.status['iterations_done'] + 1][o.name] = o.get_value()
        self.status['iterations_done'] += 1
        self._run_extensions('after_batch', bound_batch)
        self._check_finish_training('batch')
        return True

    def _run_extensions(self, method_name, *args):
        with Timer(method_name, self.profile):
            for extension in self.extensions:
                with Timer(type(extension).__name__, self.profile):
                    extension.dispatch(CallbackName(method_name), *args)

    def _check_finish_training(self, level):
        """Checks whether the current training should be terminated.

        Parameters
        ----------
        level : {'epoch', 'batch'}
            The level at which this check was performed. In some cases, we
            only want to quit after completing the remained of the epoch.

        """
        # In case when keyboard interrupt is handled right at the end of
        # the iteration the corresponding log record can be found only in
        # the previous row.
        if (self.log.current_row.get('training_finish_requested', False) or
                self.status.get('batch_interrupt_received', False)):
            raise TrainingFinish
        if (level == 'epoch' and
                self.status.get('epoch_interrupt_received', False)):
            raise TrainingFinish

    def _handle_epoch_interrupt(self, signal_number, frame):
        # Try to complete the current epoch if user presses CTRL + C
        logger.warning('Received epoch interrupt signal.' +
                       epoch_interrupt_message)
        signal.signal(signal.SIGINT, self._handle_batch_interrupt)
        self.log.current_row['epoch_interrupt_received'] = True
        # Add a record to the status. Unlike the log record it will be
        # easy to access at later iterations.
        self.status['epoch_interrupt_received'] = True

    def _handle_batch_interrupt(self, signal_number, frame):
        # After 2nd CTRL + C or SIGTERM signal (from cluster) finish batch
        self._restore_signal_handlers()
        logger.warning('Received batch interrupt signal.' +
                       batch_interrupt_message)
        self.log.current_row['batch_interrupt_received'] = True
        # Add a record to the status. Unlike the log record it will be
        # easy to access at later iterations.
        self.status['batch_interrupt_received'] = True

    def _restore_signal_handlers(self):
        signal.signal(signal.SIGINT, self.original_sigint_handler)
        signal.signal(signal.SIGTERM, self.original_sigterm_handler)



features = tensor.matrix('features')
noise = tensor.matrix('noise')


# g = MLP(activations=[Logistic()], dims=[100, 784])
# d = MLP(activations=[Identity()], dims=[784, 1])
g = MLP(activations=[Identity(), Identity(), Identity(), Identity(), Rectifier()], dims=[100, 2400, 2400, 2400, 2400, 784])
d = MLP(activations=[Tanh(), Tanh(), Identity()], dims=[784, 1200, 1200, 1])

generated_samples = g.apply(noise)
discriminated_features = d.apply(features)
discriminated_samples = d.apply(generated_samples)

generator_cg = ComputationGraph(generated_samples)
discriminator_cg = ComputationGraph(discriminated_features)
dsamples_cg = ComputationGraph(discriminated_samples)
generator_parameters = generator_cg.parameters

m = 100
b_size = discriminated_features.shape[0] / 2
cost_generator = tensor.sum(tensor.log(1 + tensor.exp(-discriminated_samples))) / discriminated_samples.shape[0].astype('float32')
cost_discriminator = (tensor.sum(discriminated_features[:b_size]) + tensor.sum(tensor.log(1 + tensor.exp(-discriminated_features)))) / b_size.astype('float32')

g.weights_init = IsotropicGaussian(0.05)
d.weights_init = IsotropicGaussian(0.005)
g.biases_init = d.biases_init = Constant(0)

g.initialize()
d.initialize()

for param in generator_cg.parameters:
    param.name += '_g'

for param in discriminator_cg.parameters:
    param.name += '_d'

both = list(set(dsamples_cg.parameters) & set(generator_cg.parameters))
indices = []
for (i, par) in enumerate(dsamples_cg.parameters):
    if par in generator_cg.parameters:
        indices.append(i)

good_params = [dsamples_cg.parameters[i] for i in indices]
print 'tests'
for param in dsamples_cg.parameters:
    print param.name

discriminator_descent = GradientDescent(cost=cost_discriminator, 
                                        parameters=discriminator_cg.parameters,
                                        step_rule=RMSProp(learning_rate=0.01, decay_rate=0.97))
print filter(lambda x: x.name[-2:] == '_g', dsamples_cg.parameters)
generator_descent = GradientDescent(cost=cost_generator, 
                                    parameters=filter(lambda x: x.name[-2:] == '_g', 
                                                      dsamples_cg.parameters),
                                    # parameters=good_params,
                                    # parameters=dsamples_cg.parameters,
                                    step_rule=RMSProp(learning_rate=1., decay_rate=0.97))

generator_descent.total_step_norm.name = 'generator_total_step_norm'
generator_descent.total_gradient_norm.name = 'generator_total_gradient_norm'
discriminator_descent.total_step_norm.name = 'discriminator_total_step_norm'
discriminator_descent.total_gradient_norm.name = 'discriminator_total_gradient_norm'
from fuel.datasets import MNIST
mnist = MNIST(("train",))

from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Flatten

data_stream = Flatten(
                DataStream.default_stream(mnist, 
                                          iteration_scheme=SequentialScheme(
                                              mnist.num_examples, 
                                              batch_size=m)))

# uwaga: dyskryminator bierze 2m probek, pierwsze m to generowane, kolejne m to z danych

# observables.append(cost_discriminator)

generator_cost = theano.shared(value=np.array(0., dtype=np.float32), name='g_cost')
discriminator_cost = theano.shared(value=np.array(0., dtype=np.float32), name='d_cost')

generator_step_norm = theano.shared(value=np.array(0., dtype=np.float32), name='g_step_norm')
generator_grad_norm = theano.shared(value=np.array(0., dtype=np.float32), name='g_grad_norm')
discriminator_step_norm = theano.shared(value=np.array(0., dtype=np.float32), name='d_step_norm')
discriminator_grad_norm = theano.shared(value=np.array(0., dtype=np.float32), name='d_grad_norm')

discriminator_descent.add_updates([
    (discriminator_cost, ComputationGraph(cost_discriminator).outputs[0]),
    (discriminator_step_norm, discriminator_descent.total_step_norm),
    (discriminator_grad_norm, discriminator_descent.total_gradient_norm)])

generator_descent.add_updates([
    (generator_cost, ComputationGraph(cost_generator).outputs[0]),
    (generator_step_norm, generator_descent.total_step_norm),
    (generator_grad_norm, generator_descent.total_gradient_norm)])

observables = []
observables.append(generator_cost)
observables.append(discriminator_cost)
observables.append(generator_step_norm)
observables.append(generator_grad_norm)
observables.append(discriminator_step_norm)
observables.append(discriminator_grad_norm)

g_out = shared_like(generator_cg.outputs[0], 
                    name='generator_' + generator_cg.outputs[0].name)
    
d_out = shared_like(discriminator_cg.outputs[0], 
                    name='discriminator_' + discriminator_cg.outputs[0].name)

false_generated = theano.shared(value=np.array(0., dtype=np.float32),
                                name='false_generated')

false_dataset = theano.shared(np.array(0., dtype='float32'),
                              name='false_dataset')

gen_errors = theano.shared(np.array(0., dtype='float32'),
                           name='generator_errors')

discriminator_descent.add_updates([(false_generated, false_generated +
                                    (discriminator_cg.outputs[0] > 0.5)[:m].sum().astype('float32')),
                                   (false_dataset, false_dataset + 
                                    (discriminator_cg.outputs[0] < 0.5)[m:].sum().astype('float32'))])
generator_descent.add_updates([(gen_errors, gen_errors + (ComputationGraph(discriminated_samples).outputs[0] > 0.5).sum().astype('float32'))])


extensions = []
extensions.append(Timing(after_batch=True))
extensions.append(Checkpoint('gan.thn', every_n_batches=10000, 
                             use_cpickle=True, save_separately=['log']))
extensions.append(Printing(every_n_batches=1))


main_loop = GANMainLoop(algorithm_g=generator_descent,
                        g_out=g_out,
                        algorithm_d=discriminator_descent,
                        d_out=d_out,
                        false_generated=false_generated,
                        false_dataset=false_dataset,
                        generator_errors=gen_errors,
                        data_stream=data_stream,
                        generator=generator_cg.get_theano_function(),
                        discriminator=discriminator_cg.get_theano_function(),
                        k=1,
                        noise_per_sample=100,
                        minibatches=m,
                        extensions=extensions,
                        observables=observables)

main_loop.run()
