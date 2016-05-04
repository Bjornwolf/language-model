import sys
import yaml
from data import build_datasets
from model import build_model, build_algorithm
from monitor import build_extensions
from blocks.main_loop import GANMainLoop
from blocks.model import Model

import numpy as np
from theano import tensor
from blocks.graph import ComputationGraph
from blocks.bricks import MLP, Rectifier, Softmax
from blocks.initialization import IsotropicGaussian, Constant
from blocks.algorithms import GradientDescent, Scale
from blocks.extensions import Printing, Timing
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.utils import shared_like

x = tensor.matrix('features')
z = tensor.matrix('noise')

m = 10

g = MLP(activations=[Rectifier(), Rectifier()], dims=[10, 500, 784])
d = MLP(activations=[Rectifier(), Softmax()], dims=[784, 100, 2])

generator = g.apply(z)
discriminator = d.apply(x)
discriminator2 = d.apply(generator)

generator_cg = ComputationGraph(generator)
discriminator_cg = ComputationGraph(discriminator)
generator_parameters = generator_cg.parameters

cost_discriminator = (np.log(discriminator[:m, 1]).sum() + np.log(discriminator[m:, 0]).sum()) * -1/m
cost_generator = np.log(discriminator2[:, 0]).sum() * -1/m

g.weights_init = d.weights_init = IsotropicGaussian(0.01)
g.biases_init = d.biases_init = Constant(0)

g.initialize()
d.initialize()

discriminator_descent = GradientDescent(cost=cost_discriminator, 
                                        parameters=discriminator_cg.parameters,
                                        step_rule=Scale(0.01))
generator_descent = GradientDescent(cost=cost_generator, 
                                    parameters=generator_cg.parameters, 
                                    step_rule=Scale(0.01))

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

observables = []

g_out = shared_like(generator_cg.outputs[0], 
                    name='generator_' + generator_cg.outputs[0].name)
    
d_out = shared_like(discriminator_cg.outputs[0], 
                    name='discriminator_' + discriminator_cg.outputs[0].name)

g_obs = []
g_obs.append(shared_like(generator_descent.total_step_norm,
                         name=generator_descent.total_step_norm.name))
g_obs.append(shared_like(generator_descent.total_gradient_norm,
                         name=generator_descent.total_gradient_norm.name))
print g_obs
 
d_obs = []
d_obs.append(shared_like(discriminator_descent.total_step_norm,
                         name=discriminator_descent.total_step_norm.name))
d_obs.append(shared_like(discriminator_descent.total_gradient_norm,
                         name=discriminator_descent.total_gradient_norm.name))
print d_obs



# generator_descent.add_updates([g_out] + g_obs)
discriminator_descent.add_updates([(d_out, discriminator_cg.outputs[0])])
false_generated = tensor.scalar('false_generated')
false_dataset = tensor.scalar('false_dataset')
observables = [false_generated, false_dataset]

print observables

extensions = []
extensions.append(Timing(after_batch=True))
extensions.append(TrainingDataMonitoring(observables, after_batch=True))
# extensions.append(TrainingDataMonitoring(observables, prefix='average', 
#                                          every_n_batches=2000))
extensions.append(Checkpoint('gan.thn', every_n_batches=10000, 
                             use_cpickle=True))
extensions.append(Printing(every_n_batches=500))

main_loop = GANMainLoop(algorithm_g=generator_descent,
                        g_out=g_out,
                        algorithm_d=discriminator_descent,
                        d_out=d_out,
                        false_generated=false_generated,
                        false_dataset=false_dataset,
                        data_stream=data_stream,
                        generator=generator_cg.get_theano_function(),
                        discriminator=discriminator_cg.get_theano_function(),
                        k=1,
                        minibatches=m,
                        extensions=extensions)

main_loop.run()
