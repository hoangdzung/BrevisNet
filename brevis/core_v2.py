
import os
import numpy as np
from tensorflow.python.util import nest
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# import the necessary packages
import numpy as np
import warnings
# import the necessary packages
import brevis
from .branches import branch
from .dataset import prepare
from .evaluate import evaluate


#neptune remote ML monitoring 
# try:
#     from initNeptune import Neptune
#     warnings.warn("Logging module Neptune was found, Cloud Logging can be enabled, Check initNeptune.py to configure settings")
#     Nep = Neptune()
# except ImportError:    
warnings.warn("Logging module Neptune was not found, Cloud Logging is not enabled, check https://docs.neptune.ai/getting-started/installation for more information on how to set this up")
Nep = None

from .utils import *

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)
root_logdir = os.path.join(os.curdir, "logs\\fit\\")


class BranchModel(tf.keras.Model):
    """
    Branched model sub-class. Acts as a drop in replacement keras model class, with the additional functionality of adding branches to the model.
    """


    def __init__(self, inputs=None, outputs=None, name="", model=None, transfer=True, custom_objects=None):
        if custom_objects is None:
            custom_objects = {}
        ## add default custom objects to the custom objects dictionary, this saves having to define them everytime.
        # custom_objects = {**self.default_custom_objects,**custom_objects}
        if inputs  is None and model is None and name is not "":
            model = tf.keras.models.load_model(name,custom_objects=custom_objects)
            self.saveLocation = name
            super(BranchModel, self).__init__(inputs = model.inputs, outputs=model.outputs,name=model.name)            
        elif model is None:
            super(BranchModel, self).__init__(inputs = inputs, outputs=outputs,name=name)
        elif model is not None:
            super(BranchModel, self).__init__(inputs = model.inputs, outputs=model.outputs,name=name)

        self.transfer = transfer
        self.custom_objects = custom_objects
#         remap the depths of the layers to match the desired layout for branching
#         self._map_graph_network(self.inputs,self.outputs, True)
        self.branch_active = False

        
#     def evaluate_branch_thresholds(self, test_ds, min_accuracy=None, min_throughput=None ,threshold_fn = evaluate.threshold_fn, stopping_point = None):
#         """
#         returns a list of thresholds for each branch endpoint
#         inputs:
#             test_ds: test dataset to evaluate the thresholds on
#             min_accuracy: minimum accuracy to set the threshold for
#             min_throughput: minimum throughput to set the threshold for
#             warning: function is not guaranteed to return a threshold that satisfies the min_accuracy and min_throughput if it does not exist.
        
#         returns:
#             list of thresholds for each branch endpoint


#         TODO implement the min accuracy and throughput functionality.
#         """
#         thresholds = {}
#         for layers in self.outputs:
#             thresholds.setdefault(layers.name,0)

#         branch_predictions = evaluate.collectEvidence_branches(self, test_ds, evidence=True,stopping_point=stopping_point)

#         for i, Predictions in enumerate(branch_predictions):
#             thresholds[self.outputs[i].name] = threshold_fn(Predictions)

#         return thresholds

#     def set_branch_thresholds(self, thresholds):
#         """
#         This function sets the thresholds for each branch endpoint layer within the branch model.
#         inputs: 
#             thresholds: dictionary of thresholds in the form of {layer_name:threshold}
#         """
#         for layer in self.layers:
#             if issubclass(type(layer), branch.BranchEndpoint) or issubclass(type(layer), tf.keras.Model):
#                 for key in thresholds.keys():
#                     if layer.name in key or f'{layer.name}/' in key:
#                         layer.threshold = thresholds[key]
#                         print(layer.name, "set to: ", layer.threshold)
#         return None
               
        
        
#         # return network_nodes, nodes_by_depth, layers, layers_by_depth
#     def _run_internal_graph(self, inputs, training=None, mask=None):
#         """custom version of _run_internal_graph
#             used to allow for interuption of the graph by an internal layer if conditions are met.
#         Computes output tensors for new inputs.

#         # Note:
#             - Can be run on non-Keras tensors.

#         Args:
#             inputs: Tensor or nested structure of Tensors.
#             training: Boolean learning phase.
#             mask: (Optional) Tensor or nested structure of Tensors.

#         Returns:
#             output_tensors
#         """
#         # print("_run_internal_graph --custom")
#         # print("branches enabled", self.branch_active)
#         inputs = self._flatten_to_reference_inputs(inputs)
#         if mask is None:
#             masks = [None] * len(inputs)
#         else:
#             masks = self._flatten_to_reference_inputs(mask)
#         for input_t, mask in zip(inputs, masks):
#             input_t._keras_mask = mask

#         # Dictionary mapping reference tensors to computed tensors.
#         tensor_dict = {}
#         tensor_usage_count = self._tensor_usage_count
#         for x, y in zip(self.inputs, inputs):
#             y = self._conform_to_reference_input(y, ref_input=x)
#             x_id = str(id(x))
#             tensor_dict[x_id] = [y] * tensor_usage_count[x_id]

#         nodes_by_depth = self._nodes_by_depth
#         depth_keys = list(nodes_by_depth.keys())
#         depth_keys.sort(reverse=True)
    
#         for depth in depth_keys:
#             nodes = nodes_by_depth[depth]
#             for node in nodes:
#                 # print(node.layer.name)
#                 if node.is_input:
#                     continue  # Input tensors already exist.

#                 if any(t_id not in tensor_dict for t_id in node.flat_input_ids):
#                     continue  # Node is not computable, try skipping.

#                 args, kwargs = node.map_arguments(tensor_dict)
#                 outputs = node.layer(*args, **kwargs)
#                 # Update tensor_dict.
#                 for x_id, y in zip(node.flat_output_ids, nest.flatten(outputs)):
#                     tensor_dict[x_id] = [y] * tensor_usage_count[x_id]
                
#                 ## check if branch exiting is turned on and if current layer is a potential exit.
#                 # print(node.layer.name, hasattr(node.layer, 'branch_exit'))
#                 if not training:
#                     if self.branch_active == True and hasattr(node.layer, 'branch_exit'):  
#                         ## check if the confidence of output of the layer is equal to or above the threshold hyperparameter
#                         # print("threshold: ", node.layer.threshold, "evidence: ", tf.reduce_sum(node.layer.evidence(outputs)))
#                         if node.layer.branch_exit and (tf.reduce_sum(node.layer.evidence(outputs)) >= node.layer.confidence_threshold): ##check if current layer's exit is active
#                             # print("branch exit activated")
#                             output_tensors = []
#                             for x_id, y in zip(node.flat_output_ids, nest.flatten(outputs)):
#                                 for x in self.outputs:
#                                     output_id = str(id(x))  
#                                     if output_id == x_id:
#                                         output_tensors.append(tensor_dict[x_id])
#                                     else:
#                                         # print(tensor_dict[x_id][0].shape)
#                                         output_tensors.append(tf.zeros(tensor_dict[x_id][0].shape))
#                                     # x_id_output = str(id(x))
#                                     # assert x_id in tensor_dict, 'Could not compute output ' + str(x)
#                                     # output_tensors.append(tensor_dict[x_id])

#                             return nest.pack_sequence_as(self._nested_outputs, output_tensors)
#         output_tensors = []
#         for x in self.outputs:
#             x_id = str(id(x))
#             assert x_id in tensor_dict, 'Could not compute output ' + str(x)
#             output_tensors.append(tensor_dict[x_id].pop())

#         return nest.pack_sequence_as(self._nested_outputs, output_tensors)

    def add_branches(self,branchName, branchPoints=[], exact = True, target_input = False, compact = False, loop=True,num_outputs=10):
        if len(branchPoints) == 0:
            return
        # ["max_pooling2d","max_pooling2d_1","dense"]
        # branch.newBranch_flatten
        if loop:
            self = branch.add_loop(self,branchName, branchPoints,exact=exact, target_input = target_input, compact = compact,num_outputs=num_outputs)
        else:
            self = branch.add(self,branchName,branchPoints, exact=exact, target_input = target_input, compact = compact,num_outputs=num_outputs)
#         print("branch added", newModel)
#         self.__dict__.update(newModel.__dict__)
        print(self.summary())
        return self
    
    def add_distill(self,teacher_output, branchName, branchPoints=[], exact = True,  loop=True,num_outputs=10):
        if len(branchPoints) == 0:
            return
        # ["max_pooling2d","max_pooling2d_1","dense"]
        # branch.newBranch_flatten
        newModel = branch.add_distil(self,teacher_output, None, branchName, branchPoints,exact=exact)
        print("branch added", newModel)
        self.__dict__.update(newModel.__dict__)

        # self.set_branchExits(True)
        # print(self)
        # self.summary()
        ### remap the graph of layers for the model
        # self._map_graph_network(self.inputs,self.outputs, True)
        return self

    def add_targets(self, num_outputs=10):
        outputs = list(self.outputs)
        inputs = []
        ready = False
        targets = None
        for i in self.inputs:
            if i.name == "targets":
                ready = True
            inputs.append(i)
        print("targets already present? ", ready)
        if not ready:
            print("added targets")
            targets = keras.Input(shape=(num_outputs, ), name="targets")
            inputs.append(targets)
        else:
            targets = self.get_layer('targets').output
        new_model = brevis.BranchModel(inputs=inputs, outputs=outputs, name=self.name, transfer=self.transfer, custom_objects=self.custom_objects)

        self.__dict__.update(new_model.__dict__)
        return self

#     def compile(self, loss, optimizer, metrics=['accuracy'], run_eagerly=True, **kwargs):
#         """ compile the model with custom options, either ones provided here or ones already set"""
#         super().compile(loss=loss, optimizer=optimizer, metrics=metrics, **kwargs)

    def setFrozenTraining(self,Frozen):
        """ sets the trainable status of all main path layers in the model"""
        if Frozen == True: 
            print("Freezing Main Layers and setting branch layers training to true")
            for i in range(len(self.layers)):
                if "branch" in self.layers[i].name:
                    self.layers[i].trainable = True
                else: 
                    self.layers[i].trainable = False               
        else:
            print("Setting Main Layers  and branch layers training to true")
            for i in range(len(self.layers)):
                self.layers[i].trainable = True
                # print("setting ",self.layers[i].name," training to true")
