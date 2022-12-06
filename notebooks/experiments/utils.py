import tensorflow as tf 
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from keras_flops import get_flops 

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, plot_precision_recall_curve
from numpy import sqrt, argmax
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from tqdm import tqdm 
import pandas as pd 
from openpyxl import Workbook

import sys
sys.path.append("../..") # Adds higher directory to python modules path.
import brevis
from brevis import branches
from brevis import evaluate

# class lambda_update(tf.keras.callbacks.Callback):
#     def __init__(self, annealing_max,  lambda_t=0, max_t=1):
#         self.start_val = tf.Variable(initial_value=lambda_t, dtype=tf.float32) 
        
#         self.lambda_t = tf.Variable(initial_value=lambda_t, dtype=tf.float32) #updates each epoch
#         self.max_t = tf.Variable(initial_value=max_t, dtype=tf.float32)
#         self.annealing_max = tf.Variable(initial_value=annealing_max, dtype=tf.float32)

#     def on_epoch_begin(self, epoch, logs={}): #needs to be on begin, otherwise the epoch update won't increase the value from 0 to 0.1 till the 3rd epoch...
#         val = tf.reduce_min([self.max_t, tf.cast(epoch+self.start_val, tf.dtypes.float32) / tf.cast(self.annealing_max, tf.dtypes.float32)])
#         tf.print("annealing coef updated to:", val)
#         self.lambda_t.assign(val)

# class growth_update(tf.keras.callbacks.Callback):
#     ''' callback update method that checks the performance of the model against the validation set to decide if the annealing coef should be increased.
#         provides greater control of the additional loss elements by updating their hyperparameters inteligently, rather then with only a preset schedule.
        
#     '''
#     def __init__(self, annealing_max,  lambda_t=0, max_t=1, starting_epoch = 0):
#         self.starting_epoch = starting_epoch
#         self.start_val = tf.Variable(initial_value=lambda_t, dtype=tf.float32) 
        
#         self.step = tf.Variable(initial_value = 0,dtype=tf.float32)
#         self.lambda_t = tf.Variable(initial_value=lambda_t, dtype=tf.float32) #updates each epoch
#         self.max_t = tf.Variable(initial_value=max_t, dtype=tf.float32)
#         self.annealing_max = tf.Variable(initial_value=annealing_max, dtype=tf.float32)
#         self.training = tf.Variable(initial_value=False, dtype=tf.bool)              
#         self.past_val_acc= tf.Variable(initial_value =0, dtype=tf.float32)
#         self.val_acc= tf.Variable(initial_value =0, dtype=tf.float32)
        
#     def on_training_begin(self, logs={}):
#         ''' indicate that training has begun, so val growth is an option.
#         '''
#         tf.print("training commenced, validation growth enabled")
#         self.training.assign(True)

#     def on_epoch_begin(self, epoch, logs={}): #needs to be on begin, otherwise the epoch update won't increase the value from 0 to 0.1 till the 3rd epoch...
#         val = self.lambda_t
#         if epoch >= self.starting_epoch:
#             if self.val_acc >= self.past_val_acc:
#                 val = tf.reduce_min([self.max_t, tf.cast((self.step - self.starting_epoch) +self.start_val , tf.dtypes.float32) / tf.cast(self.annealing_max, tf.dtypes.float32)])
#                 tf.print("annealing coef updated to:", val)
#                 self.lambda_t.assign(val)
#                 self.past_val_acc.assign(self.val_acc)
#                 self.step.assign(self.step + 1)
#             else:
#                 tf.print("val acc did not improve from {}, annealing coef not updated, remains at:{}".format(self.past_val_acc.numpy(), val.numpy()))

#     def on_test_end(self, logs=None):
#         """ if training, save the performance results
#         """
#         print("This is the logs", logs)
# #         self.val_acc.assign(logs.get('branch_exit_accuracy')+logs.get('branch_exit_1_accuracy'))

class lambda_update(tf.keras.callbacks.Callback):
    def __init__(self, annealing_max,  lambda_t=0, max_t=1, starting_epoch = 0):
        self.starting_epoch = starting_epoch
        self.start_val = tf.Variable(initial_value=lambda_t, dtype=tf.float32) 
        
        self.lambda_t = tf.Variable(initial_value=lambda_t, dtype=tf.float32) #updates each epoch
        self.max_t = tf.Variable(initial_value=max_t, dtype=tf.float32)
        self.annealing_max = tf.Variable(initial_value=annealing_max, dtype=tf.float32)

    def on_epoch_begin(self, epoch, logs={}): #needs to be on begin, otherwise the epoch update won't increase the value from 0 to 0.1 till the 3rd epoch...
        val = 0
        if epoch >= self.starting_epoch:
            val = tf.reduce_min([self.max_t, tf.cast((epoch - self.starting_epoch) +self.start_val , tf.dtypes.float32) / tf.cast(self.annealing_max, tf.dtypes.float32)])
        tf.print("annealing coef updated to:", val)
        self.lambda_t.assign(val)
        
class growth_update(lambda_update):
    ''' callback update method that checks the performance of the model against the validation set to decide if the annealing coef should be increased.
        provides greater control of the additional loss elements by updating their hyperparameters inteligently, rather then with only a preset schedule.
        
    '''
    def __init__(self, annealing_rate, start_t=0, max_t=1, starting_epoch = 0, branch_names=["branch_exit_accuracy","branch_exit_1_accuracy"]):
        self.starting_epoch = starting_epoch
        self.start_value = tf.Variable(initial_value=start_t, dtype=tf.float32) 
        self.branch_names= branch_names
        self.step = tf.Variable(initial_value = 0,dtype=tf.float32)
        self.lambda_t = tf.Variable(initial_value=start_t, dtype=tf.float32) #updates each epoch
        self.max_t = tf.Variable(initial_value=max_t, dtype=tf.float32)
        self.annealing_max = tf.Variable(initial_value=annealing_rate, dtype=tf.float32)
        self.training = tf.Variable(initial_value=False, dtype=tf.bool)              
        self.past_val_acc= tf.Variable(initial_value =0, dtype=tf.float32)
        self.val_acc= tf.Variable(initial_value =0, dtype=tf.float32)
        
    def on_training_begin(self, logs={}):
        ''' indicate that training has begun, so val growth is an option.
        '''
        tf.print("training commenced, validation growth enabled")
        self.training.assign(True)
#     def on_training_end(self, logs={}):
#         ''' indicate that training has ended, so turn off val growth. Not sure if this is actually needed...
#         '''
#         tf.print("training commenced, validation growth enabled")
#         self.training.assign(False)
    def on_epoch_begin(self, epoch, logs={}): #needs to be on begin, otherwise the epoch update won't increase the value from 0 to 0.1 till the 3rd epoch...
        val = self.lambda_t
        if epoch >= self.starting_epoch-1:
            tf.print(self.step)
            if self.val_acc >= self.past_val_acc:
                self.step.assign(self.step + 1)
                val = tf.reduce_min([self.max_t, tf.cast((self.step - self.starting_epoch) +self.start_value , tf.dtypes.float32) / tf.cast(self.annealing_max, tf.dtypes.float32)])
                tf.print("annealing coef updated to:", val)
                self.lambda_t.assign(val)
                self.past_val_acc.assign(self.val_acc)
                # self.step.assign(self.step + 1)
            else:
                tf.print("val acc did not improve from {}, annealing coef not updated, remains at:{}".format(self.past_val_acc.numpy(), val.numpy()))
        else:
            self.step.assign(self.step + 1)
            tf.print("annealing coef will start on epoch:", self.starting_epoch)
     # tf.print("past val acc =", self.past_val_acc)
        # self.past_val_acc.assign(self.val_acc)
        
    def on_test_end(self, logs=None):
        """ if training, save the performance results
        """
        results = 0
        for name in self.branch_names:               
            results += logs.get(name)
        self.val_acc.assign(results)  
        
def exp_evidence(logits): 
    return tf.exp(tf.clip_by_value(logits/10,-10,10))

def KL(alpha,K):
    # print("K:",K)
    beta=tf.constant(np.ones((1,K)),dtype=tf.float32)
    S_alpha = tf.reduce_sum(alpha,axis=1,keepdims=True)
    S_beta = tf.reduce_sum(beta,axis=1,keepdims=True)
    lnB = tf.compat.v1.lgamma(S_alpha) - tf.reduce_sum(tf.compat.v1.lgamma(alpha),axis=1,keepdims=True)
    lnB_uni = tf.reduce_sum(tf.compat.v1.lgamma(beta),axis=1,keepdims=True) - tf.compat.v1.lgamma(S_beta)
    dg0 = tf.compat.v1.digamma(S_alpha)
    dg1 = tf.compat.v1.digamma(alpha)
    kl = tf.reduce_sum((alpha - beta)*(dg1-dg0),axis=1,keepdims=True) + lnB + lnB_uni
    return kl

def _KL(alphas, target_alphas,  precision=None, target_precision=None, epsilon=1e-8):
    '''
    New KL divergence function. 
    '''
    target_alphas = tf.cast(target_alphas,tf.float32)
    alphas = tf.cast(alphas,tf.float32)
    if not precision:
        precision = tf.reduce_sum(alphas, axis=1, keepdims=True)
    if not target_precision:
        target_precision = tf.reduce_sum(target_alphas, axis=1, keepdims=True)
    precision = tf.cast(precision,tf.float32)
    target_precision = tf.cast(target_precision,tf.float32)
    
    precision_term = tf.compat.v1.lgamma(target_precision) - tf.compat.v1.lgamma(precision)
    alphas_term = tf.reduce_sum(
        tf.compat.v1.lgamma(alphas + epsilon)
        - tf.compat.v1.lgamma(target_alphas + epsilon)
        + (target_alphas - alphas)
        * (
            tf.compat.v1.digamma(target_alphas + epsilon)
            - tf.compat.v1.digamma(target_precision + epsilon)
        ),
        axis=1,
        keepdims=True,
    )
    cost = tf.squeeze(precision_term + alphas_term)
    return cost

def reverse_kl(alphas, target_alphas,  precision=None, target_precision=None, epsilon=1e-8):
    return _KL(target_alphas,alphas, precision=None, target_precision=None, epsilon=1e-8)

def DirichletKLLoss(labels, logits, reverse=True):
    # alpha = tf.exp(logits)
    alpha = tf.exp(tf.clip_by_value(logits/10,-10,10))
    target_concentration = tf.reduce_sum(alpha,axis=1,keepdims=True)
    target_alphas = (tf.ones_like(alpha) + (target_concentration * labels))
    alpha = alpha + 1
    if reverse:
        cost = reverse_kl(alpha, target_alphas)
    else:
        cost = _KL(alpha, target_alphas)
    if tf.math.is_nan(tf.reduce_sum(cost)):
        tf.print("logits",logits, summarize=-1)
        tf.print("alpha",alpha, summarize=-1)
        tf.print("cost", cost, summarize=-1)
    return cost


def branch_conv2d(prevLayer, n_classes, depths, targets=None, teacher_sm = None, teaching_features=None):
    """ Add a new branch to a model connecting at the output of prevLayer. 
        NOTE: use the substring "branch" in all names for branch nodes. this is used as an identifier of the branching layers as opposed to the main branch layers for training
    """ 
    branchLayer = keras.layers.Conv2D(filters=depths[0], kernel_size=(1,1), strides=(1,1), activation='relu',name=tf.compat.v1.get_default_graph().unique_name("branch_conv2d"), input_shape=(prevLayer.shape))(prevLayer)
    branchLayer = keras.layers.BatchNormalization(name=tf.compat.v1.get_default_graph().unique_name("branch_batchnorm"))(branchLayer)  
    branchLayer = keras.layers.Conv2D(filters=depths[0], kernel_size=(1,1), strides=(1,1), activation='relu',name=tf.compat.v1.get_default_graph().unique_name("branch_conv2d"), input_shape=(branchLayer.shape))(branchLayer)
    branchLayer = keras.layers.BatchNormalization(name=tf.compat.v1.get_default_graph().unique_name("branch_batchnorm"))(branchLayer)  
    branchLayer = keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2),name=tf.compat.v1.get_default_graph().unique_name("branch_maxpool"))(branchLayer)
    # branchLayer = keras.layers.Dropout(0.2,name=tf.compat.v1.get_default_graph().unique_name("branch_dropout"))(branchLayer)
    branchLayer = keras.layers.Conv2D(filters=depths[1], kernel_size=(1,1), strides=(1,1), activation='relu',name=tf.compat.v1.get_default_graph().unique_name("branch_conv2d"), input_shape=(branchLayer.shape))(branchLayer)
    branchLayer = keras.layers.BatchNormalization(name=tf.compat.v1.get_default_graph().unique_name("branch_batchnorm"))(branchLayer)
    branchLayer = keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2),name=tf.compat.v1.get_default_graph().unique_name("branch_maxpool"))(branchLayer)
    # branchLayer = keras.layers.Dropout(0.2,name=tf.compat.v1.get_default_graph().unique_name("branch_dropout"))(branchLayer)
    branchLayer = layers.Flatten(name=tf.compat.v1.get_default_graph().unique_name("branch_flatten"))(branchLayer)
    branchLayer = layers.Dense(1024,activation='relu',name=tf.compat.v1.get_default_graph().unique_name("branch_dense"))(branchLayer)
    branchLayer = keras.layers.Dropout(0.2,name=tf.compat.v1.get_default_graph().unique_name("branch_dropout"))(branchLayer)
    branchLayer = layers.Dense(512,activation='relu',name=tf.compat.v1.get_default_graph().unique_name("branch_dense"))(branchLayer)
    output = keras.layers.Dense(n_classes, name=tf.compat.v1.get_default_graph().unique_name("branch_exit"))(branchLayer)
    # output = (layers.Softmax(name=tf.compat.v1.get_default_graph().unique_name("branch_softmax"))(output))
    return output

def Brevis_loss_final(lambda_callback: lambda_update, gamma=0.0001):
    ''' Loss function of Evidential Dirichlet Networks
        Expected Mean Square Error + KL divergence
    '''
    def custom_loss_function(p, logits):
        evidence = exp_evidence(logits)
        # evidence = tf.nn.softplus(logits)
        alpha = evidence + 1
        S = tf.reduce_sum(alpha,axis=1,keepdims=True) 
        E = alpha - 1
        m = alpha / S
        A = tf.reduce_sum((p-m)**2, axis=1, keepdims=True) 
        B = tf.reduce_sum(alpha*(S-alpha)/(S*S*(S+1)), axis=1, keepdims=True) 
        annealing_coef =  lambda_callback.lambda_t
        
        bool_mask = tf.cast(p,tf.bool)
        ic_mask = tf.cast(1 - p,tf.bool)
        ic_bool_mask = tf.cast(ic_mask,tf.bool)
        ic_alpha_masked = tf.cast(tf.ragged.boolean_mask(alpha, ic_bool_mask).to_tensor(),tf.float32)
        #### info reg
        _A = (ic_alpha_masked -1) ** 2
        B_1 = tf.math.polygamma(1.,ic_alpha_masked) 
        B_2 = tf.math.polygamma(1., tf.reduce_sum(ic_alpha_masked,axis=1,keepdims=True))
        _B = (B_1 - B_2)
        info_reg =  .5* tf.reduce_sum(_A * _B,axis=1)
        info_reg = annealing_coef * (info_reg * 2)
        
        # annealing_coef =  0.0001
        # alp = E*(1-p) + 1 
        # C =   annealing_coef * KL(alp,10)
        # C =   annealing_coef * DirichletKLLoss(p,logits, True)
        D = gamma * -tf.reduce_mean(tfp.distributions.Dirichlet(alpha).entropy())
        # tf.print((A + B),summarize=-1)
        # tf.print((info_reg + D),summarize=-1)
        
        # tf.print(((A + B) + info_reg + D).shape)
        return (A + B) + info_reg +  D # info_reg + D  #+ info_reg #+ C + D
    return custom_loss_function

# def auxLoss(lambda_callback: lambda_update):
#     def auxloss(p, logits):
#         evidence = exp_evidence(logits)
#             # evidence = tf.nn.softplus(logits)
#         alpha = evidence + 1
#         S = tf.reduce_sum(alpha,axis=1,keepdims=True) 
#         E = alpha - 1
#         m = alpha / S
#         A = tf.reduce_sum((p-m)**2, axis=1, keepdims=True) 
#         B = tf.reduce_sum(alpha*(S-alpha)/(S*S*(S+1)), axis=1, keepdims=True) 
#         annealing_coef =  lambda_callback.lambda_t

#         bool_mask = tf.cast(p,tf.bool)
#         ic_mask = tf.cast(1 - p,tf.bool)
#         ic_bool_mask = tf.cast(ic_mask,tf.bool)
#         ic_alpha_masked = tf.cast(tf.ragged.boolean_mask(alpha, ic_bool_mask).to_tensor(),tf.float32)
#         #### info reg
#         _A = (ic_alpha_masked -1) ** 2
#         B_1 = tf.math.polygamma(1.,ic_alpha_masked) 
#         B_2 = tf.math.polygamma(1., tf.reduce_sum(ic_alpha_masked,axis=1,keepdims=True))
#         _B = (B_1 - B_2)
#         info_reg =  .5* tf.reduce_sum(_A * _B,axis=1)
#         info_reg = annealing_coef * (info_reg * 2)
#         C =   annealing_coef * DirichletKLLoss(p,logits, True)
#         D = 0.0001 * -tf.reduce_mean(tfp.distributions.Dirichlet(alpha).entropy())
#         # return (A + B) + C #+ D #+ info_reg #+ C + D
#         # tf.print(C)
#         return tf.reduce_mean(info_reg+ D )
#     return auxloss     


def getPredictions_Energy(model, input_set, stopping_point=None,num_classes=10, is_ood=False):
    '''
        Function for collecting the model's predictions on a test set. 

        Returns a list of DataFrames for each exit of the model.    
    '''
    num_branches = len(model.outputs) # the number of output layers for the purpose of providing labels

    Uncert = [[] for _ in range(num_branches)] # DBU
    Results = [[] for _ in range(num_branches)]
    Labels = [[] for _ in range(num_branches)]
    Energy = [[] for _ in range(num_branches)] #brevis
    Entropy = [[] for _ in range(num_branches)] # branchy
    calibration = [[] for _ in range(num_branches)] # calbiration
    
    y = np.concatenate([y for x, y in input_set], axis=0)
    predictions = model.predict(input_set)[0]
        
    for branch_idx, outputs in enumerate(predictions):
        for i, prediction in tqdm(enumerate(outputs), desc="Processing branch {}".format(branch_idx + 1)):
            evidence = exp_evidence(prediction)
            alpha = evidence + 1
            S = sum(alpha)
            E = alpha - 1
            Mass = alpha / S
            u = num_classes / S
            Uncert[branch_idx].append(u.numpy().mean())
            Results[branch_idx].append(np.argmax(prediction))
            Labels[branch_idx].append(np.argmax(y[i]))
            Energy[branch_idx].append( -(logsumexp(np.array(prediction))))
            Entropy[branch_idx].append(brevis.utils.calcEntropy_Tensors2(tf.nn.softmax(prediction)).numpy())
            calibration[branch_idx].append(np.amax(tf.nn.softmax(prediction).numpy()))
            

    Outputs=[]
    for branch_idx in range(num_branches):

        df = pd.DataFrame({"x":Results[branch_idx],"y":Labels[branch_idx],
                           "uncert":Uncert[branch_idx],
                            "energy":Energy[branch_idx],
                            'entropy':Entropy[branch_idx],
                            'calibration':calibration[branch_idx],
                          })
        conditions = [df['x'] == df['y'],df['x'] != df['y']]
        choices = [1, 0]
        if is_ood:
            df['correct'] = 0
            df['outlier'] = 0
        else:
            df['correct'] = np.int32(np.select(conditions, choices, default=None))
            df['outlier'] = 1
            
        Outputs.append(df)
    return Outputs

def save_outputs(filename, outputs):
    wb = Workbook()
    ws = wb.active
    with pd.ExcelWriter(filename+'.xlsx', engine="openpyxl") as writer:
        writer.book = wb
        writer.sheets = dict((ws.title, ws) for ws in wb.worksheets)

        for i, df in enumerate(outputs):
            name = "exit{}".format(i+1)
            df.to_excel(writer, name)
        writer.save()
    wb.close()
    
def read_outputs(fileName):
    output_test = list(pd.read_excel(fileName+'.xlsx',['exit1','exit2','exit3']).values())
    return output_test

def calc_AUC(output_df,metrics=['energy'],plot=False, pos_label = 0):
    '''
    AUC calculation function for list of output dataframes
    returns a list of threshold for the gmean of each set of outputs.    
    '''
    lessThanMetrics = ["energy","uncert","entropy"]
    _thresholds = []
    y_test = np.int32(output_df['correct'])
    plots = []
    if type(metrics) is not list:
        metrics = [metrics]
        
    for metric in metrics:    
        print("metric", metric)
        lr_auc = roc_auc_score(y_test, output_df[metric])
        if metric in lessThanMetrics:
            pos_label = 0
        else:
            pos_label = 1
        fpr, tpr, thresholds = roc_curve(y_test, output_df[metric],pos_label=pos_label)
        gmeans = sqrt(tpr * (1-fpr))
        # print(gmeans)
        # locate the index of the largest g-mean
        ix = argmax(gmeans)
        threshold = thresholds[ix]
        print(metric," lr_auc",lr_auc, 'Best Threshold={}, G-Mean={}, TPR={}, FPR={}'.format(threshold, gmeans[ix],tpr[ix],fpr[ix]))
        _thresholds.append(threshold)
        # plot the roc curve for the model
        plots.append({"fpr":fpr,"tpr":tpr,"label":metric, "ix":ix})
        
    if plot:
        plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
        for plot in plots:
            ix = plot['ix']
            plt.plot(plot["fpr"], plot["tpr"],  label=plot['label'])

            plt.scatter(plot["fpr"][ix], plot["tpr"][ix], marker='o', color='black')
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(metric)
        plt.legend()
        # show the plot
        plt.show()
    return _thresholds

def prepare_threshold(threshold_type, metric, output_ID, Correct):
    if threshold_type == "mean":
        # _threshold = np.array(ID[metric]).mean()
        Correct = output_ID.loc[(output_ID["correct"] == True)]
        _threshold = np.array(Correct[metric]).mean()
    if threshold_type == "gmean":
        print("m",metric)
        AUC_thresholds = calc_AUC(output_ID, metrics=[metric], plot = False)
        _threshold = AUC_thresholds[0]
    if threshold_type == "PR_AUC":
        precision_, recall_, proba = precision_recall_curve(output_ID['correct'], output_ID[metric])
        _threshold = sorted(list(zip(np.abs(precision_ - recall_), proba)), key=lambda i: i[0], reverse=False)[0][1]

    return _threshold

def infer_result(ID, metrics=["energy"], threshold="gmean", flops=None):
    lessThanMetrics = ["energy","uncert","entropy"]
    if type(metrics) is not list:
        metrics = [metrics]
        
    for metric_idx, metric in enumerate(metrics):
        
        rollOver_ID_indices = pd.Index([])
        Exit_Name, Thresholds, Test_accuracy, Results, Input_ID = [], [], [], [], []
        Accepted_ID_list, Accepted_Ratio_list, Acceptance_correct, Accepted_Accuracy_list = [], [], [], []
        _ID = ID.copy()
        n_ID_correct = 0
        for branch_idx, output_ID in enumerate(_ID): 
            # test_accuracy on this branch
            Test_accuracy.append(len(output_ID.loc[(output_ID["correct"] == True)])/len(output_ID))

            # correct and incorrect on this branch
            Correct = output_ID.loc[(output_ID['correct'] == True)]
            Incorrect = output_ID.loc[(output_ID['correct'] == False)]
            _threshold = prepare_threshold(threshold, metric, output_ID, Correct)
            
            if len(rollOver_ID_indices)>0:
                output_ID = output_ID.iloc[rollOver_ID_indices]
                
            if branch_idx == len(_ID)-1 :
                Exit_Name.append("Main_exit")
                _threshold = "NA"
                ID_accepted = output_ID
            else:
                Exit_Name.append("exit_{}".format(branch_idx+1))
                if metric in lessThanMetrics: ## metrics that require less than metric
                    ID_accepted = output_ID.loc[(output_ID[metric] <= _threshold)] #TP
                    ID_rejected = output_ID.loc[(output_ID[metric] > _threshold)] #FN
          
                else: ### metrics that require greater than metric
                    ID_accepted = output_ID.loc[(output_ID[metric] >= _threshold)] #TP
                    ID_rejected = output_ID.loc[(output_ID[metric] < _threshold)] #FN
                    
                rollOver_ID_indices = ID_rejected.index
                
                rejected_correct = ID_rejected.loc[(ID_rejected["correct"] == True)]  #FN
                rejected_incorrect = ID_rejected.loc[(ID_rejected[metric] ==False)] #TN

            accepted_correct = ID_accepted.loc[(ID_accepted["correct"] == True )] #TP
            accepted_incorrect = ID_accepted.loc[(ID_accepted[metric] ==False)] #FP
            
            accepted_ID_acc = len(accepted_correct) / (len( ID_accepted))
            n_ID_correct += len(accepted_correct)               
                   
            Thresholds.append(_threshold)
            Input_ID.append(len(output_ID))
            Accepted_ID_list.append(len(ID_accepted))
            Acceptance_correct.append(len(accepted_correct))
            Accepted_Accuracy_list.append(accepted_ID_acc)

        df = pd.DataFrame({
            "Exit_Name":Exit_Name,
            "ID_Inputs":Input_ID, #ID goes through this branch
            "Test_Accuracy":Test_accuracy, #Test accuracy on this branch
            "Threshold":Thresholds,
            "Accepted ID":Accepted_ID_list, 
            "Accepted_Correct":Acceptance_correct,
            "Acceptance_Accuracy":Accepted_Accuracy_list,                                
            })
        
        with pd.option_context('expand_frame_repr', False):
            print (df)
        print("Overall accuracy ID", n_ID_correct/len(ID[0]))
        if flops is not None:
            flop = sum([flop * n_accepted/len(ID[0]) for flop, n_accepted in zip(flops, Accepted_ID_list) ])
            print(flop)

def infer_result_OOD(ID, OOD, metrics=["energy"], threshold="gmean", flops=None):
    lessThanMetrics = ["energy","uncert","entropy"]
    if type(metrics) is not list:
        metrics = [metrics]
        
    for metric_idx, metric in enumerate(metrics):
        print("metric: ", metric, "threshold: ",threshold)
        
        rollOver_ID_indices, rollOver_OOD_indices = pd.Index([]), pd.Index([])
        Exit_Name, Thresholds, Test_accuracy, Results, Input_ID, Input_OOD = [], [], [], [], [], []
        Accepted_ID_list, Accepted_OOD_list, Accepted_Ratio_list, Acceptance_correct, Accepted_Accuracy_list = [], [], [], [], []
        _ID, _OOD = ID.copy(), OOD.copy()
        n_ID_correct = 0
        for branch_idx, (output_ID, output_OOD) in enumerate(zip(_ID, _OOD)): 
            # test_accuracy on this branch
            Test_accuracy.append(len(output_ID.loc[(output_ID["correct"] == True)])/len(output_ID))

            # correct and incorrect on this branch
            Correct = output_ID.loc[(output_ID['correct'] == True)]
            Incorrect = output_ID.loc[(output_ID['correct'] == False)]
            _threshold = prepare_threshold(threshold, metric, output_ID, Correct)
            
            if len(rollOver_ID_indices)>0:
                output_ID = output_ID.iloc[rollOver_ID_indices]
            if len(rollOver_OOD_indices)>0:
                output_OOD = output_OOD.iloc[rollOver_OOD_indices]
                
            if branch_idx == len(_ID)-1 :
                Exit_Name.append("Main_exit")
                _threshold = "NA"
                OOD_accepted, ID_accepted = output_OOD, output_ID
                OOD_rejected = ID_rejected = rejected_correct = rejected_incorrect = None
            else:
                Exit_Name.append("exit_{}".format(branch_idx+1))

                if metric in lessThanMetrics: ## metrics that require less than metric
                    OOD_accepted = output_OOD.loc[(output_OOD[metric].tolist() <= _threshold)] #FP
                    OOD_rejected = output_OOD.loc[(output_OOD[metric].tolist() > _threshold)] #TN
                    ID_accepted = output_ID.loc[(output_ID[metric] <= _threshold)] #TP
                    ID_rejected = output_ID.loc[(output_ID[metric] > _threshold)] #FN
          
                else: ### metrics that require greater than metric
                    OOD_accepted = output_OOD.loc[(output_OOD[metric].tolist() >= _threshold)] #FP
                    OOD_rejected = output_OOD.loc[(output_OOD[metric].tolist() < _threshold)] #TN
                    ID_accepted = output_ID.loc[(output_ID[metric] >= _threshold)] #TP
                    ID_rejected = output_ID.loc[(output_ID[metric] < _threshold)] #FN
                    
                rollOver_ID_indices = ID_rejected.index
                rollOver_OOD_indices = OOD_rejected.index
                
                rejected_correct = ID_rejected.loc[(ID_rejected["correct"] == True)]  #FN
                rejected_incorrect = ID_rejected.loc[(ID_rejected[metric] ==False)] #TN
                
            accepted_correct = ID_accepted.loc[(ID_accepted["correct"] == True )] #TP
            accepted_incorrect = ID_accepted.loc[(ID_accepted[metric] ==False)] #FP
            
            accepted_ID_acc = len(accepted_correct) / (len( ID_accepted))
            overall_accepted_acc = len(accepted_correct) / (len( ID_accepted) + len(OOD_accepted))
            n_ID_correct += len(accepted_correct)           
                   
            Thresholds.append(_threshold)

            Input_ID.append(len(output_ID))
            Input_OOD.append(len(output_OOD))
            Accepted_ID_list.append(len(ID_accepted))
            Accepted_OOD_list.append(len(OOD_accepted))
            Accepted_Ratio_list.append(len(ID_accepted)/(len(ID_accepted) + len(OOD_accepted)))
            Acceptance_correct.append(len(accepted_correct))
            Accepted_Accuracy_list.append(overall_accepted_acc)
            
        df = pd.DataFrame({
            "Exit_Name":Exit_Name,
            "ID_Inputs":Input_ID, #ID goes through this branch
            "OOD_Inputs":Input_OOD, #OOD goes through this branch
            "Test_Accuracy":Test_accuracy, #Test accuracy on this branch
            "Threshold":Thresholds,
            "Accepted ID":Accepted_ID_list, 
            "Accepted OOD":Accepted_OOD_list,

            "Accepted_Correct":Acceptance_correct,
            "Accepted_ID_Ratio":Accepted_Ratio_list,
            "Acceptance_Accuracy":Accepted_Accuracy_list,                                
            })
        
        with pd.option_context('expand_frame_repr', False):
            print (df)
        print("Overall accuracy ID", n_ID_correct/len(ID[0]))
        if flops is not None:
            flop = sum([flop * (n_accepted_id + n_accepted_ood)/(len(ID[0]) + len(OOD[0])) for flop, n_accepted_id, n_accepted_ood in zip(flops, Accepted_ID_list, Accepted_OOD_list) ])
            print(flop)
    
    
def get_branched_flops(model, exits = None):
    if exits is None:
        return get_flops(model, batch_size=1)/ 10 ** 9
    
    n_exits = len(exits)
    flops = []
    for i in range(1, n_exits + 1):
        temp_model = tf.keras.models.Model(inputs=model.inputs, outputs=[model.get_layer(exit).get_output_at(0) for exit in exits[:i]] )
        flops.append(get_flops(temp_model, batch_size=1)/ 10 ** 9)
    return flops