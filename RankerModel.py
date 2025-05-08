import pickle
import numpy as np
import scipy as sc
#from molmod.molecules import Molecule
import tensorflow.compat.v1 as tf
import pandas as pd
import time
import os
import gc
import math as m
tf.disable_v2_behavior()

from BatchDataFunctions import GerarBatch
from BatchDataFunctions import GerarBatchInference

class ModelRanker:
   ##--------------------Auxiliary Functions ----------------------
   # ## activation function --> shifted soft plus
   def ssp(self,param_valor):
      return tf.math.softplus(param_valor) - tf.math.log(tf.cast(2, tf.dtypes.float64))
   
   def cutoff_distance(self, param_batch_dist, param_weight_dist_cutoff):
      val_cutoff = param_weight_dist_cutoff[0,0]
      final_result_0 = 1 - 6*((param_batch_dist/val_cutoff)**5) + 15*((param_batch_dist/val_cutoff)**4) - 10*((param_batch_dist/val_cutoff)**3)
      final_result = tf.nn.relu(final_result_0)

      return final_result
   
   def DistanceExpansion(self, param_batch_dist,param_Weights1_rbf_mu, param_Weights1_rbf_gamma):
      aux_dist_expand = tf.expand_dims(param_batch_dist, 3)
      ## RBF expansion calculation
      aux_dist_expand0 = (aux_dist_expand - param_Weights1_rbf_mu)**2
      rbf_expansion = tf.math.exp(-1.0*tf.multiply(aux_dist_expand0, param_Weights1_rbf_gamma))

      return rbf_expansion
   ##---------------------------------------------------------------

   def NMP(self, param_batch_features, param_batch_dist, param_batch_distance_expand, param_plc_bias_dist, param_plc_bias,
           param_dist_weights_1, param_dist_weights_1_bias,param_dist_weights_2, param_dist_weights_2_bias,
           param_agg_weights_1, param_agg_weights_1_bias, param_agg_weights_2, param_agg_weights_2_bias,
           param_weight_dist_cutoff):
      
      dist_step1 = self.ssp(tf.matmul(param_batch_distance_expand, param_dist_weights_1) + param_dist_weights_1_bias)
      dist_step2 = tf.matmul(dist_step1, param_dist_weights_2) + param_dist_weights_2_bias
      feat_dist_interaction = tf.multiply(dist_step2, tf.expand_dims(param_batch_features, axis=1))
      
      batch_dist_cutoff = self.cutoff_distance(param_batch_dist, param_weight_dist_cutoff)
      aux_dist_expand = tf.expand_dims(batch_dist_cutoff, 3)
      feat_dist_interaction_radius_cutoff = tf.multiply(feat_dist_interaction, aux_dist_expand)
      
      ## Removing self atoms interactions and removing padding atoms.
      feat_dist_interaction_radius_cutoff_2 = tf.multiply(feat_dist_interaction_radius_cutoff, param_plc_bias_dist)
      
      #Aggregating the atoms neighbors information
      v = tf.reduce_sum(feat_dist_interaction_radius_cutoff_2, axis=2)
      message_1 = self.ssp(tf.matmul(v, param_agg_weights_1) + param_agg_weights_1_bias)
      message_2 = tf.matmul(message_1, param_agg_weights_2) + param_agg_weights_2_bias
      message_3 = tf.multiply(message_2, param_plc_bias) ## removing the bias for the padding atoms.

      retorno = param_batch_features + message_3

      return retorno
   
   def AtomModelBase(self, predict_aggregation = 'sum'):
      embeddings = tf.matmul(self.plc_batch_features, self.Weights_embeddings)

      rbf_distance_expand = self.DistanceExpansion(self.plc_batch_dist, self.Weights1_rbf_mu, self.Weights1_rbf_gamma)

      interaction_1 = self.NMP(embeddings, self.plc_batch_dist, rbf_distance_expand, self.plc_batch_bias_dist, self.plc_batch_bias,
                     self.Weights1_NMP_01_distance, self.bias1_NMP_01_distance,
                     self.Weights2_NMP_01_distance, self.bias2_NMP_01_distance,
                     self.Weights1_NMP_01_aggregation, self.bias1_NMP_01_aggregation,
                     self.Weights2_NMP_01_aggregation, self.bias2_NMP_01_aggregation, self.Weight_dist_cutoff)
   
      interaction_2 = self.NMP(interaction_1, self.plc_batch_dist, rbf_distance_expand, self.plc_batch_bias_dist, self.plc_batch_bias,
                        self.Weights1_NMP_02_distance, self.bias1_NMP_02_distance,
                        self.Weights2_NMP_02_distance, self.bias2_NMP_02_distance,
                        self.Weights1_NMP_02_aggregation, self.bias1_NMP_02_aggregation,
                        self.Weights2_NMP_02_aggregation, self.bias2_NMP_02_aggregation, self.Weight_dist_cutoff)

      interaction_3 = self.NMP(interaction_2, self.plc_batch_dist, rbf_distance_expand, self.plc_batch_bias_dist, self.plc_batch_bias,
                        self.Weights1_NMP_03_distance, self.bias1_NMP_03_distance,
                        self.Weights2_NMP_03_distance, self.bias2_NMP_03_distance,
                        self.Weights1_NMP_03_aggregation, self.bias1_NMP_03_aggregation,
                        self.Weights2_NMP_03_aggregation, self.bias2_NMP_03_aggregation, self.Weight_dist_cutoff)
   
      dense_layer1_step_0 = self.ssp(tf.matmul(interaction_3, self.Weights1_Dense_01) + self.bias1_Dense_01)
      dense_layer1_step_1 = tf.matmul(dense_layer1_step_0, self.Weights2_Dense_01) + self.bias2_Dense_01
      dense_layer1 = tf.multiply(dense_layer1_step_1, self.plc_batch_bias) + interaction_3

      interaction_4 = self.NMP(dense_layer1, self.plc_batch_dist, rbf_distance_expand, self.plc_batch_bias_dist, self.plc_batch_bias,
                        self.Weights1_NMP_04_distance, self.bias1_NMP_04_distance,
                        self.Weights2_NMP_04_distance, self.bias2_NMP_04_distance,
                        self.Weights1_NMP_04_aggregation, self.bias1_NMP_04_aggregation,
                        self.Weights2_NMP_04_aggregation, self.bias2_NMP_04_aggregation, self.Weight_dist_cutoff)

      interaction_5 = self.NMP(interaction_4, self.plc_batch_dist, rbf_distance_expand, self.plc_batch_bias_dist, self.plc_batch_bias,
                        self.Weights1_NMP_05_distance, self.bias1_NMP_05_distance,
                        self.Weights2_NMP_05_distance, self.bias2_NMP_05_distance,
                        self.Weights1_NMP_05_aggregation, self.bias1_NMP_05_aggregation,
                        self.Weights2_NMP_05_aggregation, self.bias2_NMP_05_aggregation, self.Weight_dist_cutoff)
      
      interaction_6 = self.NMP(interaction_5, self.plc_batch_dist, rbf_distance_expand, self.plc_batch_bias_dist, self.plc_batch_bias,
                     self.Weights1_NMP_06_distance, self.bias1_NMP_06_distance,
                     self.Weights2_NMP_06_distance, self.bias2_NMP_06_distance,
                     self.Weights1_NMP_06_aggregation, self.bias1_NMP_06_aggregation,
                     self.Weights2_NMP_06_aggregation, self.bias2_NMP_06_aggregation, self.Weight_dist_cutoff)

      dense_layer2_step_0 = self.ssp(tf.matmul(interaction_6, self.Weights1_Dense_02) + self.bias1_Dense_02)
      dense_layer2_step_1 = tf.matmul(dense_layer2_step_0, self.Weights2_Dense_02) + self.bias2_Dense_02
      dense_layer2 = tf.multiply(dense_layer2_step_1, self.plc_batch_bias) + interaction_6

      gate_layer = tf.multiply(embeddings, self.Weights_Gate_Embeddings) + tf.multiply(dense_layer1, self.Weights_Gate_Layer_01) + tf.multiply(dense_layer2, self.Weights_Gate_Layer_02)
      prediction_0 = self.ssp(tf.matmul(gate_layer, self.Weights1_Prediction) + self.bias1_prediction)
      prediction_1 = tf.matmul(prediction_0, self.Weights2_Prediction) + self.bias2_Prediction
      prediction = tf.multiply(prediction_1, self.plc_batch_bias)

      ### ---->>> Final Aggregation <<<---------------
      if predict_aggregation == 'sum':
         output_aux = tf.reduce_sum(prediction, axis=1)
      else:
         output_aux = tf.divide(tf.reduce_sum(prediction, axis=1), tf.reduce_sum(self.plc_batch_bias, axis=1)) #mean

      output = tf.matmul(output_aux, self.Weights3_Prediction)
      return output
   
   def AtomNeuralNet(self, loss_function, predict_aggregation = 'sum'):
      output = self.AtomModelBase(predict_aggregation)

      # masks are used for splitting the batch of molecules in two parts and then being possible to perform pairwise learning.
      mask1 = np.array([True if x%2==0 else False for x in range(128)]) 
      mask2 = ~mask1

      input_classifier_aux2 = tf.boolean_mask(output, mask1)
      input_classifier_aux3 = tf.boolean_mask(output, mask2)
      input_classifier = input_classifier_aux2 - input_classifier_aux3


      if loss_function=="BinaryCrossentropy":
         output_delta = tf.concat([tf.math.sigmoid(input_classifier), 1 - tf.math.sigmoid(input_classifier)], 1)
      else:
         output_delta = input_classifier

      return output_delta
   
   def TrainModel(self, param_model,
               lst_features_treino, lst_target_treino,
               lst_distancias_treino, lst_mol_sizes_treino,
               df_treino_mapping_nodup, df_treino_mapping,
               lst_features_valid, lst_target_valid,
               lst_distancias_valid, lst_mol_sizes_valid,
               df_valid_mapping_nodup, df_valid_mapping,
               loss_function, n_epochs = 1, n_batch = 64):
      
      if loss_function=="BinaryCrossentropy":
         error_aux = tf.keras.losses.BinaryCrossentropy()
         obj_loss_function = error_aux(self.plc_batch_target, param_model)
      else:
         obj_loss_function = tf.losses.mean_squared_error(self.plc_batch_target, param_model)

      obj_opt = tf.train.AdamOptimizer(learning_rate=0.0001)
      treinar = obj_opt.minimize(obj_loss_function)
      
      n_batch_per_epoch = int(len(lst_target_treino)*1.0/(2.0*n_batch))
      
      if not os.path.exists('parameters'):
         os.makedirs('./parameters')
      
      salvarParametros = tf.train.Saver()
      caminho_parametros = "./parameters/parameters.ckp"
      
      init = tf.global_variables_initializer()
      with tf.Session() as tf_sess:
         tf_sess.run(init)
         #salvarParametros.restore(tf_sess, "...")
         for i in range(n_epochs):
            inicio = time.time()
            for r in range(n_batch_per_epoch):
               batch_features, batch_dist, batch_target, batch_features_padding, batch_cfconv_padding = GerarBatch(lst_features_treino,
                                                                                                                   lst_target_treino,
                                                                                                                   lst_distancias_treino,
                                                                                                                   lst_mol_sizes_treino,
                                                                                                                   df_treino_mapping_nodup,
                                                                                                                   df_treino_mapping,
                                                                                                                   self.max_num_atm, loss_function)
               tf_sess.run(treinar, feed_dict = {self.plc_batch_features: batch_features,
                                                 self.plc_batch_dist: batch_dist,
                                                 self.plc_batch_bias: batch_features_padding,
                                                 self.plc_batch_bias_dist: batch_cfconv_padding,
                                                 self.plc_batch_target:batch_target})
               ### -----------------------------------------------------------------------------------------
               
            batch_features, batch_dist, batch_target, batch_features_padding, batch_cfconv_padding = GerarBatch(lst_features_treino,
                                                                                                                   lst_target_treino,
                                                                                                                   lst_distancias_treino,
                                                                                                                   lst_mol_sizes_treino,
                                                                                                                   df_treino_mapping_nodup,
                                                                                                                   df_treino_mapping,
                                                                                                                   self.max_num_atm, loss_function)
            pdct = tf_sess.run(obj_loss_function, feed_dict = {self.plc_batch_features: batch_features,
                                                               self.plc_batch_dist: batch_dist,
                                                               self.plc_batch_bias: batch_features_padding,
                                                               self.plc_batch_bias_dist: batch_cfconv_padding,
                                                               self.plc_batch_target: batch_target})
            mae_train = pdct
            ### -----------------------------------------------------------------------------------------
            batch_features, batch_dist, batch_target, batch_features_padding, batch_cfconv_padding = GerarBatch(lst_features_valid,
                                                                                                                lst_target_valid,
                                                                                                                lst_distancias_valid,
                                                                                                                lst_mol_sizes_valid,
                                                                                                                df_valid_mapping_nodup,
                                                                                                                df_valid_mapping,
                                                                                                                self.max_num_atm, loss_function)
            
            pdct = tf_sess.run(obj_loss_function, feed_dict = {self.plc_batch_features: batch_features,
                                                               self.plc_batch_dist: batch_dist,
                                                               self.plc_batch_bias: batch_features_padding,
                                                               self.plc_batch_bias_dist: batch_cfconv_padding,
                                                               self.plc_batch_target: batch_target})
            mae_valid = pdct
            
            ub = mae_valid
            
            params_updated = ''
            if i ==0:
               salvarParametros.save(tf_sess, caminho_parametros)
               mae_anterior = mae_valid
            else:
               if ub < mae_anterior:
                  salvarParametros.save(tf_sess, caminho_parametros)
                  mae_anterior = mae_valid
                  params_updated = '  Weights Updated :-)'
            
            fim = time.time()
            print("Epoch:", i+1, "Train Loss:", mae_train, "      Validation Loss:", mae_valid, "      Time(s):", fim-inicio, params_updated)


   def __init__(self, loss_function):
      self.max_num_atm = 23

      self.initial_feature_dim = 6 ## QM7-x

      self.dist_expand_dim = 300
      self.neurons_quantity = 128

      ############### Placeholders: Used for input data into the model pipeline ###############
      tf.compat.v1.reset_default_graph() # Reset the graph in Tensorflow

      self.plc_batch_features = tf.placeholder(dtype=tf.float64, shape=(None, self.max_num_atm, self.initial_feature_dim))
      self.plc_batch_dist = tf.placeholder(dtype=tf.float64, shape=(None, self.max_num_atm, self.max_num_atm))
      if loss_function=="BinaryCrossentropy":
         self.plc_batch_target = tf.placeholder(dtype=tf.float64, shape=(None, 2))
      else:
         self.plc_batch_target = tf.placeholder(dtype=tf.float64, shape=(None, 1))
      self.plc_batch_bias_dist = tf.placeholder(dtype=tf.float64, shape=(None, self.max_num_atm, self.max_num_atm, 1))
      self.plc_batch_bias = tf.placeholder(dtype=tf.float64, shape=(None, self.max_num_atm, 1))

      #########################################################################################

      ## ------------------->>>>Parameters of disance: RBF and Cutoff<<<<-------------------------
      self.Weights1_rbf_mu = tf.constant(np.array([[(x)*(1/10) for x in range(self.dist_expand_dim)]]), dtype=tf.dtypes.float64)

      self.Weights1_rbf_gamma = tf.constant(np.array([[10.0]]), dtype=tf.dtypes.float64)

      self.Weight_dist_cutoff = tf.Variable(tf.random.normal(shape=(1,1), mean=7.0, stddev=1, dtype=tf.dtypes.float64), name='Weight_dist_cutoff')

      ## ------------------->>>>Parameters for Embeddings<<<<-------------------------
      ### Embeddings initialized based on atomic numbers
      self.Weights_embeddings = tf.Variable(tf.random.normal(shape=(self.initial_feature_dim, self.neurons_quantity), mean=[[6],[17],[1],[7],[8],[16]], stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='Weights_embeddings')


      ## ------------------->>>> Parameters of First Interaction <<<<-------------------------
      self.Weights1_NMP_01_distance = tf.Variable(tf.random.normal(shape=(self.dist_expand_dim, self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='Weights1_NMP_01_distance')
      self.bias1_NMP_01_distance = tf.Variable(tf.random.normal(shape=(1,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='bias1_NMP_01_distance')

      self.Weights2_NMP_01_distance = tf.Variable(tf.random.normal(shape=(self.neurons_quantity,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='Weights2_NMP_01_distance')
      self.bias2_NMP_01_distance = tf.Variable(tf.random.normal(shape=(1,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='bias2_NMP_01_distance')


      self.Weights1_NMP_01_aggregation = tf.Variable(tf.random.normal(shape=(self.neurons_quantity,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='Weights1_NMP_01_aggregation')
      self.bias1_NMP_01_aggregation = tf.Variable(tf.random.normal(shape=(1,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='bias1_NMP_01_aggregation')

      self.Weights2_NMP_01_aggregation = tf.Variable(tf.random.normal(shape=(self.neurons_quantity,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='Weights2_NMP_01_aggregation')
      self.bias2_NMP_01_aggregation = tf.Variable(tf.random.normal(shape=(1,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='bias2_NMP_01_aggregation')
      ## ---------------------------------------------------------------------------

      ## ------------------->>>> Parameters of Second Interaction <<<<----------------------------
      self.Weights1_NMP_02_distance = tf.Variable(tf.random.normal(shape=(self.dist_expand_dim,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='Weights1_NMP_02_distance')
      self.bias1_NMP_02_distance = tf.Variable(tf.random.normal(shape=(1,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='bias1_NMP_02_distance')

      self.Weights2_NMP_02_distance = tf.Variable(tf.random.normal(shape=(self.neurons_quantity,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='Weights2_NMP_02_distance')
      self.bias2_NMP_02_distance = tf.Variable(tf.random.normal(shape=(1,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='bias2_NMP_02_distance')


      self.Weights1_NMP_02_aggregation = tf.Variable(tf.random.normal(shape=(self.neurons_quantity,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='Weights1_NMP_02_aggregation')
      self.bias1_NMP_02_aggregation = tf.Variable(tf.random.normal(shape=(1,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='bias1_NMP_02_aggregation')

      self.Weights2_NMP_02_aggregation = tf.Variable(tf.random.normal(shape=(self.neurons_quantity,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='Weights2_NMP_02_aggregation')
      self.bias2_NMP_02_aggregation = tf.Variable(tf.random.normal(shape=(1,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='bias2_NMP_02_aggregation')
      ## ---------------------------------------------------------------------------

      ## ------------------->>>> Parameters of Third Interaction <<<<-------------------------
      self.Weights1_NMP_03_distance = tf.Variable(tf.random.normal(shape=(self.dist_expand_dim,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='Weights1_NMP_03_distance')
      self.bias1_NMP_03_distance = tf.Variable(tf.random.normal(shape=(1,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='bias1_NMP_03_distance')

      self.Weights2_NMP_03_distance = tf.Variable(tf.random.normal(shape=(self.neurons_quantity,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='Weights2_NMP_03_distance')
      self.bias2_NMP_03_distance = tf.Variable(tf.random.normal(shape=(1,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='bias2_NMP_03_distance')


      self.Weights1_NMP_03_aggregation = tf.Variable(tf.random.normal(shape=(self.neurons_quantity,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='Weights1_NMP_03_aggregation')
      self.bias1_NMP_03_aggregation = tf.Variable(tf.random.normal(shape=(1,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='bias1_NMP_03_aggregation')

      self.Weights2_NMP_03_aggregation = tf.Variable(tf.random.normal(shape=(self.neurons_quantity,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='Weights2_NMP_03_aggregation')
      self.bias2_NMP_03_aggregation = tf.Variable(tf.random.normal(shape=(1,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='bias2_NMP_03_aggregation')
      ## ---------------------------------------------------------------------------

      ## ------------------->>>> Parameters of Fourth Interaction <<<<-------------------------
      self.Weights1_NMP_04_distance = tf.Variable(tf.random.normal(shape=(self.dist_expand_dim,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='Weights1_NMP_04_distance')
      self.bias1_NMP_04_distance = tf.Variable(tf.random.normal(shape=(1,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='bias1_NMP_04_distance')

      self.Weights2_NMP_04_distance = tf.Variable(tf.random.normal(shape=(self.neurons_quantity,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='Weights2_NMP_04_distance')
      self.bias2_NMP_04_distance = tf.Variable(tf.random.normal(shape=(1,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='bias2_NMP_04_distance')


      self.Weights1_NMP_04_aggregation = tf.Variable(tf.random.normal(shape=(self.neurons_quantity,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='Weights1_NMP_04_aggregation')
      self.bias1_NMP_04_aggregation = tf.Variable(tf.random.normal(shape=(1,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='bias1_NMP_04_aggregation')

      self.Weights2_NMP_04_aggregation = tf.Variable(tf.random.normal(shape=(self.neurons_quantity,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='Weights2_NMP_04_aggregation')
      self.bias2_NMP_04_aggregation = tf.Variable(tf.random.normal(shape=(1,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='bias2_NMP_04_aggregation')
      ## --------------------------------------------------------------------------------------

      ## ------------------->>>> Parameters of Fith Interaction <<<<-------------------------
      self.Weights1_NMP_05_distance = tf.Variable(tf.random.normal(shape=(self.dist_expand_dim,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='Weights1_NMP_05_distance')
      self.bias1_NMP_05_distance = tf.Variable(tf.random.normal(shape=(1,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='bias1_NMP_05_distance')

      self.Weights2_NMP_05_distance = tf.Variable(tf.random.normal(shape=(self.neurons_quantity,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='Weights2_NMP_05_distance')
      self.bias2_NMP_05_distance = tf.Variable(tf.random.normal(shape=(1,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='bias2_NMP_05_distance')


      self.Weights1_NMP_05_aggregation = tf.Variable(tf.random.normal(shape=(self.neurons_quantity,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='Weights1_NMP_05_aggregation')
      self.bias1_NMP_05_aggregation = tf.Variable(tf.random.normal(shape=(1,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='bias1_NMP_05_aggregation')

      self.Weights2_NMP_05_aggregation = tf.Variable(tf.random.normal(shape=(self.neurons_quantity,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='Weights2_NMP_05_aggregation')
      self.bias2_NMP_05_aggregation = tf.Variable(tf.random.normal(shape=(1,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='bias2_NMP_05_aggregation')

      ## ----------------------------------------------------------------------------------

      ## ------------------->>>> Sixth Interaction <<<<-------------------------
      self.Weights1_NMP_06_distance = tf.Variable(tf.random.normal(shape=(self.dist_expand_dim,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='Weights1_NMP_06_distance')
      self.bias1_NMP_06_distance = tf.Variable(tf.random.normal(shape=(1,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='bias1_NMP_06_distance')

      self.Weights2_NMP_06_distance = tf.Variable(tf.random.normal(shape=(self.neurons_quantity,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='Weights2_NMP_06_distance')
      self.bias2_NMP_06_distance = tf.Variable(tf.random.normal(shape=(1,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='bias2_NMP_06_distance')


      self.Weights1_NMP_06_aggregation = tf.Variable(tf.random.normal(shape=(self.neurons_quantity,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='Weights1_NMP_06_aggregation')
      self.bias1_NMP_06_aggregation = tf.Variable(tf.random.normal(shape=(1,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='bias1_NMP_06_aggregation')

      self.Weights2_NMP_06_aggregation = tf.Variable(tf.random.normal(shape=(self.neurons_quantity,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='Weights2_NMP_06_aggregation')
      self.bias2_NMP_06_aggregation = tf.Variable(tf.random.normal(shape=(1,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='bias2_NMP_06_aggregation')
      #---------------------------------------------------------------------------

      ## ------------------->>>> First Dense Layer <<<<-------------------------
      self.Weights1_Dense_01 = tf.Variable(tf.random.normal(shape=(self.neurons_quantity,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='Weights1_Dense_01')
      self.bias1_Dense_01 = tf.Variable(tf.random.normal(shape=(1,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='bias1_Dense_01')

      self.Weights2_Dense_01 = tf.Variable(tf.random.normal(shape=(self.neurons_quantity,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='Weights2_Dense_01')
      self.bias2_Dense_01 = tf.Variable(tf.random.normal(shape=(1,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='bias2_Dense_01')

      ## ------------------->>>> Second Dense Layer <<<<-------------------------
      self.Weights1_Dense_02 = tf.Variable(tf.random.normal(shape=(self.neurons_quantity,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='Weights1_Dense_02')
      self.bias1_Dense_02 = tf.Variable(tf.random.normal(shape=(1,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='bias1_Dense_02')

      self.Weights2_Dense_02 = tf.Variable(tf.random.normal(shape=(self.neurons_quantity,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='Weights2_Dense_02')
      self.bias2_Dense_02 = tf.Variable(tf.random.normal(shape=(1,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='bias2_Dense_02')

      ## ------------------->>>> Gate Layers <<<<-------------------------
      self.Weights_Gate_Embeddings = tf.Variable(tf.random.normal(shape=(1,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='Weights_Gate_Embeddings')
      self.Weights_Gate_Layer_01 = tf.Variable(tf.random.normal(shape=(1,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='Weights_Gate_Layer_01')
      self.Weights_Gate_Layer_02 = tf.Variable(tf.random.normal(shape=(1,self.neurons_quantity), mean=0.0, stddev=1/np.sqrt(self.neurons_quantity), dtype=tf.dtypes.float64), name='Weights_Gate_Layer_02')


      ## ------------------->>>> Prediction Layer <<<<-------------------------
      self.Weights1_Prediction = tf.Variable(tf.random.normal(shape=(self.neurons_quantity,32), mean=0.0, stddev=1/np.sqrt(32), dtype=tf.dtypes.float64), name='Weights1_Prediction')
      self.bias1_prediction = tf.Variable(tf.random.normal(shape=(1,32), mean=0.0, stddev=1/np.sqrt(32), dtype=tf.dtypes.float64), name='bias1_prediction')

      self.Weights2_Prediction = tf.Variable(tf.random.normal(shape=(32,32), mean=0.0, stddev=1/np.sqrt(32), dtype=tf.dtypes.float64), name='Weights2_Prediction')
      self.bias2_Prediction = tf.Variable(tf.random.normal(shape=(1,32), mean=0.0, stddev=1/np.sqrt(32), dtype=tf.dtypes.float64), name='bias2_Prediction')

      self.Weights3_Prediction = tf.Variable(tf.random.normal(shape=(32,1), mean=0.0, stddev=1/np.sqrt(32), dtype=tf.dtypes.float64), name='Weights3_Prediction')
      #########################################################################################

   ############################################## Model Inference ##################################################

   def Predict(self, modelBase, caminho_parametros,
               lst_features_teste, lst_target_teste, 
               lst_distancias_teste, lst_mol_sizes_teste):
      
      salvarParametros = tf.train.Saver()
      lst_predicted_values = list()
      with tf.Session() as tf_sess:
         salvarParametros.restore(tf_sess, caminho_parametros)
         
         for i in range(len(lst_target_teste)):
            batch_features, batch_dist, batch_target, batch_features_padding, batch_cfconv_padding = GerarBatchInference(lst_features_teste, lst_target_teste, lst_distancias_teste, lst_mol_sizes_teste, i, self.max_num_atm)
            pdct = tf_sess.run(modelBase, feed_dict = {self.plc_batch_features: batch_features,
                                                       self.plc_batch_dist: batch_dist,
                                                       self.plc_batch_bias: batch_features_padding,
                                                       self.plc_batch_bias_dist: batch_cfconv_padding})

            lst_predicted_values.append(pdct.flatten()[0])
      
      return lst_predicted_values
