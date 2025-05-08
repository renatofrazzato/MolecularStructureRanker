import numpy as np

def GetFeaturesPaddingAdjustments(param_mol_sizes, param_max_size):
  lst_bias_features = list()
  for c in param_mol_sizes:
    mol_bias_vec = np.concatenate([np.ones([c,1]), np.zeros([param_max_size-c,1])], 0)
    lst_bias_features.append(mol_bias_vec)

  return np.array(lst_bias_features)


def GetDistPaddingAdjustments(param_batch_mol_size, param_max_atm):
  lst_batch_result = list()
  for i in range(len(param_batch_mol_size)):
    lst_batch_result_aux = list()
    for r in range(param_batch_mol_size[i]):
      vec_aux = np.ones((param_batch_mol_size[i],1))
      vec_aux[r,0] = 0.0
      vec_aux_zeros = np.zeros((param_max_atm-param_batch_mol_size[i],1))
      vec_aux = np.concatenate([vec_aux,vec_aux_zeros],0)
      lst_batch_result_aux.append(vec_aux)

    for r in range(param_max_atm-param_batch_mol_size[i]):
      vec_aux = np.zeros((param_max_atm,1))
      lst_batch_result_aux.append(vec_aux)

    lst_batch_result.append(lst_batch_result_aux)
  return np.array(lst_batch_result)


def GerarBatch(p_features, p_target, 
               p_distancias, p_mol_sizes, 
               p_df_mapping_no_dup, p_df_mapping, 
               param_max_size, loss_function):
  
  lst_batch_features = list()
  lst_batch_dist = list()
  lst_batch_target = list()
  aux_batch_mol_size = list()

  df_aux_sample = p_df_mapping_no_dup.sample(n=1)
  aux_molecula = df_aux_sample['molecula'].values[0]
  aux_iso = df_aux_sample['isomero'].values[0]
  aux_conform = df_aux_sample['conformero'].values[0]

  str_query = "molecula == '" + aux_molecula + "' and isomero == '" + aux_iso + "' and conformero == '" + aux_conform + "'"
  df_aux = p_df_mapping.query(str_query)
  aux_index_moleculas = df_aux.index.values
  first_par = np.random.choice(aux_index_moleculas, size=64)
  second_par = np.random.choice(aux_index_moleculas, size=64)

  for i in range(64):
    lst_batch_features.append(p_features[first_par[i]])
    lst_batch_features.append(p_features[second_par[i]])
    lst_batch_dist.append(p_distancias[first_par[i]])
    lst_batch_dist.append(p_distancias[second_par[i]])

    if loss_function == "BinaryCrossentropy":
      if p_target[first_par[i]] > p_target[second_par[i]]:
        aux_target = [1,0]
      elif p_target[first_par[i]] < p_target[second_par[i]]:
        aux_target = [0,1]
      else:
        aux_target = [0.5,0.5]
    else:
      aux_target = [p_target[first_par[i]] - p_target[second_par[i]]]

    lst_batch_target.append(aux_target)

    aux_batch_mol_size.append(p_mol_sizes[first_par[i]])
    aux_batch_mol_size.append(p_mol_sizes[second_par[i]])

  lst_batch_features_padding = GetFeaturesPaddingAdjustments(aux_batch_mol_size, param_max_size)
  lst_batch_cfconv_padding = GetDistPaddingAdjustments(aux_batch_mol_size, param_max_size)

  return np.array(lst_batch_features), np.array(lst_batch_dist), np.array(lst_batch_target), lst_batch_features_padding, lst_batch_cfconv_padding


def GerarBatchInference(p_features, p_target, p_distancias, p_mol_sizes, idx_molecula, param_max_size):
  lst_batch_features = list()
  lst_batch_dist = list()
  lst_batch_target = list()
  aux_batch_mol_size = list()

  val_target = p_target[idx_molecula]
  val_dist = p_distancias[idx_molecula]
  val_mol_size = p_mol_sizes[idx_molecula]
  val_atm_features = p_features[idx_molecula]

  lst_batch_features.append(val_atm_features)
  lst_batch_target.append([val_target])
  lst_batch_dist.append(val_dist)
  aux_batch_mol_size.append(val_mol_size)


  lst_batch_features_padding = GetFeaturesPaddingAdjustments(aux_batch_mol_size, param_max_size)
  lst_batch_cfconv_padding = GetDistPaddingAdjustments(aux_batch_mol_size, param_max_size)

  return np.array(lst_batch_features), np.array(lst_batch_dist), np.array(lst_batch_target), lst_batch_features_padding, lst_batch_cfconv_padding