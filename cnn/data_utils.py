import os
import sys
import pickle as pickle
import numpy as np
import tensorflow as tf


def _read_data(data_path, train_files):
  """Reads CIFAR-10 format data. Always returns NHWC format.

  Returns:
    images: np tensor of size [N, H, W, C]
    labels: np tensor of size [N]
  """
  images, labels = [], []
  for file_name in train_files:
    print(file_name)
    full_name = os.path.join(data_path, file_name)
    with open(full_name,'rb') as finp:
      data = pickle.load(finp,encoding='iso-8859-1')
      batch_images = data["data"].astype(np.float32) / 255.0
      batch_labels = np.array(data["labels"], dtype=np.int32)
      images.append(batch_images)
      labels.append(batch_labels)
  images = np.concatenate(images, axis=0)
  labels = np.concatenate(labels, axis=0)
  images = np.reshape(images, [-1, 3, 32, 32])
  images = np.transpose(images, [0, 2, 3, 1])

  return images, labels

def read_vein_data(data_path,num_valids=0):
    data_path="data/vein/img"
    model_name="fv_img_new_10"
    times=int(model_name.split("_")[-1])
    arg_num=25

    dataset=np.loadtxt(open("file/"+model_name+"_path_file.txt"), dtype=np.str, delimiter=' ')
    train=dataset[dataset[:,0]=='train',1:]
    test=dataset[dataset[:,0]=='test',1:]
    np.random.shuffle(train)
    x_train=train[:,0]
    y_train=train[:,1].astype('int64')
    class_num=int(np.max(y_train))+1
    
    test=test[np.lexsort(test.T)]
    test=test.reshape((-1,times,2))
    test=test[:,:,0]

    x_train_for_err=train[np.lexsort(train.T)]
    x_train_for_err=x_train_for_err.reshape((-1,arg_num*times,2))[:30,:,0]   
    idxs=np.random.choice(arg_num*times,times,replace=False)
    x_train_for_err=x_train_for_err[:,idxs]

    img_paths, labels = {}, {}

    img_paths["train"],labels["train"]=x_train,y_train
    img_paths["valid"]=test[:10,:6]
    img_paths["test"]=test
    img_paths["train_valid"]=x_train_for_err
    return img_paths,labels

def read_data(data_path, train_portion):
  print ("-" * 80)
  print ("Reading data")

  images, labels = {}, {}

  train_files = [
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5",
  ]
  test_file = [
    "test_batch",
  ]
  images["train"], labels["train"] = _read_data(data_path, train_files)

  num_train = len(images["train"])
  indices = list(range(num_train))
  split = int(np.floor(train_portion * num_train))

  images["valid"] = images["train"][split:num_train]
  labels["valid"] = labels["train"][split:num_train]

  images["train"] = images["train"][:split]
  labels["train"] = labels["train"][:split]


  images["test"], labels["test"] = _read_data(data_path, test_file)

  print ("Prepropcess: [subtract mean], [divide std]")
  mean = np.mean(images["train"], axis=(0, 1, 2), keepdims=True)
  std = np.std(images["train"], axis=(0, 1, 2), keepdims=True)

  print ("mean: {}".format(np.reshape(mean * 255.0, [-1])))
  print ("std: {}".format(np.reshape(std * 255.0, [-1])))

  images["train"] = (images["train"] - mean) / std

  images["valid"] = (images["valid"] - mean) / std
  images["test"] = (images["test"] - mean) / std

  return images, labels

