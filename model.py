import warnings

# Suppress Tensorflow warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")


import numpy as np
import math
import katdal
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import warnings
import torch
# Your code here


block_size = 8
batch_size = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'


#Data Loading


"""
Connecting directly to the Archive.
"""

path="https://archive-gw-1.kat.ac.za/1705348101/1705348101_sdp_l0.full.rdb?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJFUzI1NiJ9.eyJpc3MiOiJrYXQtYXJjaGl2ZS5rYXQuYWMuemEiLCJhdWQiOiJhcmNoaXZlLWd3LTEua2F0LmFjLnphIiwiaWF0IjoxNzA1NjcwOTk5LCJwcmVmaXgiOlsiMTcwNTM0ODEwMSJdLCJleHAiOjE3MDYyNzU3OTksInN1YiI6InRrYXNzaWVAc2FyYW8uYWMuemEiLCJzY29wZXMiOlsicmVhZCJdfQ.389mqtfMswUJ2ADfySC0lAtqZCrL1xqDvQ4Hw30cPS_NUF69bCx-MOxm11TdCcSGL_EEbKda88O64UPNXacnxw"


def get_corrprods(vis):
    bl = vis.corr_products
    bl_idx = []
    for i in range(len(bl)):
        bl_idx.append((bl[i][0][0:-1]+bl[i][1][0:-1]))
    return np.array(bl_idx)


def get_bl_idx(vis, nant):
    """
    Get the indices of the correlation products.

    Parameters:
    -----------
    vis : katdal.visdatav4.VisibilityDataV4
       katdal data object
    nant : int
       number of antennas

    Returns:
    --------
    output : numpy array
       array of baseline indices
    """
    nant = nant
    A1, A2 = np.triu_indices(nant, 1)
   #print(A1)
    # Creating baseline antenna combinations
    corr_products = np.array(['m{:03d}m{:03d}'.format(A1[i], A2[i]) for i in range(len(A1))])
    df = pd.DataFrame(data=np.arange(len(A1)), index=corr_products).T
    corr_prods = get_corrprods(vis)
    bl_idx = df[corr_prods].values[0].astype(np.int32)

    return bl_idx

def read_rdb(path):
    data = katdal.open(path)
    data.select(dumps = slice(0,10), scans ='track', pol='HH', corrprods='cross')
    data_HH = np.zeros((4096, 2016))
    bl_idx_HH = get_bl_idx( data, 64)
    data_HH[:,:] = np.zeros_like(data_HH)
    data_HH[:, bl_idx_HH] = np.abs(data.vis[2,:,:])
    
    df = pd.DataFrame(data_HH)

    return data_HH, df

def get_batch(split):

    data_test =  train_data if split == 'train' else val_data
    data_test = torch.tensor(data_test)
    ix = torch.randint(len(data_test)-block_size, (batch_size, ))
    x = torch.stack([data_test[i:i+block_size] for i in ix])
    y = torch.stack([data_test[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
data, df = read_rdb(path)
n = int(0.9*len(data)) 
train_data = data[:n]
val_data = data[n:]
xtrain_data, ytrain_data = get_batch('split')
xtest, ytest = get_batch('train')

print(xtrain_data.shape)
print(xtrain_data)
print(ytrain_data.shape)
print(ytrain_data)




def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res












def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1)(x)
    return keras.Model(inputs, outputs)



input_shape = xtrain_data.shape[1:]

model = build_model(
    input_shape,
    head_size=256,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=[128],
    mlp_dropout=0.4,
    dropout=0.25,
)

model.compile(
    loss="mean_squared_error",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4)
)
#model.summary()

callbacks = [keras.callbacks.EarlyStopping(patience=10, \
    restore_best_weights=True)]

model.fit(
    xtrain_data,
    ytrain_data,
    validation_split=0.2,
    epochs=200,
    batch_size=64,
    callbacks=callbacks,
)

model.evaluate(x_test, y_test, verbose=1)


