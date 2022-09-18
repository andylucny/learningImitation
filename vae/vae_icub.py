import tensorflow as tf # 2.8.0
import keras
from keras import backend as K
from keras import layers
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

original_dim = 10
intermediate_dim = 6
latent_dim = 2 #2-4

# Create encoder
inputs = keras.Input(shape=(original_dim,))
h = layers.Dense(intermediate_dim, activation='relu')(inputs)
z_mean = layers.Dense(latent_dim)(h)
z_log_sigma = layers.Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=0.1)
    return z_mean + K.exp(z_log_sigma) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_sigma])
encoder = keras.Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

# Create decoder
latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
x = layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = layers.Dense(original_dim, activation='tanh')(x)
decoder = keras.Model(latent_inputs, outputs, name='decoder')

# instantiate VAE
outputs = decoder(encoder(inputs)[2])
vae = keras.Model(inputs, outputs, name='vae')

reconstruction_loss = keras.losses.mse(inputs, outputs)
reconstruction_loss *= original_dim
kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
vae.compile(optimizer=optimizer)

# load dataset
dataset = np.loadtxt('dataset.npy')
np.random.shuffle(dataset)
x_ = dataset[:,0:10]
y_ = dataset[:,10:16]

x_ = x_/180 # -1..1 for tanh
m = 4*len(x_)//5
x_train = x_[:m]
y_train = y_[:m]
x_test = x_[m:]
y_test = y_[m:]

# train
#earlyStop = tf.keras.callbacks.EarlyStopping(patience=10)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_lr=0.00001)
history = vae.fit(x_train, x_train,epochs=20,batch_size=32,validation_data=(x_test, x_test),callbacks=[reduce_lr])

# save model 
vae.save('vae-iCub-arms-autoencoder.h5')

#plot history
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.savefig('training.png')
plt.show()

# test
x_test_encoded = encoder.predict(x_test, batch_size=32)[0]

# visualize
labels = ['rx','ry','rz','lx','ly','lz']
for d in range(6):
    plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test[:,d])
    plt.colorbar()
    plt.savefig(labels[d]+'.png')
    #plt.show()

# distillate encoder and decoder
encoder.compile(optimizer='adam')
encoder.save('vae-iCub-arms-encoder.h5')
decoder.compile(optimizer='adam')
decoder.save('vae-iCub-arms-decoder.h5')

