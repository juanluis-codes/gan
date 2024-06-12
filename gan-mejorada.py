import os
import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import numpy as np
import PIL
from tensorflow.keras import layers
import time
import pandas as pd

class GAN:
    def __init__(self, celebA_path, buffer_size=30000, batch_size=256, epochs=100, noise_dim=100):
        self.celebA_path = celebA_path
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.noise_dim = noise_dim
        self.num_examples_to_generate = 16
        self.seed = tf.random.normal([self.num_examples_to_generate, self.noise_dim])
        self.generator = self.make_generator_model()
        self.discriminator = self.make_discriminator_model()
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)
        self.images_dir = './images'
        os.makedirs(self.images_dir, exist_ok=True)
        self.losses = []

    def preprocess_data(self):
        celebA_images = glob.glob(os.path.join(self.celebA_path, "*.jpg"))
        celebA_images = celebA_images[:30000]
        celebA_images = np.array([np.array(PIL.Image.open(img)) for img in celebA_images])
        celebA_images = tf.image.resize(celebA_images, (64, 64))
        celebA_images = tf.cast(celebA_images, tf.float32)
        celebA_images = (celebA_images - 127.5) / 127.5
        train_dataset = tf.data.Dataset.from_tensor_slices(celebA_images).shuffle(self.buffer_size).batch(self.batch_size)
        return train_dataset

    def make_generator_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(100,)))
        model.add(layers.Dense(8*8*256, use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Reshape((8, 8, 256)))
        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        return model

    def make_discriminator_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(64, 64, 3)))
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        model.add(layers.Flatten())
        model.add(layers.Dense(1))
        return model

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([self.batch_size, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return gen_loss, disc_loss

    def train_and_save_model(self):
        dataset = self.preprocess_data()
        for epoch in range(self.epochs):
            start = time.time()
            gen_loss_epoch = 0
            disc_loss_epoch = 0

            for image_batch in dataset:
                gen_loss, disc_loss = self.train_step(image_batch)
                gen_loss_epoch += gen_loss
                disc_loss_epoch += disc_loss

            gen_loss_epoch /= len(dataset)
            disc_loss_epoch /= len(dataset)
            self.losses.append((epoch + 1, gen_loss_epoch, disc_loss_epoch, time.time() - start))

            self.generate_and_save_images(epoch + 1, self.seed)

            if (epoch + 1) % 15 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

        self.generate_and_save_images(self.epochs, self.seed)
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)
        self.plot_losses()

    def generate_and_save_images(self, epoch, test_input):
        predictions = self.generator(test_input, training=False)

        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow((predictions[i, :, :, :] + 1) / 2)
            plt.axis('off')

        plt.savefig(os.path.join(self.images_dir, 'image_at_epoch_{:04d}.png'.format(epoch)))
        plt.close(fig)

    def plot_losses(self):
        epochs, gen_losses, disc_losses, times = zip(*self.losses)

        fig, ax = plt.subplots()
        ax.plot(epochs, gen_losses, label='Generator Loss')
        ax.plot(epochs, disc_losses, label='Discriminator Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()

        plt.savefig(os.path.join(self.images_dir, 'losses.png'))
        plt.close(fig)

        df = pd.DataFrame(self.losses, columns=['Epoch', 'Generator Loss', 'Discriminator Loss', 'Time (s)'])
        df.to_csv(os.path.join(self.images_dir, 'losses.csv'), index=False)

# Configuración inicial y entrenamiento
celebA_path = "C:\\Users\\juanl\\OneDrive\\Documentos\\3-Estudios\\MASTER\\TFM\\POC\\img_align_celeba\\img_align_celeba"
gan = GAN(celebA_path)
gan.train_and_save_model()
