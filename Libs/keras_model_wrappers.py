import tensorflow as tf
import numpy as np

import os
import re
# Tensorflow Dist
import tensorflow_probability as tfp
# Custom lib
import image_utils

# CLASS USED TO WRAP ALL GAN MODELS LIKE
class GAN_wrapper(tf.keras.Model):

    def __init__(self, name, discriminator, generator, batch_size, noise_dim, num_classes, seed=1234, gan_metrics=[], base_path="", max_saved_models=3):

        super(GAN_wrapper, self).__init__(name=name)
        self.base_path = base_path

        # Restore Models if not passed as argument
        if (not discriminator) or (not generator):
            print(f"Restoring model from {base_path}")
            self.load_model()
        else:
            self.discriminator = discriminator
            self.generator = generator

        self.noise_dim = noise_dim
        self.batch_size = batch_size
        self.num_classes = num_classes
        # Gan Metrics
        self.gan_metrics = gan_metrics
        # Random seed
        self.seed = seed

        # Number of Saved Model to Mantain
        self.max_saved_models = max_saved_models
        self.save_iteration = 0  # Eliminate save_iteration-max_saved_models file

    def compile(self, discriminator_optimizer, generator_optimizer, discriminator_loss, generator_loss):
        # super.compile()
        super(GAN_wrapper, self).compile()
        # Losses
        self.discriminator_loss = discriminator_loss
        self.generator_loss = generator_loss

        # Optimizers
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer

    # Save the models to file
    def save_model(self):

        # Delete the exceding saved model
        if self.save_iteration >= self.max_saved_models:
            os.remove(self.base_path +
                      f"Generator_{self.save_iteration-self.max_saved_models}")
            os.remove(
                self.base_path + f"Discriminator_{self.save_iteration-self.max_saved_models}")

        # Save model
        _ = self.generator.save(
            self.base_path + f"Generator_{self.save_iteration}", save_format='h5')  # h5 lighter save
        _ = self.discriminator.save(
            self.base_path + f"Discriminator_{self.save_iteration}", save_format='h5')

        # Update counter for saved files
        self.save_iteration += 1

    # Reload the latest models from file
    def load_model(self):

        saved_filenames = os.listdir(self.base_path)

        # Find latest model file
        last_saved_iteration = 0
        for saved_filename in saved_filenames:
            last_saved_iteration = max(int(re.search(
                r'\d+', saved_filename, re.IGNORECASE).group()), last_saved_iteration)

        self.generator = tf.keras.models.load_model(
            self.base_path + f"Generator_{last_saved_iteration}")
        self.discriminator = tf.keras.models.load_model(
            self.base_path + f"Discriminator_{last_saved_iteration}")


# Generator, Discriminator vanilla
class BASE_GAN(GAN_wrapper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Generate the noise for generator
    def sample_noise(self, batch_size):
        return tf.random.normal(shape=[batch_size, self.noise_dim])

    # Generate examples from generator
    def sample_from_model(self, num_examples_to_generate=10):

        noise_sample = self.sample_noise(
            [num_examples_to_generate, self.noise_dim])
        
        generated_images = self.generator.predict(
                noise_sample, verbose=0)
        # Display Generated Images        
        image_utils.display_multiple_image(image_utils.denormalize_images(
                generated_images),  size=(900, 900), inline=True)


            
    @tf.function
    def train_step(self, real_images):
        print("Tracing... ")
        # Discriminator
        with tf.GradientTape() as disc_tape:
            # Sample Noise
            noise_input = self.sample_noise(self.batch_size)

            # Generate Images given inputs
            generated_images = self.generator(noise_input, training=True)

            # Output of the discriminator based on fake images
            fake_output = self.discriminator(generated_images, training=True)
            # Output of the discriminator based on real images
            real_output = self.discriminator(real_images, training=True)

            # Discriminator loss based on real and fake images
            disc_loss, real_loss, fake_loss = self.discriminator_loss(
                real_output, fake_output)

        # Discriminator Gradient
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables)
        # Discriminator Backprop
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        #  Generator
        with tf.GradientTape() as gen_tape:

            noise_input = self.sample_noise(self.batch_size)

            # Generate Images given Noise
            generated_images = self.generator(noise_input, training=True)

            # Output of the discriminator based on fake images
            disc_output = self.discriminator(generated_images, training=True)

            # Generator loss based only on fake images
            gen_loss = self.generator_loss(disc_output)

        # Generator Gradient
        gradients_of_generator = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables)
        # Generator Backprop
        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables))

        # Compute some metrics
        for gan_metric in self.gan_metrics:
            # Check if it is a metric for the discriminator
            if gan_metric.GAN_module == 'discriminator':
                gan_metric.update_metric(
                    output_on_real_images=real_output, output_on_fake_images=fake_output)

            if gan_metric.GAN_module == 'generator':
                gan_metric.update_metric(disc_output=disc_output)


# Information Maximization GAN
class InfoGAN_wrapper(GAN_wrapper):

    def __init__(self, q_model, cont_variables, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Architecture
        self.q_model = q_model
        # Continous distributions
        self.cont_dist = tfp.distributions.Normal(
            loc=[0]*cont_variables, scale=[1]*cont_variables)

    def compile(self, discriminator_optimizer, generator_optimizer, discriminator_loss, generator_loss):
        # super.compile()
        super().compile(discriminator_optimizer, generator_optimizer,
                        discriminator_loss, generator_loss)
        # Losses
        self.q_cat_loss = tf.keras.losses.CategoricalCrossentropy()
        # Optimizers
        #self.q_optimizer = q_optimizer

    # Sample and display some input

    def sample_from_model(self, num_examples=1, explore_var="cont", sampling_range=[-5, 5]):

        if explore_var == "cat":
            print(" Variations of the categorical variable (Row) ")

            for i in range(self.num_classes):

                noise_input, cat_input, cont_input = self.sample_generator_input(
                    batch_size=num_examples)

                # Replicate the same input several times
                #noise_input = np.array( [ noise_input[0]]*self.num_classes )
                cat_input = np.array([i]*num_examples)
                #cont_input = np.array( [ cont_input[0]]*self.num_classes )

                # Batch data
                gen_input = [noise_input, cat_input, cont_input]
                # Generate Images
                generated_images = self.generator(gen_input, training=False)
                # Plot the Images
                print(f"Class {i}")
                image_utils.display_multiple_image(image_utils.denormalize_images(
                    generated_images),  size=(900, 900), inline=True)

        elif explore_var == "cont":
            # Sampling interval
            minval = sampling_range[0]
            maxval = sampling_range[1]
            print(" Variations of the continous variable (Column) ")
            print(
                f"Continous values sampled and sorted in range [{minval}, {maxval}]")

            for _ in range(0, 5):

                # Sample some input
                noise_input, cat_input, _ = self.sample_generator_input(
                    batch_size=1)

                # Replicate the same input several times

                noise_input = np.array([noise_input[0]]*num_examples)
                cat_input = np.array([cat_input[0]]*num_examples)
                #cont_input = np.array( [ cont_input[0]]*num_examples )

                # Explore the continous value conditioning
                #cont_input = tf.sort( tf.random.uniform( [num_examples,1] , minval=minval, maxval=maxval), axis=0 )
                cont_input = tf.sort(
                    self.cont_dist.sample([num_examples]), axis=-1)

                # Batch data
                gen_input = [noise_input, cat_input, cont_input]
                # Generate Images
                generated_images = self.generator(gen_input, training=False)
                # Plot the Images
                image_utils.display_multiple_image(image_utils.denormalize_images(
                    generated_images),  size=(900, 900), inline=True)

    def sample_generator_input(self, batch_size):

        # create noise input
        noise_input = tf.random.normal(
            [batch_size, self.noise_dim], seed=self.seed)
        # Create categorical latent code
        cat_input = tf.random.uniform(
            [batch_size], minval=0, maxval=self.num_classes, dtype=tf.int32, seed=self.seed)
        #label = tf.one_hot(label, depth=self.num_classes)
        # Create one continuous latent code
        #cont_input = tf.random.uniform([batch_size, 1], minval=-1, maxval=1, seed=self.seed)
        cont_input = self.cont_dist.sample([batch_size])

        return noise_input, cat_input, cont_input


    # Perform a training step over the input
    @tf.function
    def train_step(self, real_images):
        print("Tracing...")

        # Discriminator
        # Allow the discriminator to be trained
        self.discriminator.trainable = True

        with tf.GradientTape() as disc_tape:

            # Sample Noise
            noise_input, cat_input, cont_input = self.sample_generator_input(
                self.batch_size)

            # Generate Images given inputs
            generated_images = self.generator(
                [noise_input, cat_input, cont_input], training=True)

            # Output of the discriminator based on fake images
            fake_output = self.discriminator(generated_images, training=True)
            # Output of the discriminator based on real images
            real_output = self.discriminator(real_images, training=True)

            # Discriminator loss based on real and fake images
            disc_loss, real_loss, fake_loss = self.discriminator_loss(
                real_output, fake_output)

        # Discriminator Gradient
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables)
        # Discriminator Backprop
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        # Generator
        with tf.GradientTape() as gen_tape:

            # Sample Noise
            noise_input, cat_input, cont_input = self.sample_generator_input(
                self.batch_size)

            # Generate Images given Noise
            generated_images = self.generator(
                [noise_input, cat_input, cont_input], training=True)

            # Output of the discriminator based on fake images
            disc_output = self.discriminator(generated_images, training=True)

            # Generator loss based only on fake images
            gen_loss = self.generator_loss(disc_output)

            # Information Maximization Part

            cat_output, mu, sigma = self.q_model(
                generated_images, training=True)
            
            # Categorical loss
            cat_loss = self.q_cat_loss(tf.one_hot(
                cat_input, depth=self.num_classes), cat_output)

            # Also possible to use the KL divergence instead of sampling probability
            # Use Gaussian distributions to represent the output
            generator_dist = tfp.distributions.Normal(loc=mu, scale=sigma)
            # Losses (negative log probability density function as we want to maximize the probability density function)
            cont_loss = tf.reduce_mean(-generator_dist.log_prob(cont_input))

            # Auxiliary model loss (INFOGAN)
            infoGAN_loss = cat_loss + 0.1*cont_loss

            # Final Generator loss
            generator_total_loss = gen_loss + infoGAN_loss

        # Do not update the discriminator part, just the q network
        self.discriminator.trainable = False

        # Generator Gradients
        gradients_of_generator = gen_tape.gradient(
            generator_total_loss, self.q_model.trainable_variables + self.generator.trainable_variables)

        # Generator Backprop
        self.generator_optimizer.apply_gradients(zip(
            gradients_of_generator, self.q_model.trainable_variables + self.generator.trainable_variables))

        '''
        # Gradients Q_model+Generator
        q_gradients = q_tape.gradient(q_loss, self.q_model.trainable_variables + self.generator.trainable_variables)
        # Backprop
        self.q_optimizer.apply_gradients(zip(q_gradients, self.q_model.trainable_variables+self.generator.trainable_variables))

        
        
        # Generator Gradient
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        # Generator Backprop
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        '''

        # Compute some metrics (Last step only considered)
        for gan_metric in self.gan_metrics:
            # Check if it is a metric for the discriminator
            if gan_metric.GAN_module == 'discriminator':
                gan_metric.update_metric(
                    output_on_real_images=real_output, output_on_fake_images=fake_output)

            if gan_metric.GAN_module == 'generator':
                gan_metric.update_metric(disc_output=disc_output)

            if gan_metric.GAN_module == 'generic_metric':
                gan_metric.update_metric(infoGAN_loss)

    # Save the models to file

    def save_model(self):
        super().save_model()
        # Silence the notice
        # _= self.q_model.save(f"q_model_{self.name}", save_format='h5') # Need to update to have max_saved_models, in case

    # Reload the models from file
    def load_model(self):
        super().load_model()
        #self.q_model = tf.keras.load_model(self.base_path+ f"q_model_{self.name}.h5")


# INFOGAN + LATENT SPACE MATCHING
# Information Maximization GAN
class InfoGAN_LM_wrapper(InfoGAN_wrapper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.MSE = tf.keras.losses.MeanSquaredError()

    # Perform a training step over the input

    @tf.function
    def train_step(self, real_images):

        print("Tracing... ")

        # Discriminator
        # Allow the discriminator to be trained
        self.discriminator.trainable = True

        with tf.GradientTape() as disc_tape:

            # Sample Noise
            noise_input, cat_input, cont_input = self.sample_generator_input(
                self.batch_size)
            #g_label, c_1, gen_noise = self.sample_noise(self.batch_size)

            #gen_input = self.concat_inputs( (g_label, c_1, gen_noise) )

            # Generate Images given inputs
            generated_images = self.generator(
                [noise_input, cat_input, cont_input], training=True)

            # Output of the discriminator based on fake images
            fake_output = self.discriminator(generated_images, training=True)
            # Output of the discriminator based on real images
            real_output = self.discriminator(real_images, training=True)

            # Discriminator loss based on real and fake images
            disc_loss, real_loss, fake_loss = self.discriminator_loss(
                real_output, fake_output)

        # Discriminator Gradient
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables)
        # Discriminator Backprop
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        # Generator
        # Train Twice
        for _ in range(2):
            # In order to work properly
            self.discriminator.trainable = True

            with tf.GradientTape() as gen_tape:

                # Sample Noise
                noise_input, cat_input, cont_input = self.sample_generator_input(
                    self.batch_size)

                # Generate Images given Noise
                generated_images = self.generator(
                    [noise_input, cat_input, cont_input], training=True)

                # Output of the discriminator based on fake images
                disc_output = self.discriminator(
                    generated_images, training=True)

                # Generator loss based only on fake images
                gen_loss = self.generator_loss(disc_output)

                # Aux losses
                cat_output, mu, sigma, latent_space_fake = self.q_model(
                    generated_images, training=True)

                # Categorical loss
                cat_loss = self.q_cat_loss(tf.one_hot(
                    cat_input, depth=self.num_classes), cat_output)
                # Use Gaussian distributions to represent the output
                generator_dist = tfp.distributions.Normal(loc=mu, scale=sigma)
                # Losses (negative log probability density function as we want to maximize the probability density function)
                cont_loss = tf.reduce_mean(-generator_dist.log_prob(cont_input))

                # Info GAN Loss
                infoGAN_loss = cat_loss + 0.2*cont_loss

                # Latent Space Loss
                _, _, _, latent_space_real = self.q_model(
                    real_images, training=True)

                latent_matching_loss = self.MSE(
                    latent_space_real, latent_space_fake)

                # Final Generator loss
                generator_total_loss = gen_loss + infoGAN_loss + latent_matching_loss

            # Do not update the discriminator part, just the q network
            self.discriminator.trainable = False

            # Generator Gradients
            gradients_of_generator = gen_tape.gradient(
                generator_total_loss, self.q_model.trainable_variables + self.generator.trainable_variables)
            # Generator Backprop
            self.generator_optimizer.apply_gradients(zip(
                gradients_of_generator, self.q_model.trainable_variables + self.generator.trainable_variables))

            # Generator Gradient
            #gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            # Generator Backprop
            #self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        #del gen_tape

        # Compute some metrics (Last step only considered)
        for gan_metric in self.gan_metrics:
            # Check if it is a metric for the discriminator
            if gan_metric.GAN_module == 'discriminator':
                gan_metric.update_metric(
                    output_on_real_images=real_output, output_on_fake_images=fake_output)

            if gan_metric.GAN_module == 'generator':
                gan_metric.update_metric(disc_output=disc_output)

            if gan_metric.GAN_module == 'generic_metric':
                if gan_metric.name == "Q_Loss":
                    gan_metric.update_metric(infoGAN_loss)

                if gan_metric.name == "Latent Matching Loss":
                    gan_metric.update_metric(latent_matching_loss)


# Contidioned GAN
class CGAN_wrapper(GAN_wrapper):
    def __init__(self, *args, **kwargs):
        # Old Lib?
        super().__init__(*args, **kwargs)

    # Sample Noise for Generator
    def sample_noise(self, batch_size):
        return tf.random.normal(shape=[batch_size, self.noise_dim])
    # Sample Random Labels for Generator
    def sample_labels(self, batch_size):
        return tf.random.uniform(maxval = self.num_classes ,shape=[batch_size], dtype = tf.int32)
    
    # Generate Data from the model
    def sample_from_model(self, num_examples_to_generate = 16):
        # Sample Noise and Labels
        noise_input = self.sample_noise(num_examples_to_generate)
        labels = self.sample_labels(num_examples_to_generate)
        # Generate images
        gen_images = self.generator.predict([noise_input, labels], verbose=0)
        # Display Images
        image_utils.display_multiple_image(image_utils.denormalize_images(
                gen_images),  size=(900, 900), inline=True)
        
    # Perfom a training step give the inputs
    @tf.function
    def train_step(self, inputs):

        print("Tracing... ")
        real_images, labels = inputs

        # Discriminator
        with tf.GradientTape() as disc_tape:

            # Sample Noise
            noise = self.sample_noise(self.batch_size)

            # Generate Images given inputs
            generated_images = self.generator([noise, labels], training=True)

            # Output of the discriminator based on fake images
            fake_output = self.discriminator(
                [generated_images, labels], training=True)
            # Output of the discriminator based on real images
            real_output = self.discriminator(
                [real_images, labels], training=True)

            # Discriminator loss based on real and fake images
            disc_loss, real_loss, fake_loss = self.discriminator_loss(
                real_output, fake_output)

        # Discriminator Gradient
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables)
        # Discriminator Backprop
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        # Generator
        with tf.GradientTape() as gen_tape:

            # Sample Noise
            noise = self.sample_noise(self.batch_size)

            # Generate Images given Noise
            generated_images = self.generator([noise, labels], training=True)

            # Output of the discriminator based on fake images
            disc_output = self.discriminator(
                [generated_images, labels], training=True)

            # Generator loss based only on fake images
            gen_loss = self.generator_loss(disc_output)

        # Generator Gradient
        gradients_of_generator = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables)
        # Generator Backprop
        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables))

        # Compute some metrics
        for gan_metric in self.gan_metrics:
            # Check if it is a metric for the discriminator
            if gan_metric.GAN_module == 'discriminator':
                gan_metric.update_metric(
                    output_on_real_images=real_output, output_on_fake_images=fake_output)

            if gan_metric.GAN_module == 'generator':
                gan_metric.update_metric(disc_output=disc_output)


# Contidioned GAN w Latent Matching Loss
class CGAN_LM_wrapper(GAN_wrapper):
    def __init__(self, *args, **kwargs):
        # Old Lib?
        super().__init__(*args, **kwargs)
        # Latent Matching loss
        self.MSE = tf.keras.losses.MeanSquaredError()
    
    @tf.function
    def train_step(self, inputs):

        print("Tracing... ")
        real_images, labels = inputs

        # Discriminator
        with tf.GradientTape() as disc_tape:

            # Sample Noise
            noise = self.sample_noise(self.batch_size)

            # Generate Images given inputs
            generated_images = self.generator([noise, labels], training=True)

            # Output of the discriminator based on fake images
            fake_output, _ = self.discriminator(
                [generated_images, labels], training=True)
            # Output of the discriminator based on real images
            real_output, latent_space_real = self.discriminator(
                [real_images, labels], training=True)

            # Discriminator loss based on real and fake images
            disc_loss, real_loss, fake_loss = self.discriminator_loss(
                real_output, fake_output)

        # Discriminator Gradient
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables)
        # Discriminator Backprop
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        # Generator
        with tf.GradientTape() as gen_tape:

            # Sample Noise
            noise = self.sample_noise(self.batch_size)

            # Generate Images given Noise
            generated_images = self.generator([noise, labels], training=True)

            # Output of the discriminator based on fake images
            disc_output, latent_space_fake = self.discriminator(
                [generated_images, labels], training=True)

            # Generator loss based only on fake images
            gen_loss = self.generator_loss(disc_output)
            # Mean over batch dimension and then MSE*2
            latent_matching_loss = tf.math.multiply(self.MSE(tf.reduce_mean(
                latent_space_real, 0), tf.reduce_mean(latent_space_fake, 0)), 5)

            # Final Generator loss
            generator_total_loss = gen_loss + latent_matching_loss

        # Generator Gradient
        gradients_of_generator = gen_tape.gradient(
            generator_total_loss, self.generator.trainable_variables)
        # Generator Backprop
        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables))

        # Compute some metrics
        for gan_metric in self.gan_metrics:
            # Check if it is a metric for the discriminator
            if gan_metric.GAN_module == 'discriminator':
                gan_metric.update_metric(
                    output_on_real_images=real_output, output_on_fake_images=fake_output)

            if gan_metric.GAN_module == 'generator':
                gan_metric.update_metric(disc_output=disc_output)

            if gan_metric.GAN_module == 'generic_metric':

                if gan_metric.name == "Latent Matching Loss":
                    gan_metric.update_metric(latent_matching_loss)
        


# Contidioned GAN w Latent Matching Loss + CLassification Loss
class CGAN_LM_Classification_wrapper(CGAN_LM_wrapper):

    def __init__(self, *args, **kwargs):
        # Old Lib?
        super().__init__(*args, **kwargs)

        self.MSE = tf.keras.losses.MeanSquaredError()
        self.SparseCrossEntropy = tf.keras.losses.SparseCategoricalCrossentropy()

    @tf.function
    def train_step(self, inputs):

        print("Tracing... ")
        real_images, labels = inputs

        # Discriminator
        with tf.GradientTape() as disc_tape:

            # Sample Noise
            noise = self.sample_noise(self.batch_size)

            # Generate Images given inputs
            generated_images = self.generator([noise, labels], training=True)

            # Output of the discriminator based on fake images
            fake_output, _, _ = self.discriminator(
                [generated_images, labels], training=True)
            # Output of the discriminator based on real images
            real_output, latent_space_real, class_output = self.discriminator(
                [real_images, labels], training=True)

            # Discriminator loss based on real and fake images
            disc_loss, real_loss, fake_loss = self.discriminator_loss(
                real_output, fake_output)
            
            #print(tf.one_hot( tf.cast(labels, tf.int32), depth=3))
            # Discriminator classification loss
            class_loss = self.SparseCrossEntropy( labels, class_output ) # flow from directory returns float sparse classes
            

            #total_loss = tf.reduce_mean( disc_loss + class_loss)
            #total_loss = disc_loss + class_loss
            total_loss = [disc_loss, class_loss]

        # Discriminator Gradient
        gradients_of_discriminator = disc_tape.gradient(
            total_loss, self.discriminator.trainable_variables)
        # Discriminator Backprop
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        # Generator
        with tf.GradientTape() as gen_tape:

            # Sample Noise
            noise = self.sample_noise(self.batch_size)

            # Generate Images given Noise
            generated_images = self.generator([noise, labels], training=True)

            # Output of the discriminator based on fake images
            disc_output, latent_space_fake, _ = self.discriminator(
                [generated_images, labels], training=True)

            # Generator loss based only on fake images
            gen_loss = self.generator_loss(disc_output)
            # Mean over batch dimension and then MSE*2
            latent_matching_loss = tf.math.multiply(self.MSE(tf.reduce_mean(
                latent_space_real, 0), tf.reduce_mean(latent_space_fake, 0)), 5)

            # Final Generator loss
            generator_total_loss = [gen_loss, latent_matching_loss]

        # Generator Gradient
        gradients_of_generator = gen_tape.gradient(
            generator_total_loss, self.generator.trainable_variables)
        # Generator Backprop
        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables))

        # Compute some metrics and update internal state
        for gan_metric in self.gan_metrics:
            # Check if it is a metric for the discriminator
            if gan_metric.GAN_module == 'discriminator':
                gan_metric.update_metric(
                    output_on_real_images=real_output, output_on_fake_images=fake_output)

            if gan_metric.GAN_module == 'generator':
                gan_metric.update_metric(disc_output=disc_output)

            if gan_metric.GAN_module == 'generic_metric':

                if gan_metric.name == "Latent Matching Loss":
                    gan_metric.update_metric(latent_matching_loss)
                
                if gan_metric.name == "Categorical Loss":
                    gan_metric.update_metric(class_loss)
                
                if gan_metric.name == "accuracy":
                    gan_metric.update_metric( labels, tf.math.argmax( class_output, axis = -1, output_type= tf.int32) )
                    
                    
