import utilities as utilities
import numpy as np
import tensorflow as tf
import news_20_preprocessing as n20p

batch_size = 200
maximum_document_length = 100
importance_sampling_size = 200
number_of_topics = 50
epochs = 200
DATA_FILE_PATH_20NEWS = "C:/UCL/Thesis/Python/"

x_input = tf.placeholder(tf.int32, shape=[None, maximum_document_length])
x_batch_size = tf.placeholder(tf.int32)
annealed_tau = tf.placeholder(tf.float32)
annealed_learning_rate = tf.placeholder(tf.float32)
global_step = tf.Variable(0, trainable=False)
training = tf.placeholder_with_default(False, shape=(), name='training')

class lda_vae(object):

    def __init__(self, data, embedding_size=25, hidden_layer_encoder=(512, 512), importance_sampling_size=importance_sampling_size, T=number_of_topics, maximum_document_length=maximum_document_length):
        self.T = T
        self.embedding_size = embedding_size
        self.hidden_layer_encoder = hidden_layer_encoder
        self.V = data.data_training.shape[1]
        self.maximum_document_length = maximum_document_length
        self.sampling_size = importance_sampling_size

        self.dropout_rate = tf.placeholder(tf.float32)

        self.alpha = 1.0 * np.ones((1, int(self.T))).astype(np.float32)
        self.mu2 = tf.constant((np.log(self.alpha).T - np.mean(np.log(self.alpha), 1)).T)
        self.var2 = tf.constant((((1.0 / self.alpha) * (1 - (2.0 / self.T))).T + (1.0 / (self.T * self.T)) * np.sum(1.0 / self.alpha, 1)).T)

        tf.set_random_seed(986)

        self.weights = {
            'input_to_embeddings': tf.Variable(tf.random_uniform([self.V, self.embedding_size], -1.0, 1.0), name='encoder'),
            'embeddings_to_h0_weights': tf.Variable(utilities.weights_initialisation(self.embedding_size * self.maximum_document_length, self.hidden_layer_encoder[0]), name='encoder'),
            'h0_to_h1_weights': tf.Variable(utilities.weights_initialisation(self.hidden_layer_encoder[0], self.hidden_layer_encoder[1]), name='encoder'),
            'h1_to_z_latent_weights': tf.Variable(utilities.weights_initialisation(self.hidden_layer_encoder[0], self.maximum_document_length * self.T), name='encoder'),
            'h1_to_mu_weights': tf.Variable(utilities.weights_initialisation(self.hidden_layer_encoder[1], self.T), name='encoder'),
            'h1_to_sigma_squared_weights': tf.Variable(utilities.weights_initialisation(self.hidden_layer_encoder[1], self.T), name='encoder'),
            'beta_weights': tf.Variable(utilities.weights_initialisation(self.T, self.V), name='decoder')}

        self.biases = {
            'embeddings_to_h0_biases': tf.Variable(utilities.biases_initialisation(self.hidden_layer_encoder[0]), name='encoder'),
            'h0_to_h1_biases': tf.Variable(utilities.biases_initialisation(self.hidden_layer_encoder[1]), name='encoder'),
            'h1_to_z_latent_biases': tf.Variable(utilities.biases_initialisation(self.maximum_document_length * self.T), name='encoder'),
            'h1_to_mu_biases': tf.Variable(utilities.biases_initialisation(self.T), name='encoder'),
            'h1_to_sigma_squared_biases': tf.Variable(utilities.biases_initialisation(self.T), name='encoder'),
            'beta_biases': tf.Variable(utilities.biases_initialisation(self.V), name='decoder')}

    def expand_mask(self, x_mask_flat, new_dim=1):

        x_mask = tf.expand_dims(x_mask_flat, dim=2)
        x_mask = tf.tile(x_mask, [1, 1, new_dim])

        return x_mask

    def convert_input_to_one_hot(self, x_input):

        one_hot_tensor = tf.one_hot(x_input, depth=self.V, on_value=1, off_value=0, axis=-1)
        one_hot_tensor = tf.cast(one_hot_tensor, dtype=tf.float32)
        x_document_length = tf.count_nonzero(x_input, axis=1)
        x_mask = tf.cast(tf.sequence_mask(x_document_length, self.maximum_document_length), dtype=tf.float32)
        one_hot_tensor = tf.multiply(one_hot_tensor, self.expand_mask(x_mask, self.V))

        return one_hot_tensor, x_mask, tf.to_float(x_document_length)

    def encoder(self, x_input):

        embed = tf.nn.embedding_lookup(self.weights['input_to_embeddings'], x_input)
        h0 = (tf.matmul(tf.contrib.layers.flatten(embed), self.weights['embeddings_to_h0_weights']) + self.biases['embeddings_to_h0_biases'])
        h0_activation = tf.nn.tanh(h0)
        h1 = (tf.matmul(h0_activation, self.weights['h0_to_h1_weights']) + self.biases['h0_to_h1_biases'])
        h1_activation = tf.nn.softplus(h1)
        h_layer_do = tf.layers.dropout(h1_activation, self.dropout_rate, training=training)

        with tf.device('/gpu:0'):
            z_logits = (tf.matmul(h_layer_do, self.weights['h1_to_z_latent_weights']) + self.biases['h1_to_z_latent_biases'])
            z_logits = tf.reshape(z_logits, [-1, self.T])

        with tf.device('/gpu:1'):
            h_mu = (tf.matmul(h_layer_do, self.weights['h1_to_mu_weights']) + self.biases['h1_to_mu_biases'])
            h_log_sigma_squared = (tf.matmul(h_layer_do, self.weights['h1_to_sigma_squared_weights']) + self.biases['h1_to_sigma_squared_biases']) # tf.contrib.layers.layer_norm()

        return z_logits, h_mu, h_log_sigma_squared

    def form_2_decoder(self, x_input_one_hot, sampled_z_list, x_mask):

        log_P_x_given_z_list = []
        total_x_input = tf.reduce_sum(x_input_one_hot, 1)

        for i in range(self.sampling_size):
            sampled_z_list_temp = tf.multiply(tf.reshape(sampled_z_list[i], [-1, self.maximum_document_length, self.T]), self.expand_mask(x_mask, self.T))
            total_z = tf.reduce_sum(sampled_z_list_temp, 1)
            log_x_reconstruction = tf.nn.log_softmax(tf.matmul(total_z, self.weights['beta_weights']) + self.biases['beta_biases'])
            log_P_x_given_z = tf.reduce_sum(total_x_input * log_x_reconstruction, 1)
            log_P_x_given_z_list.append(log_P_x_given_z)

        return log_P_x_given_z_list

    def form_1_decoder(self, x_input_one_hot, sampled_z_list, x_mask):

        log_P_x_given_z_list = []
        for i in range(self.sampling_size):
            log_x_reconstruction_padded = tf.log(tf.matmul(sampled_z_list[i], tf.nn.softmax(self.weights['beta_weights'])) + 1e-20)
            log_x_reconstruction = tf.multiply(tf.reshape(log_x_reconstruction_padded, [-1, self.maximum_document_length, self.V]), self.expand_mask(x_mask, self.V))
            log_P_x_given_z = tf.reduce_sum(x_input_one_hot * log_x_reconstruction, [1, 2])
            log_P_x_given_z_list.append(log_P_x_given_z)

        return log_P_x_given_z_list

    def model(self, x_input, annealed_tau):

        x_input_one_hot, x_mask, x_document_length = self.convert_input_to_one_hot(x_input)
        z_logits, h_mu, h_sigma_squared = self.encoder(x_input)
        unit_gumbel_sample = utilities.sample_gumbel([x_batch_size * self.maximum_document_length * self.sampling_size, self.T], epsilon=1e-20)
        sampled_z_list = []
        unit_gumbel_sample = tf.split(unit_gumbel_sample, self.sampling_size, axis=0)
        for i in range(self.sampling_size):
            sampled_z = utilities.gumbel_softmax_sample(z_logits, unit_gumbel_sample[i], annealed_tau)
            sampled_z_list.append(sampled_z)
        pi_of_z = tf.reshape(tf.nn.softmax(z_logits), [-1, self.maximum_document_length, self.T])
        log_P_x_given_z_list = self.form_1_decoder(x_input_one_hot, sampled_z_list, x_mask)

        return log_P_x_given_z_list, sampled_z_list, pi_of_z, h_mu, h_sigma_squared, x_mask

    def kl(self, pi_of_z, x_mask, h_mu, h_log_sigma_squared, sampled_z_list):

        epsilon = 1e-20
        gaussian_KL_matrix = 0.5 * (tf.div(tf.exp(h_log_sigma_squared), self.var2) + tf.multiply(tf.div((self.mu2 - h_mu), self.var2),
                                                    (self.mu2 - h_mu)) - 1 + tf.log(self.var2) - h_log_sigma_squared)
        unit_gaussian_sample = tf.random_normal([x_batch_size * self.sampling_size, self.T], 0, 1, dtype=tf.float32)
        kl_list = []
        sampled_kl_divergence_list = []
        unit_gaussian_sample = tf.split(unit_gaussian_sample, self.sampling_size, axis=0)

        for i in range(self.sampling_size):
            h_sample = h_mu + tf.exp(0.5 * h_log_sigma_squared) * unit_gaussian_sample[i]
            q_alpha = tf.nn.softmax(h_sample)
            log_alpha_z = tf.log(tf.reduce_sum(tf.multiply(tf.tile(tf.expand_dims(q_alpha, dim=1), [1, self.maximum_document_length, 1]),
                           tf.reshape(sampled_z_list[i], [-1, self.maximum_document_length, self.T])), axis=2) + epsilon)
            log_pi_z = tf.log(tf.reduce_sum(tf.multiply(pi_of_z, tf.reshape(sampled_z_list[i], [-1, self.maximum_document_length, self.T])), axis=2) + epsilon)
            sampled_kl_divergence = tf.reduce_sum(tf.multiply(log_pi_z - log_alpha_z, x_mask), axis=1)
            kl = sampled_kl_divergence + tf.reduce_sum(gaussian_KL_matrix, axis=1)
            sampled_kl_divergence_list.append(sampled_kl_divergence)
            kl_list.append(kl)

        return kl_list

    def train(self, data, number_of_epochs=epochs, test_frequency=5):

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        sorted_tuples = sorted(data.vocabulary.items(), key=lambda x: x[1])
        vocab_list = []
        for i, j in sorted_tuples:
            vocab_list.append(i)

        tau = 0.20
        minimum_tau = 0.20
        annealing_rate = 0.007

        beta_probabilities = tf.nn.softmax(self.weights['beta_weights'])

        learning_date_boundaries = [5000, 15000, 30000]
        learning_rate_vector = [0.002, 0.001, 0.0005, 0.0002]
        learning_rate = tf.train.piecewise_constant(global_step, learning_date_boundaries, learning_rate_vector)

        log_P_x_given_z_list, sampled_z_list, pi_of_z, h_mu, h_log_sigma_squared, x_mask = self.model(x_input, annealed_tau)
        kl_list = self.kl(pi_of_z, x_mask, h_mu, h_log_sigma_squared, sampled_z_list)

        log_weights_list = []
        for i in range(self.sampling_size):
            log_weights_list.append(tf.expand_dims(log_P_x_given_z_list[i] - kl_list[i], 1))
        logF = tf.concat(log_weights_list, axis=1)
        wmax = tf.reduce_max(logF , 1, keep_dims=True)
        logF_matrix = logF - wmax
        logF_normalizer = tf.clip_by_value(tf.reduce_sum(tf.exp(logF_matrix), 1, keep_dims=True), 1e-9, np.inf)
        logF_normalizer = tf.log(logF_normalizer)
        log_weights = logF_matrix - logF_normalizer
        weights = tf.stop_gradient(tf.exp(log_weights))
        loss_function =  wmax + logF_normalizer - tf.log(tf.cast(self.sampling_size, dtype=tf.float32))

        elbo = tf.reduce_mean(loss_function)

        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        gradients = tf.gradients(-logF * weights , params)
        training_optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.90, beta2=0.999).apply_gradients(zip(gradients, params), global_step=global_step)

        total_batches = int(data.number_samples_training / batch_size)

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(number_of_epochs):
                minibatches = utilities.create_minibatch(data.data_ordered_words_training.astype('int16'), batch_size)
                epoch_train_loss = 0.
                running_loss_per_doc = []
                running_nd = []
                for i in range(total_batches):
                    batch_xs, _ixs = minibatches.__next__()

                    _, _elbo, _loss_function, beta = sess.run([training_optimiser, elbo, loss_function, beta_probabilities],
                         feed_dict={x_input: batch_xs, annealed_tau: tau, x_batch_size: batch_size, self.dropout_rate: 0.20, training: True})

                    epoch_train_loss += _elbo / total_batches
                    n_d = np.expand_dims(np.count_nonzero(batch_xs, axis=1), 1)
                    running_loss_per_doc.append(-_loss_function)
                    running_nd.append(n_d)
                doc_perplexity = np.exp(np.mean(np.array(running_loss_per_doc) / np.array(running_nd)))
                corpus_perplexity = np.exp(np.sum(np.array(running_loss_per_doc)) / np.sum(np.array(running_nd)))
                print('Epoch', epoch, 'tau', tau, 'learning rate', learning_rate.eval(), 'with a training loglikelihood of', epoch_train_loss,
                      ' with document perplexity of', doc_perplexity, 'and corpus perplexity of', corpus_perplexity)
                tau = np.maximum(tau * np.exp(-annealing_rate), minimum_tau)
                if (epoch % test_frequency == 0 and epoch != 0) or epoch == number_of_epochs - 1:
                    test_batch_size = np.min([batch_size, data.data_ordered_words_test.shape[0]])
                    test_running_loss_per_doc = []
                    test_running_nd = []
                    epoch_test_loss = 0.
                    total_batches = int(data.number_samples_test / test_batch_size)
                    for doc_count in range(data.data_ordered_words_test.shape[0] // test_batch_size):
                        doc = data.data_ordered_words_test[doc_count * test_batch_size:(doc_count + 1) * test_batch_size, :]
                        with tf.device('/gpu:1'):
                            _elbo, _loss_function = sess.run([elbo, loss_function], feed_dict={x_input: doc, annealed_tau: minimum_tau, x_batch_size: test_batch_size, self.dropout_rate: 0.0, training: False})
                        n_d = np.expand_dims(np.count_nonzero(doc, axis=1), 1)
                        test_running_loss_per_doc.append(-_loss_function)
                        test_running_nd.append(n_d)
                        epoch_test_loss += _elbo/total_batches
                    doc_perplexity = np.exp(np.mean(np.array(test_running_loss_per_doc) / np.array(test_running_nd)))
                    corpus_perplexity = np.exp(np.sum(np.array(test_running_loss_per_doc)) / np.sum(np.array(test_running_nd)))
                    print('The test set ELBO is: ', epoch_test_loss)
                    print('The test set document perplexity is: ', doc_perplexity, 'and the corpus perplexity is', corpus_perplexity)
            utilities.print_top_words(beta, vocab_list)

L = n20p.news_20_preprocessing(DATA_FILE_PATH_20NEWS, maximum_document_length)
L.remove_long_documents()
L.ordered_document(random_order_flag=True)

lda = lda_vae(L)
lda.train(L)
