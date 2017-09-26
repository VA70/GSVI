import pickle
import numpy as np
import utilities as utilities

class news_20_preprocessing(object):

    def __init__(self, DATA_FILE_PATH_20NEWS, maximum_document_length, minimum_document_length=0):

        self.length_threshold = maximum_document_length
        self.min_threshold = minimum_document_length
        dataset_tr = 'train.txt.npy'
        self.data_training = np.load(DATA_FILE_PATH_20NEWS + dataset_tr, encoding='latin1')
        dataset_te = 'test.txt.npy'
        self.data_test = np.load(DATA_FILE_PATH_20NEWS + dataset_te, encoding='latin1')
        vocab = 'vocab_unix.pkl'
        self.vocabulary = pickle.load(open(DATA_FILE_PATH_20NEWS+vocab,'rb'))

        self.vocab_size = len(self.vocabulary)
        # --------------convert to one-hot representation------------------
        print('Converting data to one-hot representation')
        self.data_training = np.array([utilities.onehot(doc.astype('int'), self.vocab_size) for doc in self.data_training if np.sum(doc) != 0], dtype=np.int16)
        self.data_test = np.array([utilities.onehot(doc.astype('int'), self.vocab_size) for doc in self.data_test if np.sum(doc) != 0], dtype=np.int16)
        # --------------print the data dimentions--------------------------
        print('Data Loaded')
        print('Dim Training Data', self.data_training.shape)
        print('Dim Test Data', self.data_test.shape)

        self.number_samples_training = self.data_training.shape[0]
        self.number_samples_test = self.data_test.shape[0]
        self.training_document_length = np.sum(self.data_training, axis=1, keepdims=True)
        self.test_document_length = np.sum(self.data_test, axis=1, keepdims=True)

    def remove_long_documents(self):

        row_indices_training = np.where(np.logical_and(self.training_document_length <= self.length_threshold, self.training_document_length >= self.min_threshold))
        self.data_training = self.data_training[row_indices_training[0], :]
        self.training_document_length = self.training_document_length[row_indices_training[0], :]
        self.training_maximum_document_length = np.max(self.training_document_length)
        self.number_samples_training = self.data_training.shape[0]
        print(self.number_samples_training)

        row_indices_test = np.where(np.logical_and(self.test_document_length <= self.length_threshold, self.test_document_length >= self.min_threshold))
        self.data_test = self.data_test[row_indices_test[0], :]
        self.test_document_length = self.test_document_length[row_indices_test[0], :]
        self.test_maximum_document_length = np.max(self.test_document_length)
        self.number_samples_test = self.data_test.shape[0]
        print(self.number_samples_test)

    def ordered_document(self, random_order_flag=False):

        number_of_docs = self.data_training.shape[0]
        self.data_ordered_words_training = np.zeros((number_of_docs, self.length_threshold), dtype=np.int16)

        S_train = np.sum(self.data_training, 0) / np.sum(self.data_training)
        print('Training set entropy :', -np.sum(S_train * np.log(S_train + 1e-20)), ' corresponds to perplexity of :',
              np.exp(-np.sum(S_train * np.log(S_train + 1e-20))))

        for d in range(number_of_docs):
            non_zero_values = np.nonzero(self.data_training[d, :])

            for columns in non_zero_values:
                number_instances = self.data_training[d, columns]
            word_occurence_tuple = zip(columns, number_instances)

            word_position = 0
            for word_index, number_instances in word_occurence_tuple:
                if number_instances == 1:
                    self.data_ordered_words_training[d, word_position] = word_index
                else:
                    self.data_ordered_words_training[d, word_position:word_position + number_instances] = word_index
                word_position += number_instances

            if random_order_flag:
                np.random.seed(42)
                random_permutation = np.random.permutation(word_position)
                self.data_ordered_words_training[d, 0:word_position] = self.data_ordered_words_training[d, random_permutation]

        number_of_docs = self.data_test.shape[0]
        self.data_ordered_words_test = np.zeros((number_of_docs, self.length_threshold), dtype=np.int16)

        S_test = np.sum(self.data_test, 0) / np.sum(self.data_test)
        print('Test set cross-entropy :', -np.sum(S_train * np.log(S_test + 1e-20)), ' corresponds to perplexity of :',
              np.exp(-np.sum(S_train * np.log(S_test + 1e-20))))

        random_permutation = np.random.permutation(self.data_test.shape[0])
        self.data_test = self.data_test[random_permutation, :]
        for d in range(number_of_docs):
            non_zero_values = np.nonzero(self.data_test[d, :])

            for columns in non_zero_values:
                number_instances = self.data_test[d, columns]
            word_occurence_tuple = zip(columns, number_instances)

            word_position = 0
            for word_index, number_instances in word_occurence_tuple:
                if number_instances == 1:
                    self.data_ordered_words_test[d, word_position] = word_index
                else:
                    self.data_ordered_words_test[d, word_position:word_position + number_instances] = word_index
                word_position += number_instances

            if random_order_flag:
                np.random.seed(20)
                random_permutation = np.random.permutation(word_position)
                self.data_ordered_words_test[d, 0:word_position] = self.data_ordered_words_test[d, random_permutation]

    def reduce_vocabulary(self, word_threshold = 500):

        N = self.number_samples_training
        n_i = np.count_nonzero(self.data_training, axis=0)

        idf = np.log(N / n_i + 1e-20)
        idf[np.isinf(idf)] = 0

        tf = np.transpose(self.data_training) / np.sum(self.data_training, axis=1)

        old_err_state = np.seterr(divide='raise')
        ignored_states = np.seterr(**old_err_state)
        tf_mean = np.sum(tf, axis=1) / n_i
        tf_mean[np.isnan(tf_mean)] = 0

        tf_idf = tf_mean * idf

        sorted_tuples = sorted(self.vocabulary.items(), key=lambda x: x[1])
        vocab_list = []
        for i, j in sorted_tuples:
            vocab_list.append(i)

        new_vocab_index = tf_idf.argsort()[:-word_threshold:-1]
        new_vocab_index = np.concatenate([[0], new_vocab_index]) # keep __eos__ at position 0.
        new_vocab = []
        for i in new_vocab_index:
            new_vocab.append(vocab_list[i])

        new_vocabulary_dictionary = dict((wrd, ind[0]) for ind, wrd in np.ndenumerate(new_vocab))

        new_data_training = np.zeros([self.data_training.shape[0], word_threshold]).astype(int)
        new_data_test = np.zeros([self.data_test.shape[0], word_threshold]).astype(int)
        for new_index, old_index in np.ndenumerate(new_vocab_index):
            new_data_training[:, new_index] = np.expand_dims(self.data_training[:, old_index], axis=1)
            new_data_test[:, new_index] = np.expand_dims(self.data_test[:, old_index], axis=1)

        #remove docs with no words
        remove_index_training = np.where(np.sum(new_data_training, axis=1) == 0)
        new_data_training = np.delete(new_data_training, (remove_index_training[0]), axis=0)
        remove_index_test = np.where(np.sum(new_data_test, axis=1) == 0)
        new_data_test = np.delete(new_data_test, (remove_index_test[0]), axis=0)

        self.vocabulary = new_vocabulary_dictionary
        self.vocab_size = word_threshold
        self.data_training = new_data_training
        self.data_test = new_data_test
        self.number_samples_training = self.data_training.shape[0]
        self.number_samples_test = self.data_test.shape[0]
        self.training_document_length = np.sum(self.data_training, axis=1, keepdims=True)
        self.test_document_length = np.sum(self.data_test, axis=1, keepdims=True)