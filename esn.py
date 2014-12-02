import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, csc_matrix
from exceptions import NotImplementedError, StopIteration

class ESN(object):
    """Methods for training and running an echo state network
    """

    def __init__(self, n_neurons, n_input, n_output):
        self.n_neurons = n_neurons
        self.n_input = n_input
        self.n_output = n_output
        # recurrent weights
        self.W = None
        self.W_shp = (self.n_neurons, self.n_neurons)
        # input weights
        self.Q = None
        self.Q_shp = (self.n_neurons, self.n_input)
        # output weights
        self.R = None
        self.R_shp = (self.n_output, self.n_neurons)
        # output bias
        self.b = np.zeros(self.n_output)
        # decay rate
        self.r_decay = 1e-5
        # learning rate
        self.l_rate = 1e-3
        # network state
        self.y = np.zeros(self.n_neurons)
        # input vector
        self.x = np.zeros(self.n_input)
        # sparsity of W
        self.w_sparsity = 10. / n_neurons
        # number of train iterations in a training epoch
        self.epoch_length = 10000
        self.epochs = 10
        # priming iteratoions
        self.n_prime = 100

        #finally, initialize the weights
        self.init_network()

    def init_network(self):
        """Initializes the network parameters
        """
        mask = np.random.randn(*self.W_shp) < self.w_sparsity
        # enforce sparsity constraint
        self.W = np.random.randn(*self.W_shp)*mask
        # scale the weights by 1/(sqrt(# of presynaptic neurons)
        scale = 1. / np.sqrt(np.sum(mask, axis=1))
        self.W = np.dot(self.W.T, np.diag(scale)).T
        # set input and output weights
        self.Q = np.random.randn(*self.Q_shp)
        self.R = np.random.randn(*self.R_shp)
        # sparsify everything
        self.W = csr_matrix(self.W)
        self.Q = csr_matrix(self.Q)
        #self.R = csr_matrix(self.R)
        #self.b = csr_matrix(self.b)
        
    def step_recurrent(self):
        """Runs the current dynamics for one timestep
        """
        self.y = np.tanh(self.W.dot(self.y) + self.Q.dot(self.x))

    def get_output(self):
        """Returns the softmaxed output
        """
        z = self.R.dot(self.y) + self.b
        return np.exp(z) / np.sum(np.exp(z))

    def next_input(self):
        """Sets the next input vector of size (n_input,) to self.x
        To be implemented by inheriting class
        """
        raise NotImplementedError()

    def reset_input(self):
        """Reset the input iterator so that training epochs can be repeated
        To be implemented by inheriting class
        """
        raise NotImplementedError()

    def train(self, n_epochs=None):
        """Trains the network for the specified number of epochs
        """
        n_epochs = n_epochs or self.epochs
        for e_itr in xrange(n_epochs):
            print "Training epoch number {}".format(e_itr)
            self.next_input()
            for _ in xrange(self.epoch_length):
                try:
                    # faster to do this inline
                    self.y = np.tanh(self.W.dot(self.y) + self.Q.dot(self.x))
                    q = self.get_output()
                    self.next_input()

                    # learning rule
                    R_grad = np.outer((self.x - q), self.y)# - self.r_decay * self.R
                    b_grad = self.x - q
                    self.R = self.R +  self.l_rate*R_grad
                    self.b = self.b + 10*self.l_rate*b_grad
                except StopIteration:
                    self.reset_input()

    def run_network(self, itr, prime=True):
        """Primes the network with n_prime input and then runs it for itr iterations.
        For every iteration appends the softmaxed output to a list and returns the list
        """
        output = []
        state = []
        assert self.n_output == self.n_input
        if prime:
            self.prime_network(500)

            
        for _ in xrange(itr):
            q = self.get_output()
            # breaks abstraction
            # chooses state according to the probability distribution
            new_state = np.where(np.cumsum(q) > np.random.random())[0][0]
            self.x = np.zeros(self.n_input)
            self.x[new_state] = 1
            self.step_recurrent()
            state.append(self.y)
            output.append(self.x)

        return output, np.array(state)

    def prime_network(self, itr):
        """Primes the network with itr inputs
        """
        self.reset_input()
        for _ in xrange(self.n_prime):
            self.next_input()
            self.step_recurrent()

    def plot_state(self, itr, n_nodes=5):
        """Selects n_nodes at random from the network state and plots their
        time series activations
        """
        plt.ion()
        nodes = np.random.randint(self.n_neurons, size=n_nodes)
        output, state = self.run_network(itr, False)
        activations = state[:, nodes]
        t = np.linspace(1, itr, num=itr)
        for y_series in activations.T:
            plt.plot(t, y_series)
        plt.title("Activation of {} randomly chosen nodes over time".format(n_nodes))
        plt.xlabel("Iteration")
        plt.ylabel("Activation")


class CharESN(ESN):
    """ESN that implements character prediction
    """

    def __init__(self, n_neurons, path):
        self.path = path
        self.char_to_int = {}
        self.text = ""
        self.read_in()
        self.int_to_char = {i : char for (char, i) in self.char_to_int.iteritems()}
        self.n_alphabet = len(self.char_to_int)
        ESN.__init__(self, n_neurons, self.n_alphabet, self.n_alphabet)
        self.reset_input()


    def read_in(self):
        """Reads in the text while building the char dictionary
        """
        with open(self.path, 'r') as lewis:
            for line in lewis:
                for char in line.lower():
                    self.text += char
                    if char not in self.char_to_int:
                        curr_len = len(self.char_to_int)
                        self.char_to_int[char] = curr_len

    def vec_gen(self):
        """Generator for input. next returns the vector corresponding to the next character
        in the text
        """
        for char in self.text:
            vec = np.zeros(self.n_input)
            vec[self.char_to_int[char]] = 1
            yield vec

    def reset_input(self):
        """Resets the char vector generator
        """
        self.input_generator = self.vec_gen()

    def next_input(self):
        self.x = self.input_generator.next()

    def generate_text(self, n_chars, prime=True):
        """Primes the network and then generates n_chars of text
        """
        outputs, state = self.run_network(n_chars, prime)
        gen_text = ""
        for out in outputs:
            gen_text += self.int_to_char[np.argmax(out)]
        print gen_text

class WordESN(ESN):
    """ESN that implements word prediction
    """

    def __init__(self, n_neurons, path):
        self.path = path
        self.word_to_int = {}
        self.text = []
        self.read_in()
        self.int_to_word = {i : word  for (word, i) in self.word_to_int.iteritems()}
        self.n_alphabet = len(self.word_to_int)
        ESN.__init__(self, n_neurons, self.n_alphabet, self.n_alphabet)
        self.reset_input()


    def read_in(self):
        """Reads in the text while building the char dictionary
        """
        with open(self.path, 'r') as lewis:
            for line in lewis:
                line_list = []
                for word in self.parse_line(line):
                    line_list.append(word)
                    if word not in self.word_to_int:
                        curr_len = len(self.word_to_int)
                        self.word_to_int[word] = curr_len
                self.text.append(line_list)

    def parse_line(self, line):
        """Splits a line into its atomic components
        """
        split_on_space = line.lower().split()
        parsed = []
        for tok in split_on_space:
            first_alpha = 0
            last_alpha = len(tok) + 1
            for ind in xrange(len(tok)):
                if tok[ind].isalpha():
                    first_alpha = ind
                    break
            for ind in reversed(xrange(len(tok))):
                if tok[ind].isalpha():
                    last_alpha = ind
                    break
            tok_split = [tok[0:first_alpha],
                         tok[first_alpha:last_alpha+1],
                         tok[last_alpha+1:len(tok) + 1]
                     ]
            for subtok in tok_split:
                if len(subtok) > 0:
                    parsed.append(subtok)
        parsed.append("\n")
        return parsed
        

    def vec_gen(self):
        """Generator for input. next returns the vector corresponding to the next character
        in the text
        """
        for line in self.text:
            for word in line:
                vec = np.zeros(self.n_input)
                vec[self.word_to_int[word]] = 1
                yield vec

    def reset_input(self):
        """Resets the char vector generator
        """
        self.input_generator = self.vec_gen()

    def next_input(self):
        self.x = self.input_generator.next()

    def generate_text(self, n_chars, prime=True):
        """Primes the network and then generates n_chars of text
        """
        outputs, state = self.run_network(n_chars, prime)
        gen_text = ""
        for out in outputs:
            gen_text += self.int_to_word[np.argmax(out)]
            gen_text += " "
        print gen_text
        

alice = CharESN(750, 'alice_in_wonderland.txt')
lewis = WordESN(500, 'alice_in_wonderland.txt')