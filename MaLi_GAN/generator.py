import tensorflow as tf

class Generator(object):
    def __init__(self):
        self.gru_cell_size = 100
        

    def build(self):
        self.add_placeholders()
        self.generated_samples = self.generator()
        self.loss = self.add_loss_op(self.samples)
        self.train_op = self.add_training_op(self.loss)
    
    def add_loss_op(self, pred):

    def add_training_op(self, loss):

    def generator(self):
        gru_cell = tf.contrib.rnn.BasicGRUCell(self.gru_cell_size, state_is_tuple=False)
        
        
        

    def update_params(self):

