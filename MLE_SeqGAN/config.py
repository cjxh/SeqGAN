class GenConfig():
    def __init__(self):
        self.EMB_DIM = 300
        self.HIDDEN_DIM = 150
        self.SEQ_LENGTH = 20
        self.START_TOKEN = 0

        self.PRE_EPOCH_NUM = 350  # change the pre-train epoch here
        self.TRAIN_ITER = 1  # generator
        self.SEED = 88
        self.BATCH_SIZE = 64

class DisConfig():
    def __init__(self):
        self.DIS_EMBEDDING_DIM = 64
        self.DIS_FILTER_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
        self.DIS_NUM_FILTERS = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
        self.DIS_DROPOUT_KEEP_PROB = 0.75
        self.DIS_12_REG_LAMBDA = 0.2

        # Training parameters
        self.DIS_BATCH_SIZE = 64
        self.DIS_NUM_EPOCHS = 3
        self.DIS_ALTER_EPOCH = 50
        
