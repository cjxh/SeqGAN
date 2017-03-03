class GenConfig():
    def __init__(self):
        self.EMB_DIM = 32
        self.HIDDEN_DIM = 32
        self.SEQ_LENGTH = 20
        self.START_TOKEN = 0

        self.PRE_EPOCH_NUM = 350  # change the pre-train epoch here
        self.TRAIN_ITER = 1  # generator
        self.SEED = 88
        self.BATCH_SIZE = 64