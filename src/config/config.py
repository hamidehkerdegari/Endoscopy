class Config:
    def __init__(self):
        self.num_epochs = 25
        self.batch_size = 32
        self.learning_rate = 0.001
        self.num_classes = 3
        self.input_size = (224, 224)
        self.num_workers = 4
        self.split_ratio = 0.8

config = Config()
