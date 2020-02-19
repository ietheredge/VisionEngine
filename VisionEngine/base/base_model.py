class BaseModel(object):
    def __init__(self, config):
        self.config = config
        self.model = None

    # save the checkpoint to the path defined in the config file
    def save(self, checkpoint_path):
        if self.model is None:
            raise Exception("You need to build the model first.")

        print("Saving model...")
        self.model.save_weights(checkpoint_path)
        print("Model saved")

    # load the experiment from the path defined in the config file
    def load(self, checkpoint_path):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError
