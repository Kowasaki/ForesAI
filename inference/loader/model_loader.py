class ModelLoader:
    """
    The ModelLoader interface defines the functions which all loader classes needs to implement
    """
    def __init__(self, model_config):
        raise NotImplementedError

    def inference(self, input):
        raise NotImplementedError

