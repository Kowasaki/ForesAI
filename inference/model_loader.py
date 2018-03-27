class ModelLoader:
    def __init__(self, **kwargs):
        # TODO: Make interface for all loaders to inherit
        raise NotImplementedError

    def load_input(self, input):
        raise NotImplementedError

    def get_output(self):
        raise NotImplementedError

