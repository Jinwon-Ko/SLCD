def get_model(cfg):
    from models.networks.model import Detector
    model = Detector(cfg)
    return model
