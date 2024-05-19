def get_model(cfg):
    from models.networks.model import SLCD
    model = SLCD(cfg)
    return model
