import torch
import torch.backends.cudnn as cudnn
import zoo


def build_model(model_name, data_name):
    if model_name == "sstn":
        if data_name == "ip":
            model = zoo.SSNet_AEAE_IP()
        elif data_name == "up":
            model = zoo.SSNet_AEAE_UP()
        elif data_name == "sa":
            model = zoo.SSNet_AEAE_SA()
    elif model_name == "hybrid":
        bands = 30
        if data_name == "ip" or data_name == "sa":
            labels = 16
        elif data_name == "up":
            labels = 9
        model = zoo.HybridSN(bands, labels)
    elif model_name == "cnn3d":
        if data_name == "ip":
            bands = 200
            labels = 16
        elif data_name == "up":
            bands = 103
            labels = 9
        elif data_name == "sa":
            bands = 204
            labels = 16
        model = zoo.cnn3d(bands, labels, patch_size=5)
    elif model_name == "cnn1d":
        if data_name == "ip":
            bands = 200
            labels = 16
        elif data_name == "up":
            bands = 103
            labels = 9
        elif data_name == "sa":
            bands = 204
            labels = 16
        model = zoo.cnn1d(bands, labels)
    else:
        raise ValueError("This model is not supported.")

    cudnn.benchmark = True
    cudnn.enabled = True
    return model

