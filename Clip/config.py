import torch
#这个文件主要用来存放我们的一些的配置文件，batch size等等


class CFG:
    debug = False
    image_path = "./flickr30k_images/flickr30k_images"
    captions_path = "."
    batch_size = 32
    num_workers = 4
    head_lr = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    augment = True
    model_name = 'resnet50'
    image_embedding = 2048
    text_encoder_model = "./textencoder_model"
    text_embedding = 768
    text_tokenizer = "./textencoder_model"
    max_length = 200

    pretrained = True # for both image encoder and text encoder
    trainable = True # for both image encoder and text encoder
    temperature = 1.0

    # image size
    size = 224

    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 256
    dropout = 0.1