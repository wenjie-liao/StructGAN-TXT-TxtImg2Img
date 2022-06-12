import torch

def create_model(opt):
    from .txtimg2img_model import TxtImg2ImgModel, InferenceModel
    if opt.isTrain:
        model = TxtImg2ImgModel()
    else:
        model = InferenceModel()
    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids) and not opt.fp16:
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model
