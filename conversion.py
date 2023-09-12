import argparse
import json
import os
import pathlib
import onnx
import torch
# from model_pool import get_model
from models.model_pool import ModelwEmb
from models import resnext

# pip install --pre torch torchaudio torchvision --index-url https://download.pytorch.org/whl/nightly/cu121


def parse_cfg(path):
    if os.path.isfile(path):
        with open(path, "r") as f:
            config = json.load(f)
            return config

    raise FileNotFoundError("Not found config file.")


def main():
    parser = argparse.ArgumentParser(
        description="Image classification model conversion"
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="maxvit_base_tf_224.in21k",
        help="Select architecture model for conversion",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        # required=True,
        default=r"./log/mosquito_resnext50_batchsize_8_epochs_100_temp_0.07_lr_0.001_sim-sim/bcl_ckpt.best.pth.tar",
        help="Trained model information checkpoint directory",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Image size for image classification inference task",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=6,
        help="Number of classes in image classification inference task",
    )
    parser.add_argument(
        "--onnx_output_dir",
        type=str,
        default="./export_onnx",
        help="Output directory contain raw model and onnx model",
    )
    parser.add_argument(
        "--fp16",
        type=bool,
        required=False,
        default=False,
        help="onnx model type",
    )
    parser.add_argument(
        "--opset",
        type=int,
        required=False,
        default=17,
        help="onnx opset version",
    )
    parser.add_argument(
        "--dynamic_batch_size",
        type=bool,
        required=False,
        default=False,
        help="onnx opset version",
    )

    args = parser.parse_args()
    checkpoint_dir = args.checkpoint_dir
    # config_path = os.path.join(checkpoint_dir, "config_model.json")
    # model_path = os.path.join(checkpoint_dir, f"best_{args.accuracy_or_f1}.h5")
    print("Checkpoint Path:\n", checkpoint_dir)
    onnx_output_dir = args.onnx_output_dir

    # config = parse_cfg(config_path)
    num_classes = args.num_classes
    arch = args.arch
    # config["fp16"] = args.fp16
    is_fp16 = args.fp16
    dymamic_batch_size = args.dynamic_batch_size
    # config["dynamic_batch_size"] = args.dynamic_batch_size
    image_size = args.image_size
    # model = get_model(
    #     num_classes, arch, pretrained=False, print_model=False, size=image_size
    # )

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'resnet50':
        model = resnext.BCLModel(name='resnet50', num_classes=num_classes, feat_dim=args.feat_dim,
                                 use_norm=args.use_norm)
    elif args.arch == 'resnext50':
        model = resnext.BCLModel(name='resnext50', num_classes=num_classes, feat_dim=args.feat_dim,
                                 use_norm=args.use_norm)
    elif str(args.arch).startswith("convnext") or str(args.arch).startswith("maxvit"):
        feat_dim = 1024 if str(args.arch).startswith("convnext") else 768
        model = ModelwEmb(
            num_classes=num_classes,
            arch=args.arch,
            pretrained=True,
            feat_dim=feat_dim
        )
    else:
        raise NotImplementedError('This model is not supported')

    print("Loading model weight...")
    checkpoint = torch.load(checkpoint_dir, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    # make directory to save onnx model
    pathlib.Path(onnx_output_dir).mkdir(parents=True, exist_ok=True)
    if is_fp16:
        x = torch.randn((1, 3, image_size, image_size), requires_grad=True).half()
        model = model.half()
    else:
        x = torch.randn((1, 3, image_size, image_size), requires_grad=True)

    onnx_model_path = os.path.join(
        onnx_output_dir, args.arch + ".onnx"
    )
    # model.cuda()
    torch.onnx.export(
        model,
        x,
        onnx_model_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        if dymamic_batch_size
        else None,
    )
    print("Done.")
    print("Start check onnx...")
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    print("Done.")


if __name__ == "__main__":
    main()
