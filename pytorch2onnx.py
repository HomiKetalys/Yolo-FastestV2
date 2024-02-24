import argparse
import os
import sys

import torch

from model import detector
from utils import utils

def pytorch2onnx_main(cfg_path, weight_path, onnx_save_path):
    cfg = utils.load_datafile(cfg_path)
    separation = cfg["separation"]
    separation_scale = cfg["separation_scale"]
    model = detector.DetectorSp(cfg["classes"], cfg["anchor_num"], 2, separation=separation,
                                   separation_scale=separation_scale)
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.eval()
    if separation:
        model.export_onnx = 1
        test_data = torch.rand(1, 3, cfg["height"] // separation_scale, cfg["width"] // separation_scale)
        torch.onnx.export(model,  # model being run
                          test_data,  # model input (or a tuple for multiple inputs)
                          os.path.join(onnx_save_path, 'model_front.onnx'),
                          # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=11,  # the ONNX version to export the model to
                          do_constant_folding=True)  # whether to execute constant folding for optimization
        model.export_onnx = 2
        test_data = torch.rand(1, model.backbone.stage_out_channels[separation - 1], cfg["height"] // 2 ** separation,
                               cfg["width"] // 2 ** separation)
        torch.onnx.export(model,  # model being run
                          test_data,  # model input (or a tuple for multiple inputs)
                          os.path.join(onnx_save_path, 'model_post.onnx'),
                          # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=11,  # the ONNX version to export the model to
                          do_constant_folding=True)  # whether to execute constant folding for optimization
        # model.export_onnx = 3
        # test_data = torch.rand(1, 3, cfg["height"] , cfg["width"]).to(device)
        # torch.onnx.export(model,  # model being run
        #                   test_data,  # model input (or a tuple for multiple inputs)
        #                   os.path.join(onnx_save_path, 'model.onnx'),
        #                   # where to save the model (can be a file or file-like object)
        #                   export_params=True,  # store the trained parameter weights inside the model file
        #                   opset_version=11,  # the ONNX version to export the model to
        #                   do_constant_folding=True)  # whether to execute constant folding for optimization
    else:
        model.export_onnx = 1
        test_data = torch.rand(1, 3, cfg["height"], cfg["width"])
        torch.onnx.export(model,  # model being run
                          test_data,  # model input (or a tuple for multiple inputs)
                          os.path.join(onnx_save_path, 'model.onnx'),
                          # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=11,  # the ONNX version to export the model to
                          do_constant_folding=True)  # whether to execute constant folding for optimization


    

