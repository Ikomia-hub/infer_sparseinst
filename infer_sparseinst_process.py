# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
from infer_sparseinst import update_path
from ikomia import core, dataprocess
import copy
# Your imports below
from argparse import Namespace
import numpy as np

from detectron2.config import get_cfg
import os

from infer_sparseinst.sparseinst import VisualizationDemo, add_sparse_inst_config
from detectron2.utils.visualizer import GenericMask
from distutils.util import strtobool
import torch
from detectron2.config import CfgNode
from infer_sparseinst.utils import model_zoo, gdrive_download


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferSparseinstParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        # Example : self.windowSize = 25
        self.model_name = "sparse_inst_r50_giam_aug"
        self.conf_thres = 0.5
        self.custom = False
        self.cfg = ""
        self.weights = ""
        self.update = True

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        # Example : self.windowSize = int(param_map["windowSize"])
        self.update = True
        self.conf_thres = float(param_map["conf_thres"])
        self.custom = strtobool(param_map["custom"])
        self.cfg = param_map["cfg"]
        self.weights = param_map["weights"]
        self.model_name = param_map["model_name"]

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {}
        # Example : paramMap["windowSize"] = str(self.windowSize)
        param_map["conf_thres"] = str(self.conf_thres)
        param_map["custom"] = str(self.custom)
        param_map["cfg"] = self.cfg
        param_map["weights"] = self.weights
        param_map["model_name"] = self.model_name
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferSparseinst(dataprocess.CInstanceSegmentationTask):

    def __init__(self, name, param):
        dataprocess.CInstanceSegmentationTask.__init__(self, name)
        # Add input/output of the process here

        self.colors = None
        self.args = None
        self.model = None
        self.cpu_device = torch.device('cpu')
        self.class_names = None

        # Create parameters class
        if param is None:
            self.set_param_object(InferSparseinstParam())
        else:
            self.set_param_object(copy.deepcopy(param))

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def run(self):
        # Core function of your process
        # Call begin_task_run for initialization
        self.begin_task_run()

        # Examples :
        # Get input :
        input = self.get_input(0)

        # Get parameters :
        param = self.get_param_object()
        plugin_folder = os.path.dirname(os.path.abspath(__file__))
        models_folder = os.path.join(plugin_folder, "models")
        if not os.path.isdir(models_folder):
            os.mkdir(models_folder)

        if self.model is None or param.update:
            if param.custom:
                self.args = Namespace()
                self.args.opts = ["MODEL.WEIGHTS", param.weights]
                self.args.confidence_threshold = param.conf_thres
                self.args.output = ""
                self.args.input = ""
                self.args.config_file = param.cfg

                with open(self.args.config_file, 'r') as f:
                    cfg = CfgNode.load_cfg(f.read())
                    cfg.MODEL.WEIGHTS = self.args.opts[1]
                    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = self.args.confidence_threshold
                    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.args.confidence_threshold
                    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = self.args.confidence_threshold
                self.class_names = cfg.CLASS_NAMES
                self.model = VisualizationDemo(cfg).predictor
                param.update = False
            else:
                weights = os.path.join(models_folder, param.model_name+'.pth')
                if not os.path.isfile(weights):
                    gdrive_download(model_zoo[param.model_name], weights)
                self.args = Namespace()
                self.args.opts = ["MODEL.WEIGHTS", weights]
                self.args.confidence_threshold = param.conf_thres
                self.args.output = ""
                self.args.input = ""
                self.args.config_file = os.path.join(plugin_folder, "configs", param.model_name+'.yaml')
                cfg = self.setup_cfg(self.args)
                try:
                    self.class_names = cfg.CLASS_NAMES
                except AttributeError:
                    self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                                        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
                                        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                                        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
                                        'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                                        'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                                        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
                                        'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                                        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
                                        'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                                        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                                        'scissors', 'teddy bear', 'hair drier', 'toothbrush']
                param.update = False
                self.model = VisualizationDemo(cfg).predictor

            self.set_names(self.class_names)

        if input.is_data_available():
            # Get image from input/output (numpy array):
            srcImage = input.get_image()

            h, w = np.shape(srcImage)[:2]

            predictions = self.model(srcImage)
            instances = predictions["instances"].to(self.cpu_device)
            instances = instances[instances.scores > self.args.confidence_threshold]
            masks = np.asarray(instances.pred_masks.tensor)
            bboxes = [GenericMask(x, h, w).bbox() for x in masks]
            masks = [GenericMask(x, h, w).mask.astype(dtype=np.bool) for x in masks]
            classes = instances.pred_classes.numpy()
            scores = instances.scores.numpy()
            instance_seg = np.zeros((h, w))
            np.random.seed(0)
            for i, (mask, cls, score, box) in enumerate(zip(masks, classes, scores, bboxes)):
                if cls >= len(self.class_names):
                    continue
                instance_seg[mask] = i + 1
                x1, y1, x2, y2 = box
                w = float(x2 - x1)
                h = float(y2 - y1)
                self.add_instance(i, 1, int(cls), float(score), float(x1), float(y1), w, h,
                                             mask.astype(dtype='uint8'))

        # Step progress bar:
        self.emit_step_progress()

        # Call end_task_run to finalize process
        self.end_task_run()

    @staticmethod
    def setup_cfg(args):
        # load config from file and command-line arguments
        cfg = get_cfg()
        add_sparse_inst_config(cfg)
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        # Set score_threshold for builtin models
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
        cfg.freeze()
        return cfg


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferSparseinstFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_sparseinst"
        self.info.short_description = "Infer Sparseinst instance segmentation models"
        self.info.description = "Infer Sparseinst instance segmentation models"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Segmentation"
        self.info.icon_path = "icons/sparseinst.png"
        self.info.version = "1.1.0"
        # self.info.icon_path = "your path to a specific icon"
        self.info.authors = "Cheng, Tianheng and Wang, Xinggang and Chen, Shaoyu and Zhang, Wenqiang and Zhang, " \
                            "Qian and Huang, Chang and Zhang, Zhaoxiang and Liu, Wenyu "
        self.info.article = "Sparse Instance Activation for Real-Time Instance Segmentation"
        self.info.journal = "Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)"
        self.info.year = 2022
        self.info.license = "MIT License"
        # URL of documentation
        self.info.documentation_link = "https://github.com/hustvl/SparseInst#readme"
        # Code source repository
        self.info.repository = "https://github.com/hustvl/SparseInst"
        # Keywords used for search
        self.info.keywords = "infer, sparse, instance, segmentation, real-time, detectron2"

    def create(self, param=None):
        # Create process object
        return InferSparseinst(self.info.name, param)
