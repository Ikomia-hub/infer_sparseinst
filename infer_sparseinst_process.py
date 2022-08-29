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

    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        # Example : self.windowSize = int(param_map["windowSize"])
        self.update = True
        self.conf_thres = float(param_map["conf_thres"])
        self.custom = strtobool(param_map["custom"])
        self.cfg = param_map["cfg"]
        self.weights = param_map["weights"]
        self.model_name = param_map["model_name"]

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = core.ParamMap()
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
class InferSparseinst(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        # Add input/output of the process here
        # Example :  self.addInput(dataprocess.CImageIO())
        #           self.addOutput(dataprocess.CImageIO())
        self.setOutputDataType(core.IODataType.IMAGE_LABEL, 0)
        self.addOutput(dataprocess.CImageIO())
        # Add graphics output
        self.addOutput(dataprocess.CGraphicsOutput())
        # Add numeric output
        self.addOutput(dataprocess.CBlobMeasureIO())

        self.args = None
        self.model = None
        self.cpu_device = torch.device('cpu')
        self.class_names = None

        # Create parameters class
        if param is None:
            self.setParam(InferSparseinstParam())
        else:
            self.setParam(copy.deepcopy(param))

    def getProgressSteps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()

        # Examples :
        # Get input :
        input = self.getInput(0)

        # Get output :
        instance_out = self.getOutput(0)
        graphics_output = self.getOutput(2)
        numeric_output = self.getOutput(3)

        # Get parameters :
        param = self.getParam()
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

        if input.isDataAvailable():
            # Get image from input/output (numpy array):
            srcImage = input.getImage()
            graphics_output.setNewLayer("SparseInst")
            graphics_output.setImageIndex(1)
            numeric_output.clearData()
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
            colors = [[0, 0, 0]]
            np.random.seed(10)
            for i, (mask, cls, score, box) in enumerate(zip(masks, classes, scores, bboxes)):
                if cls >= len(self.class_names):
                    continue
                colors.append([int(c) for c in np.random.choice(range(256), size=3)])
                instance_seg[mask] = i + 1
                x1, y1, x2, y2 = box
                w = float(x2 - x1)
                h = float(y2 - y1)
                prop_rect = core.GraphicsRectProperty()
                prop_rect.pen_color = colors[-1]
                graphics_box = graphics_output.addRectangle(float(x1), float(y1), w, h, prop_rect)
                graphics_box.setCategory(self.class_names[cls])
                # Label
                name = self.class_names[int(cls)]
                prop_text = core.GraphicsTextProperty()
                prop_text.font_size = 8
                prop_text.color = [c // 2 for c in colors[-1]]
                graphics_output.addText(name, float(x1), float(y1), prop_text)
                # Object results
                results = []
                confidence_data = dataprocess.CObjectMeasure(
                    dataprocess.CMeasure(core.MeasureId.CUSTOM, "Confidence"),
                    float(score),
                    graphics_box.getId(),
                    name)
                box_data = dataprocess.CObjectMeasure(
                    dataprocess.CMeasure(core.MeasureId.BBOX),
                    graphics_box.getBoundingRect(),
                    graphics_box.getId(),
                    name)
                results.append(confidence_data)
                results.append(box_data)
                numeric_output.addObjectMeasures(results)
            self.setOutputColorMap(1, 0, colors)
            self.forwardInputImage(0, 1)
            instance_out.setImage(instance_seg)

        # Call to the process main routine
        # dstImage = ...

        # Set image of input/output (numpy array):
        # output.setImage(dstImage)

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()

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
        self.info.shortDescription = "Infer Sparseinst instance segmentation models"
        self.info.description = "Infer Sparseinst instance segmentation models"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Segmentation"
        self.info.iconPath = "icons/sparseinst.png"
        self.info.version = "1.0.0"
        # self.info.iconPath = "your path to a specific icon"
        self.info.authors = "Cheng, Tianheng and Wang, Xinggang and Chen, Shaoyu and Zhang, Wenqiang and Zhang, " \
                            "Qian and Huang, Chang and Zhang, Zhaoxiang and Liu, Wenyu "
        self.info.article = "Sparse Instance Activation for Real-Time Instance Segmentation"
        self.info.journal = "Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)"
        self.info.year = 2022
        self.info.license = "MIT License"
        # URL of documentation
        self.info.documentationLink = "https://github.com/hustvl/SparseInst#readme"
        # Code source repository
        self.info.repository = "https://github.com/hustvl/SparseInst"
        # Keywords used for search
        self.info.keywords = "infer, sparse, instance, segmentation, real-time, detectron2"

    def create(self, param=None):
        # Create process object
        return InferSparseinst(self.info.name, param)
