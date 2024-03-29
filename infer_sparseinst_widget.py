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

from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from infer_sparseinst.infer_sparseinst_process import InferSparseinstParam

# PyQt GUI framework
from PyQt5.QtWidgets import *
from infer_sparseinst.utils import model_zoo


# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferSparseinstWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferSparseinstParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.gridLayout = QGridLayout()
        self.combo_model = pyqtutils.append_combo(self.gridLayout, "Model name")
        for model in model_zoo:
            self.combo_model.addItem(model)
        self.combo_model.setCurrentText(self.parameters.model_name)
        self.double_spin_thres = pyqtutils.append_double_spin(self.gridLayout, "Confidence threshold",
                                                              self.parameters.conf_thres, min=0., max=1., step=1e-2)
        self.check_custom = pyqtutils.append_check(self.gridLayout, "Default weights",
                                                   not self.parameters.use_custom_model)
        self.browse_cfg = pyqtutils.append_browse_file(self.gridLayout, "Config file (.yaml)", self.parameters.config_file)
        self.browse_weights = pyqtutils.append_browse_file(self.gridLayout, "Model weights (.pth)",
                                                           self.parameters.model_weight_file)
        self.browse_weights.setEnabled(not self.check_custom.isChecked())
        self.browse_cfg.setEnabled(not self.check_custom.isChecked())
        self.check_custom.stateChanged.connect(self.on_check_custom)

        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.gridLayout)

        # Set widget layout
        self.set_layout(layout_ptr)

    def on_check_custom(self, i):
        self.browse_weights.setEnabled(not self.check_custom.isChecked())
        self.browse_cfg.setEnabled(not self.check_custom.isChecked())
        self.combo_model.setEnabled(self.check_custom.isChecked())

    def on_apply(self):
        # Apply button clicked slot

        # Get parameters from widget
        # Example : self.parameters.windowSize = self.spinWindowSize.value()
        self.parameters.update = True
        self.parameters.conf_thres = self.double_spin_thres.value()
        self.parameters.use_custom_model = not self.check_custom.isChecked()
        self.parameters.config_file = self.browse_cfg.path
        self.parameters.model_weight_file = self.browse_weights.path
        self.parameters.model_name = self.combo_model.currentText()
        # Send signal to launch the process
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferSparseinstWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "infer_sparseinst"

    def create(self, param):
        # Create widget object
        return InferSparseinstWidget(param, None)
