import logging
import os
from typing import Annotated, Optional

import vtk

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode


#
# ScapularMorphology
#


class ScapularMorphology(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("ScapularMorphology")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Segmentation")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = [
            "Kuan Liu (Ruijin hospital, SJTU)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/liukuan5625/SlicerScapularMorphology">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This file was originally developed by Kuan Liu, Ruijin Hospital, Shanghai Jiaotong University School of Medicine.
""")


#
# ScapularMorphologyWidget
#


class ScapularMorphologyWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        self._updatingGUIFromParameterNode = False

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        self.bounds = [0] * 6
        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/ScapularMorphology.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = ScapularMorphologyLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        self.ui.scapulaInputbox.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.humerusInputbox.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.outputSegmentationSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)

        self.ui.scapulaInputbox.setMRMLScene(slicer.mrmlScene)
        self.ui.humerusInputbox.setMRMLScene(slicer.mrmlScene)
        self.ui.scapulaInputbox.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.ui.humerusInputbox.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.ui.outputSegmentationSelector.nodeTypes = ["vtkMRMLSegmentationNode"]

        self.ui.heatmapCheckBox.connect('toggled(bool)', self.updateParameterNodeFromGUI)
        self.ui.cpuCheckBox.connect('toggled(bool)', self.updateParameterNodeFromGUI)

        self.initializeISrange()
        self.ui.adjustISRange.valuesChanged.connect(self.modifyISrange)

        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def initializeISrange(self):
        redCompositeNode = slicer.app.layoutManager().sliceWidget('Red').sliceLogic().GetSliceCompositeNode()
        volumeID = redCompositeNode.GetBackgroundVolumeID()
        if volumeID is None:
            self.ui.adjustISRange.setMinimumValue(0.0)
            self.ui.adjustISRange.setMaximumValue(0.0)
        else:
            volumeNode = slicer.mrmlScene.GetNodeByID(volumeID)
            volumeNode.GetBounds(self.bounds)
            self.ui.adjustISRange.setRange(self.bounds[-2], self.bounds[-1])
            self.ui.adjustISRange.setMinimumValue(self.bounds[-2])
            self.ui.adjustISRange.setMaximumValue(self.bounds[-1])

    def modifyISrange(self):
        redCompositeNode = slicer.app.layoutManager().sliceWidget('Red').sliceLogic().GetSliceCompositeNode()
        volumeID = redCompositeNode.GetBackgroundVolumeID()
        if volumeID is None:
            self.ui.adjustISRange.setMinimumValue(0.0)
            self.ui.adjustISRange.setMaximumValue(0.0)
        else:
            volumeNode = slicer.mrmlScene.GetNodeByID(volumeID)
            volumeNode.GetBounds(self.bounds)
            self.ui.adjustISRange.setRange(self.bounds[-2], self.bounds[-1])

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.GetNodeReference("ScapulaVolume"):
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.SetNodeReferenceID("ScapulaVolume", firstVolumeNode.GetID())
        if not self._parameterNode.GetNodeReference("HumerusVolume"):
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.SetNodeReferenceID("HumerusVolume", firstVolumeNode.GetID())

    def setParameterNode(self, inputParameterNode) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # Initial GUI update
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        # Update node selectors and sliders
        self.ui.scapulaInputbox.setCurrentNode(self._parameterNode.GetNodeReference("ScapulaVolume"))
        self.ui.humerusInputbox.setCurrentNode(self._parameterNode.GetNodeReference("HumerusVolume"))
        self.ui.heatmapCheckBox.checked = self._parameterNode.GetParameter("Heat") == "true"
        self.ui.cpuCheckBox.checked = self._parameterNode.GetParameter("CPU") == "true"
        self.ui.outputSegmentationSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputSegmentation"))

        # Update buttons states and tooltips
        scapulaVolume = self._parameterNode.GetNodeReference("ScapulaVolume")
        if scapulaVolume:
            self.ui.applyButton.toolTip = _("Start segmentation")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select input volume")
            self.ui.applyButton.enabled = False

        humerusVolume = self._parameterNode.GetNodeReference("HumerusVolume")
        if humerusVolume:
            self.ui.applyButton.toolTip = _("Start segmentation")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select input volume")
            self.ui.applyButton.enabled = False

        if humerusVolume or scapulaVolume:
            self.ui.outputSegmentationSelector.baseName = _("{volume_name} segmentation").format(
                volume_name=scapulaVolume.GetName())

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        self._parameterNode.SetNodeReferenceID("ScapulaVolume", self.ui.scapulaInputbox.currentNodeID)
        self._parameterNode.SetNodeReferenceID("HumerusVolume", self.ui.humerusInputbox.currentNodeID)
        self._parameterNode.SetParameter("Heat", "true" if self.ui.heatmapCheckBox.checked else "false")
        self._parameterNode.SetParameter("CPU", "true" if self.ui.cpuCheckBox.checked else "false")
        self._parameterNode.SetNodeReferenceID("OutputSegmentation", self.ui.outputSegmentationSelector.currentNodeID)

        self._parameterNode.EndModify(wasModified)

    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # Create new segmentation node, if not selected yet
            if not self.ui.outputSegmentationSelector.currentNode():
                self.ui.outputSegmentationSelector.addNode()

            scapulaNodeID = self.ui.scapulaInputbox.currentNodeID
            humerusNodeID = self.ui.humerusInputbox.currentNodeID
            if scapulaNodeID == humerusNodeID:
                slicer.util.errorDisplay("Scapula volume and humerus volume must be different.")
                return

            # IS range
            current_min = self.ui.adjustISRange.minimumValue
            current_max = self.ui.adjustISRange.maximumValue
            if current_min == self.bounds[-2] and current_max == self.bounds[-1]:
                current_min = False
                current_max = False
            if current_min == current_max:
                current_min = False
                current_max = False

            # Compute output
            self.logic.process(self.ui.scapulaInputbox.currentNode(), self.ui.humerusInputbox.currentNode(),
                               self.ui.outputSegmentationSelector.currentNode(),
                               self.ui.heatmapCheckBox.checked, self.ui.cpuCheckBox.checked,
                               current_min, current_max)


#
# ScapularMorphologyLogic
#


class ScapularMorphologyLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

        ####################### UPDATE

        self.path = os.path.dirname(os.path.abspath(__file__))
        self.dependenciesInstalled = False
        self.clearOutputFolder = True

    def setupPythonRequirements(self, upgrade=False):
        # Install PyTorch
        try:
            import PyTorchUtils
        except ModuleNotFoundError as e:
            raise RuntimeError("This module requires PyTorch extension. Install it from the Extensions Manager.")
        minimumTorchVersion = "2.0.0"
        logging.info("Initializing PyTorch...")

        torchLogic = PyTorchUtils.PyTorchUtilsLogic()
        if not torchLogic.torchInstalled():
            logging.info("PyTorch Python package is required. Installing... (it may take several minutes)")
            torch = torchLogic.installTorch(askConfirmation=True, torchVersionRequirement=f">={minimumTorchVersion}")
            if torch is None:
                raise ValueError("PyTorch extension needs to be installed to use this module.")
        else:  # torch is installed, check version
            from packaging import version
            if version.parse(torchLogic.torch.__version__) < version.parse(minimumTorchVersion):
                raise ValueError(f"PyTorch version {torchLogic.torch.__version__} is not compatible with this module."
                                 + f" Minimum required version is {minimumTorchVersion}. You can use 'PyTorch Util' module to install PyTorch"
                                 + f" with version requirement set to: >={minimumTorchVersion}")

        # Install MONAI with required components
        logging.info("Initializing MONAI...")
        # Specify minimum version 1.3, as this is a known working version (it is possible that an earlier version works, too).
        # Without this, for some users monai-0.9.0 got installed, which failed with this error:
        # "ImportError: cannot import name ‘MetaKeys’ from 'monai.utils'"
        monaiInstallString = "monai[fire,pyyaml,nibabel,pynrrd,psutil,tensorboard,skimage,itk,tqdm]>=1.3"
        if upgrade:
            monaiInstallString += " --upgrade"
        slicer.util.pip_install(monaiInstallString)
        modelInstallString = "segmentation_models_pytorch"
        slicer.util.pip_install(modelInstallString)

        self.dependenciesInstalled = True
        logging.info("Dependencies are set up successfully.")

    def downloadModel(self) -> None:
        """Download model from shared google drive link"""

        import SampleData
        import os

        url2D = "https://github.com/liukuan5625/SlicerScapularMorphology/releases/download/v1.0.0/model_2d.pt"
        url3D = "https://github.com/liukuan5625/SlicerScapularMorphology/releases/download/v1.0.0/model_3d.pt"
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        destination_folder = os.path.join(current_dir, 'Scripts')
        model2dPath = os.path.join(destination_folder, 'model_2d.pt')
        model3dPath = os.path.join(destination_folder, 'model_3d.pt')
        name2d = 'model_2d.pt'
        name3d = 'model_3d.pt'
        if not os.path.exists(model2dPath) or not os.path.exists(model3dPath):
            logging.info(
                "If download fail, visit https://github.com/liukuan5625/SlicerScapularMorphology/releases to download "
                "wights and samples, and place to filefolder ./ScapularMorphology/Scripts.")
            print("Downloading pretrained model to local directory...")
            print("...")
            SampleData.SampleDataLogic().downloadFile(url2D, destination_folder, name2d, checksum=None)
            SampleData.SampleDataLogic().downloadFile(url3D, destination_folder, name3d, checksum=None)
            print('Done.')
        print('Pre-trained model saved to: ', model2dPath, model3dPath)

    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        if not parameterNode.GetParameter("Heat"):
            parameterNode.SetParameter("Heat", "True")
        if not parameterNode.GetParameter("CPU"):
            parameterNode.SetParameter("CPU", "True")

    def logProcessOutput(self, proc, returnOutput=False):
        # Wait for the process to end and forward output to the log
        output = ""
        from subprocess import CalledProcessError
        while True:
            try:
                line = proc.stdout.readline()
                if not line:
                    break
                if returnOutput:
                    output += line
                logging.info(line.rstrip())
            except UnicodeDecodeError as e:
                # Code page conversion happens because `universal_newlines=True` sets process output to text mode,
                # and it fails because probably system locale is not UTF8. We just ignore the error and discard the string,
                # as we only guarantee correct behavior if an UTF8 locale is used.
                pass

        proc.wait()
        retcode = proc.returncode
        if retcode != 0:
            raise CalledProcessError(retcode, proc.args, output=proc.stdout, stderr=proc.stderr)
        return output if returnOutput else None

    def process(self,
                scapulaVolume: vtkMRMLScalarVolumeNode,
                humerusVolume: vtkMRMLScalarVolumeNode,
                outputSegmentation: vtkMRMLScalarVolumeNode,
                heat=False, cpu=False, ISmin=False, ISmax=False):
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param scapulaVolume: scapula thresholded volume
        :param humerusVolume: humerus thresholded volume
        :param outputSegmentation: result
        :param heat: is output heatmap result
        :param cpu: is just use cpu
        """
        # INSTALL REQUIREMENTS
        if not self.dependenciesInstalled:
            with slicer.util.tryWithErrorDisplay("Failed to install required dependencies.", waitCursor=True):
                self.setupPythonRequirements()
        self.downloadModel()

        if not scapulaVolume:
            raise ValueError("scapulaVolume is invalid")
        if not humerusVolume:
            raise ValueError("humerusVolume is invalid")

        import time
        startTime = time.time()
        logging.info(_('Processing started'))

        # Create new empty folder
        tempFolder = slicer.util.tempDirectory()

        scaInputFile = tempFolder + "/scapula-input.nrrd"
        humInputFile = tempFolder + "/humerus-input.nrrd"
        output2DSegmentationFile = tempFolder + "/2d_glenoid.nii.gz"
        output3DSegmentationFile = tempFolder + "/3d_landmarks.csv"

        # Get Python executable path
        import shutil
        pythonSlicerExecutablePath = shutil.which('PythonSlicer')
        if not pythonSlicerExecutablePath:
            raise RuntimeError("Python was not found")
        scapularMorphologyExecutablePath = os.path.join(self.path, 'Scripts', "scapularmorphology_inference.py")
        scapularMorphologyCommand = [pythonSlicerExecutablePath, scapularMorphologyExecutablePath]

        # Segment a single volume
        self.processVolume(scaInputFile, humInputFile, scapulaVolume, humerusVolume,
                           output2DSegmentationFile, output3DSegmentationFile, outputSegmentation,
                           heat, cpu, ISmin, ISmax, scapularMorphologyCommand)

        stopTime = time.time()
        logging.info(_("Processing completed in {time_elapsed:.2f} seconds").format(time_elapsed=stopTime - startTime))

        if self.clearOutputFolder:
            logging.info(_("Cleaning up temporary folder..."))
            if os.path.isdir(tempFolder):
                shutil.rmtree(tempFolder)
        else:
            logging.info(_("Not cleaning up temporary folder: {temp_folder}").format(temp_folder=tempFolder))

    def processVolume(self, scaInputFile, humInputFile, scapulaVolume, humerusVolume,
                      output2DSegmentationFile, output3DLandmarkFile, outputSegmentation,
                      heat, cpu, ISmin, ISmax, scapularMorphologyCommand):
        """Segment a single volume
        """

        # Write input volume to file
        # TotalSegmentator requires NIFTI
        logging.info(_("Writing input file to {scapula_input_file}").format(scapula_input_file=scaInputFile))
        scavolumeStorageNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLSegmentationStorageNode")
        scavolumeStorageNode.SetFileName(scaInputFile)
        scavolumeStorageNode.UseCompressionOff()
        scavolumeStorageNode.WriteData(scapulaVolume)
        scavolumeStorageNode.UnRegister(None)

        logging.info(_("Writing input file to {humerus_input_file}").format(humerus_input_file=humInputFile))
        humvolumeStorageNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLSegmentationStorageNode")
        humvolumeStorageNode.SetFileName(humInputFile)
        humvolumeStorageNode.UseCompressionOff()
        humvolumeStorageNode.WriteData(humerusVolume)
        humvolumeStorageNode.UnRegister(None)

        # Get options
        options = ["--scapula_dir", scaInputFile, "--humerus_dir", humInputFile,
                   "--output_dir_2d", output2DSegmentationFile, "--output_dir_3d", output3DLandmarkFile,
                   "--weight_dir_2d", os.path.join(self.path, 'Scripts'), "--weight_model_name_2d", "model_2d.pt",
                   "--weight_dir_3d", os.path.join(self.path, 'Scripts'), "--weight_model_name_3d", "model_3d.pt"]
        if heat:
            options = options + ["--heat", str(heat)]
        if cpu:
            options = options + ["--cpu", str(cpu)]

        if ISmin and ISmax:
            options = options + ["--minISrange", str(ISmin)] + ["--maxISrange", str(ISmax)]

        logging.info(_('Creating segmentations with ScapularMorphology AI...'))
        logging.info(_("ScapularMorphology arguments: {options}").format(options=options))
        proc = slicer.util.launchConsoleProcess(scapularMorphologyCommand + options)
        self.logProcessOutput(proc)

        # Load result
        logging.info(_('Importing segmentation results...'))
        self.readSegmentation(output2DSegmentationFile)
        self.readLandmark(output3DLandmarkFile)
        if heat:
            self.readLandmarkHeatmap(output3DLandmarkFile)

    def readSegmentation(self, output2DSegmentationFile):
        segNode2D = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "ScapularGlenoid")
        scavolumeStorageNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLSegmentationStorageNode")
        scavolumeStorageNode.SetFileName(output2DSegmentationFile)
        scavolumeStorageNode.ReadData(segNode2D)

        segmentIDs = vtk.vtkStringArray()
        segmentation = segNode2D.GetSegmentation()
        segmentation.GetSegmentIDs(segmentIDs)
        segNode2D.GetSegmentation().GetSegmentIDs(segmentIDs)
        if segmentIDs.GetNumberOfValues() > 0:
            firstSegmentID = segmentIDs.GetValue(0)
            segment = segmentation.GetSegment(firstSegmentID)
            segment.SetColor(1.0, 0.0, 0.0)
            displayNode = segNode2D.GetDisplayNode()
            displayNode.SetSegmentVisibility3D(firstSegmentID, True)
            segmentation.SetConversionParameter("Smoothing factor", "0.0")
            segmentation.RemoveRepresentation("Closed surface")
            segNode2D.CreateClosedSurfaceRepresentation()

    def readLandmark(self, output3DLandmarkFile):
        markupsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        slicer.modules.markups.logic().ImportControlPointsFromCSV(markupsNode, output3DLandmarkFile)
        displayNode = markupsNode.GetDisplayNode()
        displayNode.SetGlyphScale(5.0)
        displayNode.SetTextScale(5.0)
        markupsNode.SetName("ScapulaLandmarks")

    def readLandmarkHeatmap(self, output3DLandmarkFile):
        landmarks_name = ['TS', 'AI', 'AA', 'AC', 'PC']
        for name in landmarks_name:
            volumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", f"Heatmap_{name}")
            storageNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLVolumeArchetypeStorageNode")
            filename = output3DLandmarkFile.replace('.csv', f'_{name}.nii.gz')
            storageNode.SetFileName(filename)
            storageNode.ReadData(volumeNode)


#
# ScapularMorphologyTest
#

class ScapularMorphologyTest(ScriptedLoadableModuleTest):
    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_ScapularMorphology()

    def test_ScapularMorphology(self):
        """ Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data
        import SampleData
        import os
        urlraw = "https://github.com/liukuan5625/SlicerScapularMorphology/releases/download/v1.0.0/sample_raw.nii.gz"
        urlsca = "https://github.com/liukuan5625/SlicerScapularMorphology/releases/download/v1.0.0/sample_scapula.nii.gz"
        urlhum = "https://github.com/liukuan5625/SlicerScapularMorphology/releases/download/v1.0.0/sample_humerus.nii.gz"
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        destination_folder = os.path.join(current_dir, 'Scripts')
        samplerawPath = os.path.join(destination_folder, 'sample_raw.nii.gz')
        samplescaPath = os.path.join(destination_folder, 'sample_scapula.nii.gz')
        samplehumPath = os.path.join(destination_folder, 'sample_humerus.nii.gz')
        nameraw = 'sample_raw.nii.gz'
        namesca = 'sample_scapula.nii.gz'
        namehum = 'sample_humerus.nii.gz'
        self.delayDisplay('Download sample')
        if not os.path.exists(samplerawPath):
            SampleData.SampleDataLogic().downloadFile(urlraw, destination_folder, nameraw, checksum=None)
        if not os.path.exists(samplescaPath):
            SampleData.SampleDataLogic().downloadFile(urlsca, destination_folder, namesca, checksum=None)
        if not os.path.exists(samplehumPath):
            SampleData.SampleDataLogic().downloadFile(urlhum, destination_folder, namehum, checksum=None)
        print('Sample data saved to: ', destination_folder)

        [success, rawVolume] = slicer.util.loadVolume(samplerawPath, returnNode=True)
        [success, scapulaVolume] = slicer.util.loadSegmentation(samplescaPath, returnNode=True)
        [success, humerusVolume] = slicer.util.loadSegmentation(samplehumPath, returnNode=True)
        outputSegmentation = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')

        # Test the module logic
        testLogic = True

        if testLogic:
            logic = ScapularMorphologyLogic()
            self.delayDisplay('Set up required Python packages')
            logic.setupPythonRequirements()
            self.delayDisplay('Compute output')
            logic.process(scapulaVolume, humerusVolume, outputSegmentation,
                          False, False, False, False)

        else:
            logging.warning("test_TotalSegmentator1 logic testing was skipped")

        self.delayDisplay('Test passed')
