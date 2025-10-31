# process_window.py
import os
import numpy as np
import nibabel as nib
import pywt
from scipy.ndimage import affine_transform
from scipy.io import savemat, loadmat
from scipy.signal import butter, filtfilt
from PyQt6.QtWidgets import QWidget, QLabel,QComboBox, QFileDialog, QPushButton, QListWidget, QProgressDialog,QLineEdit, QFormLayout, QMessageBox
from PyQt6.QtGui import QIntValidator 
from nilearn.input_data import NiftiLabelsMasker
from nilearn.image import resample_to_img
import matlab.engine
import matlab
from nibabel.processing import resample_from_to

class ProcessWindow(QWidget):
    def __init__(self):
        super().__init__()
        # Initialize MATLAB Engine (this may take a few seconds)
        try:
            print("Starting MATLAB Engine...")
            self.eng = matlab.engine.start_matlab()
            print("MATLAB Engine started successfully!")
        except Exception as e:
            print(f"Failed to start MATLAB Engine: {e}")
            print("MATLAB Engine features will be disabled.")
            self.eng = None
        self.createComponents()
        self.startupFcn()
      
    # Reset default value. (This only update the parameters not the field)
    def parameters_default(self):
        self.settings = {
        'file_type': 0,
        'file_list': [],
        'file_path':[],
        'atlas_type': 0,
        'atlas_list': [],
        'atlas_path': [],
        'TR' : 1,
        'window_size': 20,
        'step_size': 1,
        'filter_type': 0,
        'filter_setting1': '',
        'filter_setting2': '',
        'kernel': 0,
        'kernel_setting': ''
        }

    # Update parameters from field
    def parameters_update_setting(self):
        self.settings['TR'] = int(self.TRs_input.text()) if self.TRs_input.text().isdigit() else 1
        self.settings['window_size'] = int(self.window_size_input.text()) if self.window_size_input.text().isdigit() else 20
        self.settings['step_size'] = int(self.step_size_input.text()) if self.step_size_input.text().isdigit() else 1
        self.settings['filter_type'] = self.filter_input.currentIndex()
        self.settings['filter_setting1'] = self.filter_edit_field_1.text() if self.filter_edit_field_1.text() else ""
        self.settings['filter_setting2'] = self.filter_edit_field_2.text() if self.filter_edit_field_2.text() else ""
        self.settings['kernel'] = self.window_kernel_input.currentIndex()
        self.settings['kernel_setting'] = self.sigma_edit_field.text() if self.sigma_edit_field.text() else ""

    # Update field from parameters
    def parameters_update_field(self):
        # File dropdown
        self.file_format_dropdown.setCurrentIndex(self.settings.get('file_type', 0))
        index = self.settings.get('file_list', 0)
        if isinstance(index, list):
            index = index[0] if index else 0  # get first item or default to 0
        self.file_list_widget.setCurrentRow(index)
        # Atlas dropdown
        self.atlas_input.setCurrentIndex(self.settings.get('atlas_type', 0))
        index = self.settings.get('atlas_list', 0)
        try:
            self.atlas_list_widget.setCurrentRow(int(index))
        except (ValueError, TypeError):
            self.atlas_list_widget.setCurrentRow(0)

        self.TRs_input.setText(str(self.settings.get('TR', 1)))
        self.window_size_input.setText(str(self.settings.get('window_size', 20)))
        self.step_size_input.setText(str(self.settings.get('step_size', 1)))
        self.filter_input.setCurrentIndex(self.settings.get('filter_type', 0))
        self.filter_edit_field_1.setText(str(self.settings.get('filter_setting1', 0)))
        self.filter_edit_field_2.setText(str(self.settings.get('filter_setting2', 0)))
        self.window_kernel_input.setCurrentIndex(self.settings.get('kernel', 0))
        self.sigma_edit_field.setText(str(self.settings.get('kernel_setting', 0)))
    
    # Gaussian Kernel Function
    def gauss_kernel(g_x, g_m, g_s):
        result = np.exp(-((g_x - g_m) ** 2) / (2 * g_s ** 2))
        return result
    
    # Code that executes after component creation
    def startupFcn(self):
        self.parameters_default()
        # initialize the atlas dropdown interface (doesn't affect much the function, just the entrance appearace)
        path = os.path.join(self.settings.get('path', ''), 'atlas', 'BN_atlas_1_25mm.nii.txt')
        if os.path.exists(path):
            with open(path, 'r') as f:
                roi_names = [line.strip() for line in f.readlines()]
            self.settings['atlas_list'] = roi_names # can be replace with self.atlas_list (actually self.settings['___'] = self.___ )
        else:
            roi_names = ['[File not found]']
        self.settings['atlas_path'] = []
        self.atlas_list_widget.clear()
        self.atlas_list_widget.addItems(roi_names)

    # Value changed function: FileFormatDropDown
    def gatp_file_type(self):
        self.file_type = self.file_format_dropdown.currentIndex() 
        if self.file_type == 0:
            # Enable atlas dropdown, disable atlas list and button
            self.atlas_input.setEnabled(True)
            self.atlas_list_widget.setEnabled(False)
            self.add_atlas_button.setEnabled(False)
            self.add_atlas_button.setText("Add Masks")
            self.atlas_input.setCurrentIndex(1)  # Default to Brainnetome
            # Load Brainnetome ROI names
            path = os.path.join(self.settings.get('path', ''), 'atlas', 'BN_atlas_1_25mm.nii.txt')
            if os.path.exists(path):
                with open(path, 'r') as f:
                    roi_names = [line.strip() for line in f.readlines()]
                self.settings['atlas_list'] = roi_names
            else:
                roi_names = ['[File not found]']
            self.atlas_list_widget.clear()
            self.atlas_list_widget.addItems(roi_names)
            # Clear file list
            self.settings['file_list'] = []
            self.settings['file_path_list'] = []
            self.file_list_widget.clear()
        elif self.file_type == 1:
            # Disable atlas dropdown, enable atlas list and button
            self.atlas_input.setEnabled(False)
            self.atlas_list_widget.setEnabled(True)
            self.add_atlas_button.setEnabled(True)
            self.add_atlas_button.setText("Add Labels")
            # Clear atlas and file lists
            self.settings['atlas_list'] = []
            self.atlas_list_widget.clear()
            self.settings['file_list'] = []
            self.settings['file_path_list'] = []
            self.file_list_widget.clear()
    # Open image or matrix file function  
    def gatp_open_file(self):
        self.file_type = self.file_format_dropdown.currentIndex() 
        if self.file_type == 0: #image
            filter_str = "NIfTI Files (*.nii *.nii.gz)"
        elif self.file_type == 1: #matrix
            filter_str = "MATLAB Matrix Files (*.mat)"
        else:
            return
        # Open file dialog
        files, _ = QFileDialog.getOpenFileNames(self, "Select One or More Files", "", filter_str)
        if not files:
            return
        self.settings['file_list'] = files
        self.settings['file_path'] = [f.rsplit('/', 1)[0] for f in files]
        self.file_list_widget.clear()
        self.file_list_widget.addItems([f.rsplit('/', 1)[-1] for f in files])
    
    # Value changed function: AtlasDropDown
    def gatp_atlas_type(self):
        value = self.atlas_input.currentIndex()
        if value ==0:
            self.atlas_list_widget.setEnabled(False)
            self.add_atlas_button.setEnabled(False)
            path = os.path.join(self.settings.get('path', ''), 'atlas', 'BN_atlas_1_25mm.nii.txt')
            if os.path.exists(path):
                with open(path, 'r') as f:
                    roi_names = [line.strip() for line in f.readlines()]
                self.settings['atlas_list'] = roi_names
            else:
                roi_names = ['[File not found]']
            self.settings['atlas_path'] = []
            self.atlas_list_widget.clear()
            self.atlas_list_widget.addItems(roi_names)
        elif value == 1:
            self.atlas_list_widget.setEnabled(False)
            self.add_atlas_button.setEnabled(False)
            path = os.path.join(self.settings.get('path', ''), 'atlas', 'AAL2v1_2mm.nii.txt')
            if os.path.exists(path):
                with open(path, 'r') as f:
                    roi_names = [line.strip() for line in f.readlines()]
                self.settings['atlas_list'] = roi_names
            else:
                roi_names = ['[File not found]']
            self.settings['atlas_path'] = []
            self.atlas_list_widget.clear()
            self.atlas_list_widget.addItems(roi_names)

        elif value == 2:
            # Load AAL ROI names from file
            self.atlas_list_widget.setEnabled(True)
            self.add_atlas_button.setEnabled(True)
            self.settings['atlas_list'] = []
            self.settings['atlas_path'] = []
            self.atlas_list_widget.clear()
        
    # Button pushed function: Button_AddAtlas
    def gatp_open_atlas(self):
        self.file_type = self.file_format_dropdown.currentIndex() 
        if self.file_type == 0:
            # If input is image: allow multiple .nii or .nii.gz files
            files, _ = QFileDialog.getOpenFileNames(self, "Select One or More Atlas Files", "", "NIfTI Files (*.nii *.nii.gz)")
            if not files:
                return
            self.settings['atlas_list'] = files
            self.settings['atlas_path'] = [f.rsplit('/', 1)[0] for f in files]
            self.atlas_list_widget.clear()
            self.atlas_list_widget.addItems([f.rsplit('/', 1)[-1] for f in files])

        elif self.file_type == 1:
            # If input is .txt ROI names
            file, _ = QFileDialog.getOpenFileName(self, "Select ROI Name File", "", "Text Files (*.txt)")
            if not file:
                return
            with open(file, 'r') as f:
                roi_names = [line.strip() for line in f.readlines()]
            self.settings['atlas_list'] = roi_names
            self.settings['atlas_path'] = file.rsplit('/', 1)[0]
            self.atlas_list_widget.clear()
            self.atlas_list_widget.addItems(roi_names)
    # Value changed function: WindowKernelDropDown
    def gatp_kernel(self):
        self.kernel = self.window_kernel_input.currentIndex()   
        # No kernel
        if self.kernel == 0:  
            self.sigma_edit_field.setEnabled(False)
            self.sigma_label.setEnabled(False)
        # Gaussian kernel
        if self.kernel == 1:  
            self.sigma_edit_field.setEnabled(True)
            self.sigma_label.setEnabled(True)

    # Value changed function: FilterDropDown
    def gatp_filter_type(self):
        self.filter_type = self.filter_input.currentIndex()  
        if self.filter_type == 0:  # None
            self.filter_label_2.setEnabled(False)
            self.filter_edit_field_2.setEnabled(False)
            self.filter_label_1.setEnabled(False)
            self.filter_edit_field_1.setEnabled(False)
            self.filter_label_2.setText("Filter Option")
            self.filter_label_1.setText("Filter Option")

        elif self.filter_type == 1:  # Bandpass
            self.filter_label_2.setEnabled(True)
            self.filter_edit_field_2.setEnabled(True)
            self.filter_label_1.setEnabled(True)
            self.filter_edit_field_1.setEnabled(True)
            self.filter_label_2.setText("Lower Limit (1/Hz)")
            self.filter_label_1.setText("Upper Limit (1/Hz)")

        elif self.filter_type == 2:  # Highpass
            self.filter_label_2.setEnabled(False)
            self.filter_edit_field_2.setEnabled(False)
            self.filter_label_1.setEnabled(True)
            self.filter_edit_field_1.setEnabled(True)
            self.filter_label_2.setText("Filter Option")
            self.filter_label_1.setText("Cut-off Frequency (1/Hz)")

        elif self.filter_type == 3:  # Lowpass
            self.filter_label_2.setEnabled(False)
            self.filter_edit_field_2.setEnabled(False)
            self.filter_label_1.setEnabled(True)
            self.filter_edit_field_1.setEnabled(True)
            self.filter_label_2.setText("Filter Option")
            self.filter_label_1.setText("Cut-off Frequency (1/Hz)")

        elif self.filter_type == 4:  # Wavelet
            self.filter_label_2.setEnabled(True)
            self.filter_edit_field_2.setEnabled(True)
            self.filter_label_1.setEnabled(True)
            self.filter_edit_field_1.setEnabled(True)
            self.filter_label_2.setText("Wavelet Selections")
            self.filter_label_1.setText("Wavelet Levels")
    # Button pushed function: Button_Default
    def gatp_defaultset(self):
        self.parameters_default()
        self.parameters_update_field()

    # Button pushed function: SaveSettingsButton
    def gatp_savesetting(self):
        self.parameters_update_setting() #Update settings from fields
        print("setting:", self.settings)

    # Code that executes after component creation

    # Button pushed function: Button_LoadSetting

    #  Component initialization & Create UIFigure and components
    def createComponents(self):
        # Create GATFD_process_UIFigure 
        self.setWindowTitle('GAT-FD - Sliding Window Correlation Analysis v0.2a')
        self.setFixedSize(800, 600)
        self.layout = QFormLayout()
        
        # File Format dropdown
        self.file_format_dropdown = QComboBox()
        self.file_format_dropdown.addItems(["Image (.nii/.nii.gz)", "Matrix (.mat) (Time series)"])
        self.file_format_dropdown.setCurrentIndex(0)
        self.file_format_dropdown.currentIndexChanged.connect(self.gatp_file_type)
        self.layout.addRow("File Format", self.file_format_dropdown )
        # File List Widget
        self.file_list_widget = QListWidget()
        self.layout.addRow("File List", self.file_list_widget )
        # Load file button
        self.open_file_button = QPushButton("Load File(s)")
        self.open_file_button.clicked.connect(self.gatp_open_file)
        self.layout.addRow(self.open_file_button )

        # TRs input
        self.TRs_input = QLineEdit()
        self.TRs_input.setValidator(QIntValidator())
        self.TRs_input.setText("1")
        self.layout.addRow("TR(s)", self.TRs_input )

        # Window Size input
        self.window_size_input = QLineEdit()
        self.window_size_input.setValidator(QIntValidator())
        self.window_size_input.setText("20")
        self.layout.addRow("Window Size", self.window_size_input)

        # Step Size input
        self.step_size_input = QLineEdit()
        self.step_size_input.setValidator(QIntValidator())
        self.step_size_input.setText("1")
        self.layout.addRow("Step Size", self.step_size_input)

        # filter dropdown input
        self.filter_input = QComboBox()
        self.filter_input.addItems(['None', 'Bandpass', 'Highpass', 'Lowpass', 'Wavelet'])
        self.filter_input.setCurrentIndex(0)
        self.filter_input.currentIndexChanged.connect(self.gatp_filter_type)
        self.layout.addRow("Filter", self.filter_input)

        # filter edit field 1
        self.filter_label_1 =QLabel()
        self.filter_edit_field_1 = QLineEdit()
        self.filter_label_1.setEnabled(False)
        self.filter_edit_field_1.setEnabled(False)
        self.filter_label_1.setText("Filter Option")

        self.filter_label_2 =QLabel()
        self.filter_edit_field_2 = QLineEdit()
        self.filter_label_2.setEnabled(False)
        self.filter_edit_field_2.setEnabled(False)
        self.filter_label_2.setText("Filter Option")
        
        self.layout.addRow(self.filter_label_1, self.filter_edit_field_1)
        self.layout.addRow(self.filter_label_2, self.filter_edit_field_2)
        
        # Kernel input
        self.window_kernel_input = QComboBox()
        self.window_kernel_input.addItems(['None', 'Gaussian'])
        self.window_kernel_input.setCurrentIndex(0)
        self.window_kernel_input.currentIndexChanged.connect(self.gatp_kernel)
        self.layout.addRow("Window Kernel", self.window_kernel_input)

        # Kernel edit field
        self.sigma_label =QLabel()
        self.sigma_edit_field = QLineEdit()
        self.sigma_label.setEnabled(False)
        self.sigma_edit_field.setEnabled(False)
        self.sigma_label.setText("\sigma")
        self.layout.addRow(self.sigma_label, self.sigma_edit_field)

        # Atlas dropdown input
        self.atlas_input = QComboBox()
        self.atlas_input.addItems(['Brainnetome 1.25mm', 'AAL 2mm', 'Custom'])
        self.atlas_input.setCurrentIndex(0)
        self.atlas_input.currentIndexChanged.connect(self.gatp_atlas_type)
        self.layout.addRow("Atlas", self.atlas_input)
        # Atlas List Widget
        self.atlas_list_widget = QListWidget()
        self.atlas_list_widget.setEnabled(False)
        self.layout.addRow("Atlas List", self.atlas_list_widget )
        # Add Atlas button
        self.add_atlas_button = QPushButton('Add Mask')
        self.add_atlas_button.clicked.connect(self.gatp_open_atlas)
        self.layout.addRow(self.add_atlas_button)
    
        # Default Setting Button
        self.default_settings_button = QPushButton('Default Setting')
        self.default_settings_button.clicked.connect(self.gatp_defaultset)
        self.layout.addRow(self.default_settings_button)

        # Save button 
        self.save_button = QPushButton('Save')
        self.save_button.clicked.connect(self.gatp_savesetting)
        self.layout.addRow(self.save_button)

        # Run button 
        self.run_button = QPushButton('Run')
        self.run_button.clicked.connect(self.gatp_run_process)
        self.layout.addRow(self.run_button)

        self.setLayout(self.layout)

    # Run process
    def gatp_run_process(self):
        # ——— Update settings and parameters ———
        fnc_pro_filelength = len(self.settings['file_list'])
        self.parameters_update_setting()

        # Load filter settings based on user selection
        filter_type = self.settings['filter_type']
        if filter_type == 0:
            filter_setting1 = filter_setting2 = 0
        elif filter_type == 1:
            try:
                filter_setting1 = float(self.settings['filter_setting1'])
                filter_setting2 = float(self.settings['filter_setting2'])
                # Validate bandpass: lower < upper
                if filter_setting1 >= filter_setting2:
                    QMessageBox.critical(self, "Error", "Please enter correct band pass parameters")
                    return
            except (ValueError, TypeError):
                QMessageBox.critical(self, "Error", "Invalid bandpass filter parameters. Please enter valid numbers.")
                return
        elif filter_type in (2, 3):
            try:
                filter_setting1 = float(self.settings['filter_setting1'])
                filter_setting2 = 0
            except (ValueError, TypeError):
                QMessageBox.critical(self, "Error", "Invalid filter parameter. Please enter a valid number.")
                return
        elif filter_type == 4:
            try:
                # For wavelet filter, handle the settings more carefully
                setting1_str = str(self.settings['filter_setting1']).strip()
                setting2_str = str(self.settings['filter_setting2']).strip()
                
                # Try to convert to integers, handling potential multi-value strings
                if ' ' in setting1_str:
                    # If there are spaces, take the first value
                    filter_setting1 = int(setting1_str.split()[0])
                else:
                    filter_setting1 = int(setting1_str)
                    
                if ' ' in setting2_str:
                    # If there are spaces, take the first value
                    filter_setting2 = int(setting2_str.split()[0])
                else:
                    filter_setting2 = int(setting2_str)
            except (ValueError, TypeError):
                QMessageBox.critical(self, "Error", "Invalid wavelet filter parameters. Please enter valid integers.")
                return
        else:
            # Unexpected filter type
            print("filter type error")
            return

        # Determine atlas and window size depending on file format
        if self.file_format_dropdown.currentIndex() == 0:
            atlas_index = self.atlas_input.currentIndex()
            atlas_files = self.settings['atlas_list']
            if atlas_index == 2 and atlas_files:
                # Custom: merge multiple atlas masks into one integer-labeled mask
                masks = []
                for path in atlas_files:
                    img = nib.load(path)
                    masks.append((img.get_fdata() > 0).astype(np.int32))
                fnc_pro_atlas_masks = np.sum([m * (i+1)
                                            for i, m in enumerate(masks)], axis=0)
                atlas_img = fnc_pro_atlas_masks  # use array directly
            elif atlas_index == 0:
                # Use default Brainetome atlas
                atlas_img = os.path.join(self.settings.get('path', ''), 'atlas', 'BN_atlas_1_25mm.nii')
                fnc_pro_atlas_masks = nib.load(atlas_img).get_fdata().astype(np.int32)
            elif atlas_index == 1:
                # Use default AAL2 atlas
                atlas_img = os.path.join(self.settings.get('path', ''), 'atlas', 'AAL2v1_2mm.nii.txt')
                fnc_pro_atlas_masks = nib.load(atlas_img).get_fdata().astype(np.int32)
            else:
                QMessageBox.critical(self, "Error", "Invalid atlas selection")
                return
            atl_len = int(fnc_pro_atlas_masks.max())
            fnc_window_size = self.settings['window_size']
        else:
            # Matrix input: atlas_list length defines number of regions
            atl_len = len(self.settings['atlas_list'])
            if atl_len < 1:
                QMessageBox.critical(self, 'Error', 'Please load label')
                return
            fnc_window_size = self.settings['window_size']

        # Prompt for output folder
        fnc_out_path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not fnc_out_path:
            return

        # Initialize progress dialog
        progress_dialog = QProgressDialog("Processing...", None, 0, fnc_pro_filelength, self)
        progress_dialog.setWindowTitle("Processing")
        progress_dialog.show()

        for i in range(fnc_pro_filelength):
            file_path = os.path.join(self.settings['file_path'][i], self.settings['file_list'][i])

            if self.file_format_dropdown.currentIndex() == 0:
                # Load NIfTI with memory mapping
                func_img = nib.load(file_path, mmap=True)
                func_data = func_img.get_fdata()
                fnc_raw_len = func_data.shape[3]
                fnc_window_count = fnc_raw_len - fnc_window_size + 1

                # --- Resample atlas to match functional image (MATLAB-style) ---
                if isinstance(atlas_img, str):
                    atlas_nib = nib.load(atlas_img)
                else:
                    atlas_nib = nib.Nifti1Image(atlas_img, func_img.affine)
                atlas_data = atlas_nib.get_fdata()
                func_data = func_img.get_fdata()
                
                # CRITICAL: Ensure same data type as MATLAB (single precision)
                atlas_data = atlas_data.astype(np.float32)  # MATLAB uses 'single' (float32)
                func_data = func_data.astype(np.float32)    # MATLAB uses 'single' (float32)
                
                atlas_affine = atlas_nib.affine
                func_affine = func_img.affine
                # Get voxel sizes
                atlas_voxel_size = atlas_nib.header.get_zooms()[:3]
                func_voxel_size = func_img.header.get_zooms()[:3]
                func_phys_dims = np.array(func_data.shape[:3]) * np.array(func_voxel_size)
                expected_output_shape = tuple(int(np.ceil(dim / voxel)) for dim, voxel in zip(func_phys_dims, atlas_voxel_size))
                
                fnc_rawdata_t = func_data[..., 0].copy() 
                
                # --- Use MATLAB imwarp to match MATLAB processing exactly ---
                def fix_affine_for_matlab(aff):
                    """Ensure affine matrix is valid for MATLAB affine3d:
                    last column = [0, 0, 0, 1]'"""
                    aff = np.array(aff, dtype=float)
                    if aff.shape == (4, 4):
                        aff[:, 3] = [0, 0, 0, 1]  # enforce valid form - fix COLUMN not row
                    return aff


                if self.eng is not None:
                    print("Using MATLAB Engine for imwarp transformation...")

                    try:
                        # Convert data to MATLAB format
                        func_data_matlab = matlab.double(func_data[:, :, :, 0].tolist())
                        atlas_data_matlab = matlab.double(atlas_data.tolist())

                        # --- Fix affines for MATLAB ---
                        func_affine_fixed = fix_affine_for_matlab(func_affine)
                        atlas_affine_fixed = fix_affine_for_matlab(atlas_affine)

                        func_affine_matlab = matlab.double(func_affine_fixed.tolist())
                        atlas_affine_matlab = matlab.double(atlas_affine_fixed.tolist())

                        # Set up MATLAB workspace
                        self.eng.workspace['func_data'] = func_data_matlab
                        self.eng.workspace['func_affine'] = func_affine_matlab
                        self.eng.workspace['atlas_affine'] = atlas_affine_matlab

                        # Create MATLAB niftiinfo-like structures
                        self.eng.eval("func_info = struct();", nargout=0)
                        self.eng.eval("func_info.Transform = affine3d(func_affine);", nargout=0)
                        self.eng.eval("atlas_info = struct();", nargout=0)
                        self.eng.eval("atlas_info.Transform = affine3d(atlas_affine);", nargout=0)

                        # Create inverse transform for atlas
                        self.eng.eval("atlas_invtran = invert(atlas_info.Transform);", nargout=0)

                        # ---- First imwarp step (func to world space) ----
                        self.eng.eval("fnc_rawdata_t = imwarp(func_data, func_info.Transform);", nargout=0)
                        fnc_rawdata_t_matlab = self.eng.workspace['fnc_rawdata_t']
                        fnc_rawdata_t = np.array(fnc_rawdata_t_matlab, dtype=np.float32)

                        # ---- Second imwarp step (world to atlas space) ----
                        self.eng.workspace['fnc_rawdata_t'] = matlab.double(fnc_rawdata_t.tolist())
                        self.eng.eval("fnc_rawdata_tt = imwarp(fnc_rawdata_t, atlas_invtran);", nargout=0)
                        fnc_rawdata_tt_matlab = self.eng.workspace['fnc_rawdata_tt']
                        fnc_rawdata_tt = np.array(fnc_rawdata_tt_matlab, dtype=np.float32)

                    except Exception as e:
                        print(f"MATLAB imwarp failed: {e}")
                        print("Falling back to nibabel resampling...")
                        # Set flag to use fallback
                        self.eng = None
                        
                if self.eng is None:
                    print("Using nibabel resampling (fallback)...")
                    # Fallback to original nibabel method
                    from nibabel.processing import resample_from_to
                    func_first_vol_img = nib.Nifti1Image(fnc_rawdata_t, func_affine)
                    expected_affine = atlas_affine.copy()
                    atlas_ref_img = nib.Nifti1Image(np.zeros(expected_output_shape, dtype=np.float32), expected_affine)
                    resampled_img = resample_from_to(func_first_vol_img, atlas_ref_img, order=1)
                    fnc_rawdata_tt = resampled_img.get_fdata().astype(np.float32)
                    
                    # Also create atlas reference for fallback
                    atlas_img_nib = nib.Nifti1Image(atlas_data, atlas_affine)
                    resampled_atlas_img = resample_from_to(atlas_img_nib, atlas_ref_img, order=0)
                    resampled_atlas = resampled_atlas_img.get_fdata().astype(np.int32)
                
                # Calculate shift indices like MATLAB for comparison
                def get_matlab_style_indices(data_shape, atlas_shape):
                    results = {}
                    for dim, (data_dim, atlas_dim) in enumerate(zip(data_shape, atlas_shape)):
                        dim_name = ['X', 'Y', 'Z'][dim]
                        if data_dim > atlas_dim:
                            # When data is larger than atlas, we need to crop the data
                            xshift = (data_dim - atlas_dim) // 2
                            # Atlas placement: fill entire atlas
                            axs = 1  # MATLAB 1-based
                            axe = atlas_dim  # MATLAB 1-based inclusive
                            # Data extraction: extract middle portion
                            dxs = xshift + 1  # MATLAB 1-based
                            dxe = xshift + atlas_dim  # MATLAB 1-based inclusive
                            print(f"{dim_name} shift (data > atlas): xshift = {xshift}")
                            results[f'ax{dim_name.lower()}s'] = axs
                            results[f'ax{dim_name.lower()}e'] = axe
                            results[f'dx{dim_name.lower()}s'] = dxs
                            results[f'dx{dim_name.lower()}e'] = dxe
                        else:
                            # When atlas is larger than data, we need to pad/center the data in atlas
                            xshift = (atlas_dim - data_dim) // 2
                            # Atlas placement: place data in center of atlas
                            axs = xshift + 1  # MATLAB 1-based
                            axe = xshift + data_dim  # MATLAB 1-based inclusive
                            # Data extraction: take all data
                            dxs = 1  # MATLAB 1-based
                            dxe = data_dim  # MATLAB 1-based inclusive
                            print(f"{dim_name} shift (atlas > data): xshift = {xshift}")
                            results[f'ax{dim_name.lower()}s'] = axs
                            results[f'ax{dim_name.lower()}e'] = axe
                            results[f'dx{dim_name.lower()}s'] = dxs
                            results[f'dx{dim_name.lower()}e'] = dxe
                    return results
                indices = get_matlab_style_indices(fnc_rawdata_tt.shape, atlas_data.shape)
                
                resampled_func = np.zeros(expected_output_shape + (func_data.shape[3],), dtype=func_data.dtype)
                
                if self.eng is not None:
                    print(f"Processing all {func_data.shape[3]} timepoints with MATLAB imwarp...")
                    
                    try:
                        for t in range(func_data.shape[3]):
                            if t % 10 == 0:  # Progress indicator
                                print(f"  Processing timepoint {t+1}/{func_data.shape[3]}")
                            
                            # Convert current timepoint to MATLAB format
                            func_vol_matlab = matlab.double(func_data[:, :, :, t].tolist())
                            self.eng.workspace['func_vol'] = func_vol_matlab
                            # Apply two-step transformation using MATLAB imwarp
                            # Step 1: Apply functional transform (unchanged shape like MATLAB)
                            self.eng.eval("fnc_rawdata_t_vol = imwarp(func_vol, func_info.Transform);", nargout=0)
                            
                            # Step 2: Apply atlas transform (shape changes like MATLAB)
                            self.eng.eval("fnc_rawdata_tt_vol = imwarp(fnc_rawdata_t_vol, atlas_invtran);", nargout=0)
                            
                            # Get result back to Python
                            fnc_rawdata_tt_vol_matlab = self.eng.workspace['fnc_rawdata_tt_vol']
                            resampled_func[..., t] = np.array(fnc_rawdata_tt_vol_matlab, dtype=np.float32)
                            
                    except Exception as e:
                        print(f"MATLAB imwarp failed for timepoint processing: {e}")
                        print("MATLAB Engine disabled for this session.")
                        self.eng = None
                        
                if self.eng is None:
                    print("Cannot process without MATLAB Engine. Please restart the application.")
                    return
                    
                resampled_atlas = atlas_data  # Use original atlas as MATLAB does
                
                print(f"✓ MATLAB approach confirmed:")
                print(f"  - Functional data resampled to: {resampled_func.shape[:3]}")
                print(f"  - Atlas remains in original space: {resampled_atlas.shape}")
                # Use the original atlas shape as the target (exactly like MATLAB)
                target_shape = atlas_data.shape  
                # Initialize output arrays with exact target shape (original atlas shape)
                final_func = np.zeros(target_shape + (func_data.shape[3],), dtype=resampled_func.dtype)
                final_atlas = resampled_atlas  # Use original atlas directly (no cropping needed)
                
                axs = indices.get('axxs', 1) - 1  # Convert start to 0-based
                axe = indices.get('axxe', target_shape[0])  # End stays same (becomes exclusive in Python)
                ays = indices.get('axys', 1) - 1  # Convert start to 0-based  
                aye = indices.get('axye', target_shape[1])  # End stays same (becomes exclusive in Python)
                azs = indices.get('axzs', 1) - 1  # Convert start to 0-based
                aze = indices.get('axze', target_shape[2])  # End stays same (becomes exclusive in Python)
                
                dxs = indices.get('dxxs', 1) - 1  # Convert start to 0-based
                dxe = indices.get('dxxe', resampled_func.shape[0])  # End stays same (becomes exclusive in Python)
                dys = indices.get('dxys', 1) - 1  # Convert start to 0-based
                dye = indices.get('dxye', resampled_func.shape[1])  # End stays same (becomes exclusive in Python)
                dzs = indices.get('dxzs', 1) - 1  # Convert start to 0-based
                dze = indices.get('dxze', resampled_func.shape[2])  # End stays same (becomes exclusive in Python)
                final_func[axs:axe, ays:aye, azs:aze, :] = resampled_func[dxs:dxe, dys:dye, dzs:dze, :]
                func_data = final_func
                atlas_for_calculation = final_atlas  # Use cropped atlas for ROI extraction
                
                # Verify spatial dimensions match (essential for atlas calculation)
                assert func_data.shape[:3] == atlas_for_calculation.shape, f"Shape mismatch: func {func_data.shape[:3]} != atlas {atlas_for_calculation.shape}"
                print(f'✓ Spatial alignment verified: func={func_data.shape[:3]}, atlas={atlas_for_calculation.shape}')
                
                # Calculate atlased data (MATLAB equivalent)
                print(f'{i+1}/{fnc_pro_filelength}: Applying Atlas')
                
                # Use the MATLAB-style cropped atlas and functional data for ROI extraction
                fnc_rawdata_len = func_data.shape[3]
                
                atlased_data = np.zeros((fnc_rawdata_len, atl_len), dtype=np.float32)  # Match MATLAB single precision
                fnc_pro_atlas_masks_flat = atlas_for_calculation.flatten(order='F')  # Use MATLAB-style cropped atlas
                fnc_data_flat = func_data.reshape(atlas_for_calculation.shape[0] * atlas_for_calculation.shape[1] * atlas_for_calculation.shape[2], fnc_rawdata_len, order='F')  # Use MATLAB-style cropped data
            
                for ii in range(1, atl_len + 1):  
                    temp_pos = (fnc_pro_atlas_masks_flat == ii)
                    temp_value = fnc_data_flat[temp_pos, :]
                    roi_timeseries = np.nanmean(temp_value, axis=0)
                    atlased_data[:, ii-1] = roi_timeseries
            else:
                # .mat input: load full matrix (time × regions)
                mat = loadmat(file_path)
                key = next(k for k in mat if not k.startswith('__'))
                atlased_data = mat[key].astype(np.float32)
                fnc_raw_len, _ = atlased_data.shape
                fnc_window_count = fnc_raw_len - fnc_window_size + 1
                atl_len = atlased_data.shape[1]

            progress_dialog.setValue(i+1)

            # ——— Apply temporal filter if requested ———
            tr = self.settings['TR']
            if filter_type == 1:
                # Bandpass filter - use MATLAB Engine for exact match
                if self.eng is not None:
                    try:
                        # Convert to MATLAB array and apply bandpass column by column
                        atlased_data_matlab = matlab.double(atlased_data.tolist())
                        atl_len = atlased_data.shape[1]
                        
                        # Convert frequency range to MATLAB double array
                        freq_range = matlab.double([1.0/filter_setting2, 1.0/filter_setting1])
                        sample_rate = 1.0/tr
                        
                        # Apply bandpass to each column (ROI) separately to match MATLAB loop
                        for ii in range(atl_len):
                            # Extract column ii (0-indexed in Python)
                            column_data = atlased_data[:, ii]
                            column_matlab = matlab.double(column_data.tolist())
                            
                            # Apply bandpass with MATLAB's parameter order
                            filtered_column = self.eng.bandpass(column_matlab, 
                                                                      freq_range, 
                                                                      sample_rate)
                            # Convert back to numpy and store
                            atlased_data[:, ii] = np.array(filtered_column).flatten()
                        
                        print(f"Applied MATLAB bandpass filter: {1.0/filter_setting2:.4f} - {1.0/filter_setting1:.4f} Hz")
                    except Exception as e:
                        print(f"MATLAB bandpass failed: {e}, falling back to scipy")
                        # Fallback to original scipy implementation
                        fs = 1.0 / tr
                        nyq = fs / 2.0
                        low = (1.0 / filter_setting2) / nyq
                        high = (1.0 / filter_setting1) / nyq
                        b, a = butter(2, [low, high], btype='band')
                        atlased_data = filtfilt(b, a, atlased_data, axis=0)
                else:
                    # Fallback to original scipy implementation
                    fs = 1.0 / tr
                    nyq = fs / 2.0
                    low = (1.0 / filter_setting2) / nyq
                    high = (1.0 / filter_setting1) / nyq
                    b, a = butter(2, [low, high], btype='band')
                    atlased_data = filtfilt(b, a, atlased_data, axis=0)
            elif filter_type == 2:
                # Highpass filter - use MATLAB Engine for exact match
                if self.eng is not None:
                    try:
                        # Convert cutoff frequency to MATLAB double
                        cutoff_freq = 1.0/filter_setting1
                        sample_rate = 1.0/tr
                        
                        # Apply highpass to each column (ROI) separately to match MATLAB loop
                        for ii in range(atlased_data.shape[1]):
                            # Extract column ii (0-indexed in Python)
                            column_data = atlased_data[:, ii]
                            column_matlab = matlab.double(column_data.tolist())
                            
                            # Apply highpass with MATLAB's parameter order
                            filtered_column = self.eng.highpass(column_matlab, 
                                                               cutoff_freq, 
                                                               sample_rate)
                            # Convert back to numpy and store
                            atlased_data[:, ii] = np.array(filtered_column).flatten()
                        
                        print(f"Applied MATLAB highpass filter: {cutoff_freq:.4f} Hz")
                    except Exception as e:
                        print(f"MATLAB highpass failed: {e}, falling back to scipy")
                        # Fallback to original scipy implementation
                        fs = 1.0 / tr
                        nyq = fs / 2.0
                        cutoff = (1.0 / filter_setting1) / nyq
                        b, a = butter(2, cutoff, btype='high')
                        atlased_data = filtfilt(b, a, atlased_data, axis=0)
                else:
                    # Fallback to original scipy implementation
                    fs = 1.0 / tr
                    nyq = fs / 2.0
                    cutoff = (1.0 / filter_setting1) / nyq
                    b, a = butter(2, cutoff, btype='high')
                    atlased_data = filtfilt(b, a, atlased_data, axis=0)
            elif filter_type == 3:
                # Lowpass filter - use MATLAB Engine for exact match
                if self.eng is not None:
                    try:
                        # Convert cutoff frequency to MATLAB double
                        cutoff_freq = 1.0/filter_setting1
                        sample_rate = 1.0/tr
                        
                        # Apply lowpass to each column (ROI) separately to match MATLAB loop
                        for ii in range(atlased_data.shape[1]):
                            # Extract column ii (0-indexed in Python)
                            column_data = atlased_data[:, ii]
                            column_matlab = matlab.double(column_data.tolist())
                            
                            # Apply lowpass with MATLAB's parameter order
                            filtered_column = self.eng.lowpass(column_matlab, 
                                                              cutoff_freq, 
                                                              sample_rate)
                            # Convert back to numpy and store
                            atlased_data[:, ii] = np.array(filtered_column).flatten()
                        
                        print(f"Applied MATLAB lowpass filter: {cutoff_freq:.4f} Hz")
                    except Exception as e:
                        print(f"MATLAB lowpass failed: {e}, falling back to scipy")
                        # Fallback to original scipy implementation
                        fs = 1.0 / tr
                        nyq = fs / 2.0
                        cutoff = (1.0 / filter_setting1) / nyq
                        b, a = butter(2, cutoff, btype='low')
                        atlased_data = filtfilt(b, a, atlased_data, axis=0)
                else:
                    # Fallback to original scipy implementation
                    fs = 1.0 / tr
                    nyq = fs / 2.0
                    cutoff = (1.0 / filter_setting1) / nyq
                    b, a = butter(2, cutoff, btype='low')
                    atlased_data = filtfilt(b, a, atlased_data, axis=0)
            elif filter_type == 4:
                # Wavelet denoising per column - use MATLAB Engine for exact match
                if self.eng is not None:
                    try:
                        # Apply wavelet denoising to each column (ROI) separately to match MATLAB loop
                        for col in range(atlased_data.shape[1]):
                            # Extract column data
                            column_data = atlased_data[:, col]
                            column_matlab = matlab.double(column_data.tolist())
                            
                            # Set variables in MATLAB workspace  
                            self.eng.workspace['column_data'] = column_matlab
                            self.eng.workspace['decomp_level'] = filter_setting1
                            self.eng.workspace['selection_level'] = filter_setting2
                            
                            # Apply MATLAB MODWT (Maximal Overlap Discrete Wavelet Transform)
                            self.eng.eval("temp_data = modwt(column_data, decomp_level);", nargout=0)
                            
                            # Create index of excluded channels and zero them out
                            self.eng.eval("temp_idx = 1:(decomp_level+1);", nargout=0)
                            self.eng.eval("temp_idx(selection_level) = [];", nargout=0)
                            self.eng.eval("temp_data(temp_idx, :) = 0;", nargout=0)
                            
                            # Reconstruct the signal using IMODWT
                            self.eng.eval("reconstructed = imodwt(temp_data);", nargout=0)
                            
                            # Get the result back to Python
                            reconstructed_matlab = self.eng.workspace['reconstructed']
                            reconstructed_array = np.array(reconstructed_matlab).flatten()
                            
                            # Store result 
                            atlased_data[:, col] = reconstructed_array
                        
                        print(f"Applied MATLAB MODWT filter: level={filter_setting1}, selection={filter_setting2}")
                    except Exception as e:
                        print(f"MATLAB MODWT failed: {e}, falling back to pywt")
                        # Fallback to original pywt implementation
                        for col in range(atlased_data.shape[1]):
                            coeffs = pywt.wavedec(atlased_data[:, col], 'db1', level=filter_setting1)
                            for idx in range(len(coeffs)):
                                if idx+1 != filter_setting2:
                                    coeffs[idx] = np.zeros_like(coeffs[idx])
                            atlased_data[:, col] = pywt.waverec(coeffs, 'db1')[:fnc_raw_len]
                else:
                    # Fallback to original pywt implementation
                    for col in range(atlased_data.shape[1]):
                        coeffs = pywt.wavedec(atlased_data[:, col], 'db1', level=filter_setting1)
                        for idx in range(len(coeffs)):
                            if idx+1 != filter_setting2:
                                coeffs[idx] = np.zeros_like(coeffs[idx])
                        atlased_data[:, col] = pywt.waverec(coeffs, 'db1')[:fnc_raw_len]

            # ——— Sliding-window correlation ———
            corr_data = np.zeros((fnc_window_count, atl_len, atl_len), dtype=np.float32)
            for w in range(fnc_window_count):
                window = atlased_data[w:w+fnc_window_size, :]
                if self.settings['kernel'] == 2:
                    idx = np.arange(1, fnc_window_size+1)
                    center = fnc_window_size/2 + 1
                    gauss = np.exp(-((idx-center)**2)/(2*(self.settings['kernel_setting']**2)))
                    window = window * gauss[:, None]
                tmp = np.corrcoef(window, rowvar=False)
                tmp[np.isnan(tmp)] = 0
                corr_data[w] = tmp

            # Save results to .mat
            subj_data = {
                'd_atlas': atlased_data,
                'd_corr': corr_data,
                'd_setting': self.settings,
                'd_windowsize': self.settings['window_size'],
                'd_stepsize': self.settings['step_size'],
                'd_kernel': self.settings['kernel'],
                'd_atlas_list': self.settings['atlas_list']
            }
            fname = os.path.splitext(self.settings['file_list'][i])[0]
            savemat(os.path.join(fnc_out_path, f"{fname}.mat"), {'subj_data': subj_data}, format='5', do_compression=True)
            progress_dialog.setLabelText(f"{i+1}/{fnc_pro_filelength}: Done")

        # Close dialog and notify completion
        progress_dialog.close()
        QMessageBox.information(self, "Finished", "Finished")

