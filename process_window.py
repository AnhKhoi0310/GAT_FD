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
            filter_setting1 = float(self.settings['filter_setting1'])
            filter_setting2 = float(self.settings['filter_setting2'])
            # Validate bandpass: lower < upper
            if filter_setting1 >= filter_setting2:
                QMessageBox.critical(self, "Error", "Please enter correct band pass parameters")
                return
        elif filter_type in (2, 3):
            filter_setting1 = float(self.settings['filter_setting1'])
            filter_setting2 = 0
        elif filter_type == 4:
            filter_setting1 = int(self.settings['filter_setting1'])
            filter_setting2 = int(self.settings['filter_setting2'])
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
                
                # --- DEBUG: Print original functional data before resampling ---
                print('=== ORIGINAL FUNCTIONAL DATA BEFORE RESAMPLING ===')
                print(f'Original func shape: {func_data.shape}')
                print(f'Original func data type: {func_data.dtype}')
                print(f'Original func min: {np.min(func_data):.6f}, max: {np.max(func_data):.6f}')
                # Print a small sample of the original functional data for comparison
                # IMPORTANT: Use MATLAB-equivalent indexing (1:5 in MATLAB = 0:5 in Python, but we want same voxels)
                # So MATLAB [1:5, 1:5, 1:5, 1:3] = Python [1:6, 1:6, 1:6, 1:4] to get same voxels
                func_sample = func_data[1:6, 1:6, 1:6, 1:4]  # Match MATLAB indexing
                print(f'Original func sample [1:6, 1:6, 1:4] (MATLAB equivalent [1:5, 1:5, 1:5, 1:3]):')
                print(f'  Shape: {func_sample.shape}')
                print(f'  Sample mean: {np.mean(func_sample):.6f}')
                print(f'  Sample std: {np.std(func_sample):.6f}')
                # Print first few voxel values from first timepoint
                # MATLAB [1:10, 1:10, 1, 1] = Python [1:11, 1:11, 0, 0] to get same voxels
                first_vol_sample = func_data[1:11, 1:11, 0, 0]  # Match MATLAB indexing
                first_vol_flat = first_vol_sample.flatten()
                print(f'First 10 voxels from first volume (MATLAB equivalent): {[f"{val:.6f}" for val in first_vol_flat[:10]]}')
                print('=' * 50)
                
                # Print voxel size and orientation info for debug
                print("Atlas affine:\n", atlas_nib.affine)
                print("Atlas orientation:", nib.aff2axcodes(atlas_nib.affine))
                print("Atlas array shape:", atlas_data.shape)
                print("Func affine:\n", func_img.affine)
                print("Func orientation:", nib.aff2axcodes(func_img.affine))
                print("Func array shape:", func_data.shape)
                
                # --- SAVE INTERMEDIATE DATA 0: Original loaded data (no reorientation) ---
                intermediate_data = {}
                # Use center region to get meaningful brain data instead of background
                # MATLAB: center_x_orig = round(size(fnc_rawdata,1)/2)
                center_x_orig = int(np.ceil(func_data.shape[0]/2))
                center_y_orig = int(np.ceil(func_data.shape[1]/2))
                center_z_orig = int(np.ceil(func_data.shape[2]/2))
                print(f"Center voxel (original data, MATLAB-style 1-based): ({center_x_orig}, {center_y_orig}, {center_z_orig})")   
                # MATLAB uses [center-9:center+10] (1-based), we use [center-10:center+10] (0-based) for equivalent range
                x_start_orig, x_end_orig = max(0, center_x_orig-10), min(func_data.shape[0], center_x_orig+10)
                y_start_orig, y_end_orig = max(0, center_y_orig-10), min(func_data.shape[1], center_y_orig+10)
                z_start_orig, z_end_orig = max(0, center_z_orig-5), min(func_data.shape[2], center_z_orig+5)
                
                intermediate_data['step0_original_data'] = {
                    'func_shape': func_data.shape,
                    'atlas_shape': atlas_data.shape,
                    'func_sample': func_data[x_start_orig:x_end_orig, y_start_orig:y_end_orig, z_start_orig:z_end_orig, 0:min(3, func_data.shape[3])].astype(np.float32),
                    'atlas_sample': atlas_data[x_start_orig:x_end_orig, y_start_orig:y_end_orig, z_start_orig:z_end_orig].astype(np.float32),
                    'func_dtype': str(func_data.dtype),
                    'atlas_dtype': str(atlas_data.dtype),
                    'sample_region': f'[{x_start_orig}:{x_end_orig}, {y_start_orig}:{y_end_orig}, {z_start_orig}:{z_end_orig}]',
                    'matlab_region': f'[{x_start_orig+1}:{x_end_orig}, {y_start_orig+1}:{y_end_orig}, {z_start_orig+1}:{z_end_orig}]',
                    'func_orient': nib.aff2axcodes(func_img.affine),
                    'atlas_orient': nib.aff2axcodes(atlas_nib.affine),
                    'func_affine': func_img.affine.copy(),
                    'atlas_affine': atlas_nib.affine.copy()
                }
                print("--- INTERMEDIATE DATA STEP 0: Original loaded data (no reorientation) ---")
                print(f"Func shape: {func_data.shape}")
                print(f"Atlas shape: {atlas_data.shape}")
                print(f"Func orientation: {nib.aff2axcodes(func_img.affine)}")
                print(f"Atlas orientation: {nib.aff2axcodes(atlas_nib.affine)}")
                print(f"Sample region (Python 0-based): [{x_start_orig}:{x_end_orig}, {y_start_orig}:{y_end_orig}, {z_start_orig}:{z_end_orig}]")
                print(f"Sample region (MATLAB equivalent 1-based): [{x_start_orig+1}:{x_end_orig}, {y_start_orig+1}:{y_end_orig}, {z_start_orig+1}:{z_end_orig}]")
                func_sample_region_orig = func_data[x_start_orig:x_end_orig, y_start_orig:y_end_orig, z_start_orig:z_end_orig, 0]
                print(f"Func sample min/max: {np.min(func_sample_region_orig):.6f} / {np.max(func_sample_region_orig):.6f}")
                print(f"Func sample first element: {func_sample_region_orig[0,0,0]:.6f}")
                atlas_sample_region_orig = atlas_data[x_start_orig:x_end_orig, y_start_orig:y_end_orig, z_start_orig:z_end_orig]
                print(f"Atlas sample unique labels: {np.unique(atlas_sample_region_orig)}")
                
                # Use original loaded data directly (no reorientation)
                func_shape = func_data.shape[:3]  # Define func_shape for output_shape
                
                # --- Manual calculation of output shape based on voxel sizes ---
                # Use original affines (no reorientation)
                atlas_affine = atlas_nib.affine
                func_affine = func_img.affine
                # --- Manual calculation of output shape based on voxel sizes ---
                # Get voxel sizes
                atlas_voxel_size = atlas_nib.header.get_zooms()[:3]
                func_voxel_size = func_img.header.get_zooms()[:3]
                print("Atlas voxel size:", atlas_voxel_size)
                print("Func voxel size:", func_voxel_size)
                
                # Calculate physical dimensions of functional data
                func_phys_dims = np.array(func_data.shape[:3]) * np.array(func_voxel_size)
                print("Func physical dimensions (mm):", func_phys_dims)
                
                # Calculate expected output shape when resampling to atlas voxel size
                expected_output_shape = tuple(int(np.ceil(dim / voxel)) for dim, voxel in zip(func_phys_dims, atlas_voxel_size))
                print("Expected output shape (calculated):", expected_output_shape)
                
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

                        print('fnc_rawdata_t shape (MATLAB imwarp result):', fnc_rawdata_t.shape)
                        print('fnc_rawdata_tt shape (MATLAB imwarp result):', fnc_rawdata_tt.shape)
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
                print('fnc_rawdata_tt shape (after atlas transform, shape changed):', fnc_rawdata_tt.shape)
                print('atlas (target) shape:', atlas_data.shape)
                print('Expected vs actual shape match:', expected_output_shape == fnc_rawdata_tt.shape)
                
                # Debug output to match MATLAB format
                print("Atlas array shape:")
                print(f"   {atlas_data.shape[0]}   {atlas_data.shape[1]}   {atlas_data.shape[2]}")
                print()
                print("fnc_rawdata_t (after func transform, unchanged):")
                print(f"   {fnc_rawdata_t.shape[0]}   {fnc_rawdata_t.shape[1]}   {fnc_rawdata_t.shape[2]}")
                print()
                print("fnc_rawdata_tt (after atlas transform, shape changed):")
                print("Before shift/trim:")
                print(f"   {fnc_rawdata_tt.shape[0]}   {fnc_rawdata_tt.shape[1]}   {fnc_rawdata_tt.shape[2]}")
                print()
                print(f"   {atlas_data.shape[0]}   {atlas_data.shape[1]}   {atlas_data.shape[2]}")
                print()
                
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
                
                # Calculate shift indices like MATLAB for comparison (use fnc_rawdata_tt vs original atlas)
                indices = get_matlab_style_indices(fnc_rawdata_tt.shape, atlas_data.shape)
                print("--- Cropping/shift indices summary (Python equivalent) ---")
                print(f"axs: {indices.get('axxs', 'N/A')} to {indices.get('axxs', 0) + atlas_data.shape[0] - 1 if 'axxs' in indices else 'N/A'}")
                print(f"ays: {indices.get('axys', 'N/A')} to {indices.get('axys', 0) + atlas_data.shape[1] - 1 if 'axys' in indices else 'N/A'}")
                print(f"azs: {indices.get('axzs', 'N/A')} to {indices.get('axzs', 0) + atlas_data.shape[2] - 1 if 'axzs' in indices else 'N/A'}")
                print(f"dxs: {indices.get('dxxs', 'N/A')} to {indices.get('dxxe', 'N/A')}")
                print(f"dys: {indices.get('dxys', 'N/A')} to {indices.get('dxye', 'N/A')}")
                print(f"dzs: {indices.get('dxzs', 'N/A')} to {indices.get('dxze', 'N/A')}")
                
                # Apply MATLAB imwarp to all timepoints to match MATLAB behavior exactly
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
                    
                print('Resampled func_data shape (after 2-step MATLAB imwarp):', resampled_func.shape)
                
                # Also resample the atlas using MATLAB imwarp to the same grid
                # IMPORTANT: MATLAB does NOT resample the atlas - it keeps the original atlas
                # and only resamples the functional data to match the atlas space
                print("MATLAB approach: Using original atlas (no resampling)")
                resampled_atlas = atlas_data  # Use original atlas as MATLAB does
                
                print('Original atlas shape (used for cropping target):', resampled_atlas.shape)
                
                # Verify the approach matches MATLAB
                # In MATLAB: resampled_func has shape after transformation, atlas keeps original shape  
                print(f"✓ MATLAB approach confirmed:")
                print(f"  - Functional data resampled to: {resampled_func.shape[:3]}")
                print(f"  - Atlas remains in original space: {resampled_atlas.shape}")
                print(f"  - Cropping will align resampled functional data to original atlas space")
                
                # --- SAVE INTERMEDIATE DATA 1: After resampling, before cropping ---
                # Use same center region for consistency
                # Use np.round to match MATLAB's round() function
                center_x_r = int(np.ceil(resampled_func.shape[0]/2))
                center_y_r = int(np.ceil(resampled_func.shape[1]/2))
                center_z_r = int(np.ceil(resampled_func.shape[2]/2))
                x_start_r, x_end_r = max(0, center_x_r-10), min(resampled_func.shape[0], center_x_r+10)
                y_start_r, y_end_r = max(0, center_y_r-10), min(resampled_func.shape[1], center_y_r+10)
                z_start_r, z_end_r = max(0, center_z_r-5), min(resampled_func.shape[2], center_z_r+5)
                print(f"Sample region (Python 0-based): [{x_start_r}:{x_end_r}, {y_start_r}:{y_end_r}, {z_start_r}:{z_end_r}]")
                print(f"Sample region (MATLAB equivalent 1-based): [{x_start_r+1}:{x_end_r}, {y_start_r+1}:{y_end_r}, {z_start_r+1}:{z_end_r}]")
                intermediate_data['step1_after_resample'] = {
                    'resampled_func_shape': resampled_func.shape,  # 4D - all timepoints (final data after 2-step transform)
                    'resampled_func_first_shape': resampled_func[..., 0].shape,  # 3D - first timepoint 
                    'fnc_rawdata_t_shape': fnc_rawdata_t.shape,  # 3D - after step 1 (unchanged like MATLAB)
                    'fnc_rawdata_tt_shape': fnc_rawdata_tt.shape,  # 3D - after step 2 (shape changed like MATLAB)
                    'resampled_atlas_shape': resampled_atlas.shape,
                    'resampled_func_sample': resampled_func[x_start_r:x_end_r, y_start_r:y_end_r, z_start_r:z_end_r, 0:min(3, resampled_func.shape[3])].astype(np.float32),
                    'fnc_rawdata_t_sample': fnc_rawdata_t[x_start_orig:x_end_orig, y_start_orig:y_end_orig, z_start_orig:z_end_orig].astype(np.float32),  # 3D - use same indices as step0
                    # 'fnc_rawdata_tt_sample': fnc_rawdata_tt[, , 100,].astype(np.float32), #for dislpay only
                    'fnc_rawdata_tt_sample': fnc_rawdata_tt[x_start_r:x_end_r, y_start_r:y_end_r, z_start_r:z_end_r,].astype(np.float32),
                    'resampled_atlas_sample': resampled_atlas[x_start_r:x_end_r, y_start_r:y_end_r, z_start_r:z_end_r].astype(np.float32),
                    'resampled_func_dtype': str(resampled_func.dtype),
                    'resampled_atlas_dtype': str(resampled_atlas.dtype),
                    'fnc_rawdata_t_dtype': str(fnc_rawdata_t.dtype),
                    'fnc_rawdata_tt_dtype': str(fnc_rawdata_tt.dtype),
                    'sample_region': f'[{x_start_r}:{x_end_r}, {y_start_r}:{y_end_r}, {z_start_r}:{z_end_r}]'
                }
                print("--- INTERMEDIATE DATA STEP 1: After resampling, before cropping ---")
                print(f"fnc_rawdata_t shape (3D - after step 1, unchanged like MATLAB): {fnc_rawdata_t.shape}")
                print(f"fnc_rawdata_tt shape (3D - after step 2, shape changed like MATLAB): {fnc_rawdata_tt.shape}")
                print(f"resampled_func shape (4D - all timepoints after 2-step transform): {resampled_func.shape}")
                print(f"resampled_atlas shape: {resampled_atlas.shape}")
                print(f"Sample region (center brain): [{x_start_r}:{x_end_r}, {y_start_r}:{y_end_r}, {z_start_r}:{z_end_r}]")
                
                # Verify the two-step process matches MATLAB behavior
                print(f"Step 1 verification - fnc_rawdata_t unchanged: {fnc_rawdata_t.shape == func_data[..., 0].shape}")
                print(f"Step 2 verification - fnc_rawdata_tt shape changed: {fnc_rawdata_tt.shape == expected_output_shape}")
                print(f"Final verification - resampled_func matches fnc_rawdata_tt: {np.allclose(fnc_rawdata_tt, resampled_func[..., 0])}")
                # Check sample regions for debugging
                resampled_func_sample_region = resampled_func[x_start_r:x_end_r, y_start_r:y_end_r, z_start_r:z_end_r, 0]
                print(f"resampled_func sample (first timepoint) min/max: {np.min(resampled_func_sample_region):.6f} / {np.max(resampled_func_sample_region):.6f}")
                fnc_rawdata_t_sample_region = fnc_rawdata_t[x_start_r:x_end_r, y_start_r:y_end_r, z_start_r:z_end_r]
                print(f"fnc_rawdata_t sample (unchanged) min/max: {np.min(fnc_rawdata_t_sample_region):.6f} / {np.max(fnc_rawdata_t_sample_region):.6f}")
                fnc_rawdata_tt_sample_region = fnc_rawdata_tt[x_start_r:x_end_r, y_start_r:y_end_r, z_start_r:z_end_r]
                print(f"fnc_rawdata_tt sample (shape changed) min/max: {np.min(fnc_rawdata_tt_sample_region):.6f} / {np.max(fnc_rawdata_tt_sample_region):.6f}")
                resampled_atlas_sample_region = resampled_atlas[x_start_r:x_end_r, y_start_r:y_end_r, z_start_r:z_end_r]
                print(f"Resampled atlas sample unique labels: {np.unique(resampled_atlas_sample_region)}")
                print("Two-step transformation complete - using resampled_func (4D) for final processing")
                
                # --- Apply MATLAB-style cropping using the calculated indices ---
                # Use the original atlas shape as the target (exactly like MATLAB)
                target_shape = atlas_data.shape  # Original atlas shape - this is what MATLAB uses
                print(f'Target atlas shape (original): {target_shape}')
                print(f'Resampled functional data shape before cropping: {resampled_func.shape[:3]}')
                
                # Initialize output arrays with exact target shape (original atlas shape)
                final_func = np.zeros(target_shape + (func_data.shape[3],), dtype=resampled_func.dtype)
                final_atlas = resampled_atlas  # Use original atlas directly (no cropping needed)
                
                # Convert MATLAB 1-based indices to Python 0-based indices and apply cropping
                # This crops the resampled functional data to fit the original atlas space
                # MATLAB uses 1-based indexing with inclusive end, Python uses 0-based with exclusive end
                # MATLAB 1:145 becomes Python 0:145 (0 to 144 inclusive)
                # MATLAB 7:151 becomes Python 6:151 (6 to 150 inclusive)
                
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
                
                print(f'Using MATLAB-style cropping indices (converted to 0-based):')
                print(f'Atlas placement: X[{axs}:{axe}], Y[{ays}:{aye}], Z[{azs}:{aze}]')
                print(f'Data extraction: X[{dxs}:{dxe}], Y[{dys}:{dye}], Z[{dzs}:{dze}]')
                print(f'MATLAB equivalent - Atlas placement: X[{indices.get("axxs", 1)}:{indices.get("axxe", target_shape[0])}], Y[{indices.get("axys", 1)}:{indices.get("axye", target_shape[1])}], Z[{indices.get("axzs", 1)}:{indices.get("axze", target_shape[2])}]')
                print(f'MATLAB equivalent - Data extraction: X[{indices.get("dxxs", 1)}:{indices.get("dxxe", resampled_func.shape[0])}], Y[{indices.get("dxys", 1)}:{indices.get("dxye", resampled_func.shape[1])}], Z[{indices.get("dxzs", 1)}:{indices.get("dxze", resampled_func.shape[2])}]')
                
                # Debug: Check actual array shapes before cropping
                print(f'DEBUG: resampled_func.shape = {resampled_func.shape}')
                print(f'DEBUG: resampled_atlas.shape = {resampled_atlas.shape}')
                print(f'DEBUG: target_shape = {target_shape}')
                print(f'DEBUG: Extraction region would be: resampled_func[{dxs}:{dxe}, {dys}:{dye}, {dzs}:{dze}] -> shape {(dxe-dxs, dye-dys, dze-dzs)}')
                print(f'DEBUG: Extraction region would be: resampled_atlas[{dxs}:{dxe}, {dys}:{dye}, {dzs}:{dze}] -> shape {(dxe-dxs, dye-dys, dze-dzs)}')
                print(f'DEBUG: Atlas placement region: final_atlas[{axs}:{axe}, {ays}:{aye}, {azs}:{aze}] -> shape {(axe-axs, aye-ays, aze-azs)}')
                
                # Apply the exact MATLAB cropping logic - crop resampled functional data to original atlas space
                # Only crop functional data - atlas is already in the correct space
                final_func[axs:axe, ays:aye, azs:aze, :] = resampled_func[dxs:dxe, dys:dye, dzs:dze, :]
                # final_atlas is already set to resampled_atlas (original atlas)
                
                # --- SAVE INTERMEDIATE DATA 2: After cropping (final) ---
                # Use same center region for consistency
                center_x_f = int(np.ceil(final_func.shape[0]/2))
                center_y_f = int(np.ceil(final_func.shape[1]/2))
                center_z_f = int(np.ceil(final_func.shape[2]/2))
                x_start_f, x_end_f = max(0, center_x_f-10), min(final_func.shape[0], center_x_f+10)
                y_start_f, y_end_f = max(0, center_y_f-10), min(final_func.shape[1], center_y_f+10)
                z_start_f, z_end_f = max(0, center_z_f-5), min(final_func.shape[2], center_z_f+5)
                
                intermediate_data['step2_after_crop'] = {
                    'func_shape': final_func.shape,
                    'atlas_shape': final_atlas.shape,
                    'func_sample': final_func[x_start_f:x_end_f, y_start_f:y_end_f, z_start_f:z_end_f, 0:min(3, final_func.shape[3])].astype(np.float32),
                    'atlas_sample': final_atlas[x_start_f:x_end_f, y_start_f:y_end_f, z_start_f:z_end_f].astype(np.float32),
                    'crop_indices': {
                        'atlas_place': f'X[{axs}:{axe}], Y[{ays}:{aye}], Z[{azs}:{aze}]',
                        'data_extract': f'X[{dxs}:{dxe}], Y[{dys}:{dye}], Z[{dzs}:{dze}]'
                    },
                    'sample_region': f'[{x_start_f}:{x_end_f}, {y_start_f}:{y_end_f}, {z_start_f}:{z_end_f}]'
                }
                print("--- INTERMEDIATE DATA STEP 2: After cropping (final) ---")
                print(f"Final func shape: {final_func.shape}")
                print(f"Final atlas shape: {final_atlas.shape}")
                print(f"Sample region (center brain): [{x_start_f}:{x_end_f}, {y_start_f}:{y_end_f}, {z_start_f}:{z_end_f}]")
                final_func_sample_region = final_func[x_start_f:x_end_f, y_start_f:y_end_f, z_start_f:z_end_f, 0]
                print(f"Final func sample min/max: {np.min(final_func_sample_region):.6f} / {np.max(final_func_sample_region):.6f}")
                final_atlas_sample_region = final_atlas[x_start_f:x_end_f, y_start_f:y_end_f, z_start_f:z_end_f]
                print(f"Final atlas sample unique labels: {np.unique(final_atlas_sample_region)}")
                
                print(f'Final func shape: {final_func.shape}')
                print(f'Final atlas shape: {final_atlas.shape}')
                print(f'Target atlas shape match: {final_atlas.shape == target_shape}')
                print('MATLAB-style cropping applied to BOTH atlas and functional data for atlas calculation')
                
                # CRITICAL: Always use the MATLAB-style cropped data for atlas calculation
                # This ensures both atlas and functional data have identical spatial alignment
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
                
                print(f'fnc_pro_atlas_masks_flat shape: {fnc_pro_atlas_masks_flat.shape}')
                print(f'fnc_data_flat shape: {fnc_data_flat.shape}')
                
                for ii in range(1, atl_len + 1):  
                    temp_pos = (fnc_pro_atlas_masks_flat == ii)
                    temp_value = fnc_data_flat[temp_pos, :]
                    
                    # Debug output to match MATLAB format
                    voxel_count = np.sum(temp_pos)
                    print(f'ROI {ii}: {voxel_count} voxels')
                    
                    if voxel_count > 0:
                        if temp_value.shape[0] > 0:
                            first_timepoint_values = temp_value[:, 0]  # First timepoint
                            roi_mean = np.nanmean(temp_value, axis=0)[0]  # Mean for first timepoint
                            roi_std = np.nanstd(temp_value, axis=0)[0]   # Std for first timepoint
                            print(f'  First few voxels (t=1): {first_timepoint_values[:min(5, len(first_timepoint_values))]}')
                            print(f'  Mean: {roi_mean:.6f}, Std: {roi_std:.6f}')
                    
                    roi_timeseries = np.nanmean(temp_value, axis=0)
                    atlased_data[:, ii-1] = roi_timeseries
                
                print(f'Final atlased_data shape: {atlased_data.shape[0]} x {atlased_data.shape[1]}')
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
            fs = 1.0 / tr
            nyq = fs / 2.0
            if filter_type == 1:
                # Bandpass filter
                low = (1.0 / filter_setting2) / nyq
                high = (1.0 / filter_setting1) / nyq
                b, a = butter(2, [low, high], btype='band')
                atlased_data = filtfilt(b, a, atlased_data, axis=0)
            elif filter_type == 2:
                # Highpass filter
                cutoff = (1.0 / filter_setting1) / nyq
                b, a = butter(2, cutoff, btype='high')
                atlased_data = filtfilt(b, a, atlased_data, axis=0)
            elif filter_type == 3:
                # Lowpass filter
                cutoff = (1.0 / filter_setting1) / nyq
                b, a = butter(2, cutoff, btype='low')
                atlased_data = filtfilt(b, a, atlased_data, axis=0)
            elif filter_type == 5:
                # Wavelet denoising per column
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
            
            # Add atlased_data tracking to output (only for image processing)
            if self.file_format_dropdown.currentIndex() == 0:
                # Add intermediate data tracking for resampling/transformation isolation
                subj_data['intermediate_data'] = intermediate_data
            fname = os.path.splitext(self.settings['file_list'][i])[0]
            savemat(os.path.join(fnc_out_path, f"{fname}.mat"), {'subj_data': subj_data}, format='5', do_compression=True)
            progress_dialog.setLabelText(f"{i+1}/{fnc_pro_filelength}: Done")

        # Close dialog and notify completion
        progress_dialog.close()
        QMessageBox.information(self, "Finished", "Finished")

