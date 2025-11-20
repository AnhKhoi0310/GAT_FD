# view_window.py
from PyQt6.QtWidgets import ( QWidget, QLabel, QPushButton, QSlider, QGridLayout, QFileDialog, QMessageBox, QComboBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeyEvent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import scipy.io
import h5py

class ViewWindow(QWidget    ):
    def __init__(self):
        super().__init__()
        self.createComponents()

    def gatv_load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Conditions", "","MAT files (*.mat)")
        if not file_path:
            return
        
        try:
            # First try to load with scipy.io.loadmat (for older MATLAB formats)
            mat_data = scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)
            subj_data = mat_data['subj_data']
            if isinstance(subj_data, np.ndarray):
                subj_data = subj_data.item()  # unwrap 1x1 array to struct
            d_corr = subj_data.d_corr
        except NotImplementedError:
            # If that fails, try h5py for MATLAB v7.3 files
            print("Detected MATLAB v7.3 format, using HDF5 reader...")
            try:
                with h5py.File(file_path, 'r') as f:
                    # Navigate the HDF5 structure to find d_corr
                    subj_data_ref = f['subj_data']
                    d_corr_ref = subj_data_ref['d_corr']
                    # Read the data and transpose if needed (MATLAB vs Python array ordering)
                    d_corr = np.array(d_corr_ref).T  # Transpose to match MATLAB ordering
                    
            except Exception as hdf5_error:
                QMessageBox.critical(self, "Error", f"Failed to load file: {hdf5_error}")
                return
        except Exception as general_error:
            QMessageBox.critical(self, "Error", f"Failed to load file: {general_error}")
            return
        
        self.gatv_setting['networks'] = d_corr
        self.gatv_setting['frame_limit'] = [1, len(d_corr)]
        self.gatv_setting['frame_cur'] = 1  # Initialize frame cursor
        self.frame_slider.setMinimum(1)  # Start from 1 to match MATLAB indexing
        self.frame_slider.setMaximum(len(d_corr))  # Maximum frame number
        print("Loaded file:", file_path.split("/")[-1])
        print("d_corr shape:", d_corr.shape)
        print("Total frames:", len(d_corr))
        self.gatv_update_network()

    def gatv_load_design(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Conditions", "", "MAT files (*.mat);;Text files (*.txt)")
        if not file_path:
            self.gatv_setting['iscondition'] = 0
            return

        mat_contents = scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)
        window_condition = mat_contents.get('window_condition', None)
        if window_condition is None:
            QMessageBox.warning(self, "Error", "'window_condition' not found in file.")
            return

        # Unwrap MATLAB struct if needed
        if isinstance(window_condition, np.ndarray):
            window_condition = window_condition.item()

        self.gatv_setting['condition'] = window_condition
        self.gatv_setting['iscondition'] = 1

        # Parse design_duration_list (MATLAB: str2num)
        duration_str = window_condition.design_duration_list
        if isinstance(duration_str, (np.ndarray, list)):
            if hasattr(duration_str, 'tolist'):
                duration_str = ''.join(duration_str.tolist())
            else:
                duration_str = str(duration_str)
        design_duration_list = np.array([float(x) for x in duration_str.strip().split()])
        tr = float(window_condition.tr)
        dfnc_length = int(np.floor(np.sum(design_duration_list) / tr))
        dfnc_reponse = np.array(window_condition.dfnc_reponse).flatten()
        dfnc_design = np.array(window_condition.dfnc_design).flatten()

        self.task_design_ax.clear()
        self.gatv_plot['pholder_hrf'], = self.task_design_ax.plot(
            np.arange(1, dfnc_length + 1), dfnc_reponse, '--', color=(1, 0.32, 0.16), linewidth=2
        )
        self.task_design_ax.set_xlim([0, dfnc_length])
        self.task_design_ax.set_ylim([0, np.max(dfnc_reponse) + 0.1])
        self.gatv_plot['pholder_design'], = self.task_design_ax.plot(
            np.arange(1, dfnc_length + 1), dfnc_design, 'k', linewidth=1
        )

        w_start = self.gatv_setting.get('frame_cur', 1)
        w_end = w_start + int(window_condition.window_size)
        self.gatv_plot['pholder_window'] = self.task_design_ax.fill_between(
            [w_start, w_end, w_end, w_start], [0, 0, 3, 3], color=(1, 0.96, 0.4), alpha=0.2
        )
        self.task_design_canvas.draw()

        self.frame_slider.setMaximum(dfnc_length)
        self.gatv_data['data_length'] = len(dfnc_reponse)
        self.gatv_data['data_frame_length'] = len(np.array(window_condition.dfnc_window_condi).flatten())
        self.gatv_data['data_length_diff'] = int(np.floor(
            (self.gatv_data['data_length'] - self.gatv_data['data_frame_length']) / 2
        ))
    def gatv_update_network(self):
        if 'networks' in self.gatv_setting:
            self.gatv_setting['frame_cur'] = int(self.frame_slider.value())
            self.frame_slider_cursor.setText(str(self.gatv_setting['frame_cur']))
            self.network_ax.clear()
            data = self.gatv_setting['networks']
            
            # Ensure frame_cur is within valid bounds
            max_frames = len(data) if data.ndim == 3 else 1
            frame_idx = min(max(1, self.gatv_setting['frame_cur']), max_frames) - 1  # Convert to 0-based index
            
            if data.ndim == 3:
                matrix = data[frame_idx, :, :]
            else:
                matrix = data  # just one matrix, no frame dimension
            self.network_ax.imshow(matrix, cmap='viridis')
            self.network_canvas.draw()

    def createComponents(self):
        # Create GATFD_process_UIFigure 
        self.setWindowTitle("GAT-FD - Result Display v0.2a")
        self.setGeometry(100, 100, 832, 810)

        self.central_widget = QWidget()
        self.setLayout(QGridLayout())  # Directly set layout on the QWidget
        layout = self.layout()

        self.gatv_setting = {}
        self.gatv_data = {}
        self.gatv_plot = {}

        # UIAxes_Network
        self.network_canvas = FigureCanvas(Figure())
        self.network_ax = self.network_canvas.figure.add_subplot(111)
        self.network_ax.set_title("Network")
        layout.addWidget(self.network_canvas, 0, 1)

        # FrameSlider and Label
        # self.frame_slider_label = QLabel("Frame")
        # layout.addWidget(self.frame_slider_label, 0.25, 0)

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(1)  # Start from 1 to match MATLAB indexing
        self.frame_slider.setMaximum(100)
        self.frame_slider.setValue(1)  # Initialize to frame 1
        self.frame_slider.valueChanged.connect(self.gatv_update_network)
        layout.addWidget(self.frame_slider, 1, 0)

        self.frame_slider_cursor = QLabel("1")  # Initialize to frame 1
        layout.addWidget(self.frame_slider_cursor, 1, 2)

        # LoadProcessedFileButton
        self.load_processed_button = QPushButton("Load Processed File")
        self.load_processed_button.clicked.connect(self.gatv_load_file)
        layout.addWidget(self.load_processed_button, 2, 1)

        # UIAxes_TaskDesign
        self.task_design_canvas = FigureCanvas(Figure())
        self.task_design_ax = self.task_design_canvas.figure.add_subplot(111)
        layout.addWidget(self.task_design_canvas, 0, 0)

        # UIAxes_Timeseries
        self.timeseries_canvas = FigureCanvas(Figure())
        self.timeseries_ax = self.timeseries_canvas.figure.add_subplot(111)
        layout.addWidget(self.timeseries_canvas, 3, 0, 1, 3)

        # LoadTemporalMaskButton
        self.load_temporal_button = QPushButton("Load Temporal Mask")
        self.load_temporal_button.clicked.connect(self.gatv_load_design)
        layout.addWidget(self.load_temporal_button, 2, 0)

        # LoadNetworkPropertyFileButton
        self.load_property_button = QPushButton("Load Network Property File")
        # self.load_property_button.clicked.connect(self.gatv_load_measure)
        layout.addWidget(self.load_property_button, 4, 0)

        # Dropdown Labels and Widgets
        self.threshold_label = QLabel("Threshold")
        layout.addWidget(self.threshold_label, 5, 0)

        self.threshold_dropdown = QComboBox()
        self.threshold_dropdown.addItem("Mean")
        layout.addWidget(self.threshold_dropdown, 5, 1)

        self.measure_label = QLabel("Measure")
        layout.addWidget(self.measure_label, 6, 0)

        self.measure_dropdown = QComboBox()
        layout.addWidget(self.measure_dropdown, 6, 1)

        self.subject_label = QLabel("Subject")
        layout.addWidget(self.subject_label, 7, 0)

        self.subject_dropdown = QComboBox()
        self.subject_dropdown.addItem("GroupAverage")
        layout.addWidget(self.subject_dropdown, 7, 1)

        self.update_button = QPushButton("Update")
        # self.update_button.clicked.connect(self.gatv_update_timeseries)
        layout.addWidget(self.update_button, 8, 0)

        self.show()