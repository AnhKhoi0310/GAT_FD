# view_window.py
from PyQt6.QtWidgets import ( QWidget, QLabel, QPushButton, QSlider, QGridLayout, QFileDialog, QMessageBox, QComboBox, QVBoxLayout)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeyEvent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
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

    def gatv_load_measure(self):
        """Load network property file - equivalent to MATLAB gatv_load_measure function"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Conditions", "", "MAT files (*.mat)")
        if not file_path:
            return
        
        try:
            # First try to load with scipy.io.loadmat (for older MATLAB formats)
            mat_contents = scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)
            
            # Try to load the expected data structures
            dnet_data_data_mat_global = mat_contents.get('dnet_data_data_mat_global', None)
            dnet_data_threshold_list = mat_contents.get('dnet_data_threshold_list', None)
            dnet_data_files = mat_contents.get('dnet_data_files', None)
            dnet_data_measures_glob = mat_contents.get('dnet_data_measures_glob', None)
            
        except NotImplementedError:
            # If that fails, try h5py for MATLAB v7.3 files
            print("Detected MATLAB v7.3 format, using HDF5 reader...")
            try:
                with h5py.File(file_path, 'r') as f:
                    # Navigate the HDF5 structure to find the data
                    # Note: h5py reads MATLAB arrays with transposed dimensions, so we need to transpose back
                    if 'dnet_data_data_mat_global' in f:
                        dnet_data_data_mat_global = np.array(f['dnet_data_data_mat_global']).T  # Transpose to match MATLAB ordering
                    else:
                        dnet_data_data_mat_global = None
                    
                    if 'dnet_data_threshold_list' in f:
                        dnet_data_threshold_list = np.array(f['dnet_data_threshold_list']).T
                    else:
                        dnet_data_threshold_list = None
                        
                    if 'dnet_data_files' in f:
                        dnet_data_files = np.array(f['dnet_data_files']).T
                    else:
                        dnet_data_files = None
                        
                    if 'dnet_data_measures_glob' in f:
                        dnet_data_measures_glob = np.array(f['dnet_data_measures_glob']).T
                    else:
                        dnet_data_measures_glob = None
                    
            except Exception as hdf5_error:
                QMessageBox.critical(self, "Error", f"Failed to load HDF5 file: {hdf5_error}")
                return
        except Exception as general_error:
            QMessageBox.critical(self, "Error", f"Failed to load file: {general_error}")
            return
        
        if dnet_data_data_mat_global is None:
            QMessageBox.critical(self, "Error", "No global measures to display")
            return
            
        try:
            # Process the data similar to MATLAB version
            temp_data_size = dnet_data_data_mat_global.shape
            print(f"Original data shape: {temp_data_size}")
            
            if len(temp_data_size) == 4:
                # Create expanded data array with means (MATLAB: temp_data_size(3)+1, temp_data_size(4)+1)
                expanded_shape = (temp_data_size[0], temp_data_size[1], temp_data_size[2] + 1, temp_data_size[3] + 1)
                self.gatv_data['data'] = np.zeros(expanded_shape)
                
                # Copy original data (MATLAB: app.gatv_data.data(:,:,2:end,2:end)=temp_data.dnet_data_data_mat_global;)
                self.gatv_data['data'][:, :, 1:, 1:] = dnet_data_data_mat_global
                
                # Calculate means (MATLAB: app.gatv_data.data(:,:,1,2:end)=mean(temp_data.dnet_data_data_mat_global,3);)
                self.gatv_data['data'][:, :, 0, 1:] = np.mean(dnet_data_data_mat_global, axis=2)
                # MATLAB: app.gatv_data.data(:,:,:,1)=mean(app.gatv_data.data(:,:,:,2:end),4);
                self.gatv_data['data'][:, :, :, 0] = np.mean(self.gatv_data['data'][:, :, :, 1:], axis=3)
                print(f"Expanded data shape: {self.gatv_data['data'].shape}")
            else:
                # For 3D or other dimensioned data, we need to expand appropriately
                if len(temp_data_size) == 3:
                    # If 3D, assume it's (timepoints, measures, thresholds) and add subjects dimension
                    expanded_shape = (temp_data_size[0], temp_data_size[1], temp_data_size[2] + 1, 2)
                    self.gatv_data['data'] = np.zeros(expanded_shape)
                    self.gatv_data['data'][:, :, 1:, 1] = dnet_data_data_mat_global
                    self.gatv_data['data'][:, :, 0, 1] = np.mean(dnet_data_data_mat_global, axis=2)
                    self.gatv_data['data'][:, :, :, 0] = np.mean(self.gatv_data['data'][:, :, :, 1:], axis=3)
                else:
                    # For other cases, try to reshape to 4D
                    self.gatv_data['data'] = dnet_data_data_mat_global.reshape(temp_data_size[0], temp_data_size[1], 1, 1)
                print(f"Processed data shape: {self.gatv_data['data'].shape}")
            
            # Update threshold dropdown
            self.threshold_dropdown.clear()
            self.threshold_dropdown.addItem("Mean")
            if dnet_data_threshold_list is not None:
                threshold_items = [str(x) for x in dnet_data_threshold_list.flatten()]
                self.threshold_dropdown.addItems(threshold_items)
            
            # Update subject dropdown  
            self.subject_dropdown.clear()
            self.subject_dropdown.addItem("GroupAverage")
            if dnet_data_files is not None:
                if isinstance(dnet_data_files, np.ndarray):
                    file_items = [str(x) for x in dnet_data_files.flatten()]
                else:
                    file_items = [str(dnet_data_files)]
                self.subject_dropdown.addItems(file_items)
                
            # Update measure dropdown
            self.measure_dropdown.clear()
            if dnet_data_measures_glob is not None:
                if isinstance(dnet_data_measures_glob, np.ndarray):
                    measure_items = [str(x) for x in dnet_data_measures_glob.flatten()]
                else:
                    measure_items = [str(dnet_data_measures_glob)]
                self.measure_dropdown.addItems(measure_items)
                
            print("Successfully loaded network property file")
            
        except Exception as processing_error:
            QMessageBox.critical(self, "Error", f"Failed to process network property file: {processing_error}")

    def gatv_update_timeseries(self):
        """Update timeseries plot - equivalent to MATLAB gatv_update_timeseries function"""
        # Check if all dropdowns have valid selections
        if (self.measure_dropdown.currentIndex() < 0 or 
            self.threshold_dropdown.currentIndex() < 0 or 
            self.subject_dropdown.currentIndex() < 0):
            QMessageBox.critical(self, "Error", "Please make selection")
            return
            
        if 'data' not in self.gatv_data:
            QMessageBox.critical(self, "Error", "Please load network property file first")
            return
        
        try:
            # Get indices for data selection - MATLAB uses 1-based indexing via ItemsData
            # MATLAB: MeasureDropDown.Value, ThresholdDropDown.Value, SubjectDropDown.Value
            # MATLAB ItemsData = 1:length(Items), so Values are 1, 2, 3, ...
            # Our Python data array is set up so: index 0 = "Mean", index 1+ = actual data
            # Python currentIndex() returns 0, 1, 2, ... which directly maps to our array indices
            measure_idx = self.measure_dropdown.currentIndex()
            threshold_idx = self.threshold_dropdown.currentIndex() 
            subject_idx = self.subject_dropdown.currentIndex()
            
            print(f"Selected indices (0-based for Python array): measure={measure_idx}, threshold={threshold_idx}, subject={subject_idx}")
            print(f"Data shape: {self.gatv_data['data'].shape}")
            
            # Extract timeseries data (MATLAB: temp_data=app.gatv_data.data(:,MeasureDropDown.Value,ThresholdDropDown.Value,SubjectDropDown.Value))
            temp_data = self.gatv_data['data'][:, measure_idx, threshold_idx, subject_idx]
            print(f"temp_data shape: {temp_data.shape}, min/max: {np.min(temp_data):.6f} / {np.max(temp_data):.6f}")
            
            # Clear the axis and any twin axes
            self.timeseries_ax.clear()
            
            # Remove any existing twin axes to start fresh
            for ax in self.timeseries_canvas.figure.axes[1:]:
                ax.remove()
            
            # MATLAB: yyaxis(app.UIAxes_Timeseries,'left');
            # This sets the left y-axis as active in MATLAB
            
            if self.gatv_setting.get('iscondition', 0):
                print("Plotting with condition overlay")
                # Plot with condition overlay (MATLAB logic)
                data_length_diff = self.gatv_data.get('data_length_diff', 0)
                data_frame_length = self.gatv_data.get('data_frame_length', len(temp_data))
                data_length = self.gatv_data.get('data_length', len(temp_data))
                
                print(f"Condition plot: data_length_diff={data_length_diff}, data_frame_length={data_frame_length}, data_length={data_length}")
                
                # MATLAB: plot(app.UIAxes_Timeseries,app.gatv_data.data_length_diff:app.gatv_data.data_length_diff-1+app.gatv_data.data_frame_length,temp_data,'b','LineWidth',2);
                # MATLAB range: data_length_diff:(data_length_diff-1+data_frame_length) = data_length_diff:(data_length_diff+data_frame_length-1)
                # Python equivalent: range(data_length_diff, data_length_diff + data_frame_length) but using MATLAB 1-based indexing convention
                x_range = np.arange(data_length_diff, data_length_diff + data_frame_length)  # MATLAB: data_length_diff:data_length_diff-1+data_frame_length
                print(f"Left plot x_range: {x_range[:5]}...{x_range[-5:]} (length={len(x_range)})")
                
                self.timeseries_ax.plot(x_range, temp_data, 'b', linewidth=2, label='Network Property')
                self.timeseries_ax.relim()  # Recalculate limits
                self.timeseries_ax.autoscale_view()  # Auto-scale the view (equivalent to MATLAB ylim('auto'))
                self.timeseries_ax.set_ylabel('Network Measure', color='b')
                self.timeseries_ax.tick_params(axis='y', labelcolor='b')
                
                # MATLAB: yyaxis(app.UIAxes_Timeseries,'right');
                # MATLAB: plot(app.UIAxes_Timeseries,1:app.gatv_data.data_length,app.gatv_setting.condition.dfnc_reponse,'--r','LineWidth',2);
                timeseries_ax_right = self.timeseries_ax.twinx()
                
                if 'condition' in self.gatv_setting and hasattr(self.gatv_setting['condition'], 'dfnc_reponse'):
                    dfnc_reponse = np.array(self.gatv_setting['condition'].dfnc_reponse).flatten()
                    x_range_condition = np.arange(1, data_length + 1)  # MATLAB: 1:app.gatv_data.data_length
                    print(f"Right plot x_range: {x_range_condition[:5]}...{x_range_condition[-5:]} (length={len(x_range_condition)})")
                    print(f"dfnc_reponse shape: {dfnc_reponse.shape}, min/max: {np.min(dfnc_reponse):.6f} / {np.max(dfnc_reponse):.6f}")
                    
                    timeseries_ax_right.plot(x_range_condition, dfnc_reponse, '--r', linewidth=2, label='Task Response')
                    timeseries_ax_right.relim()  # Recalculate limits
                    timeseries_ax_right.autoscale_view()  # Auto-scale the view (equivalent to MATLAB ylim('auto'))
                    timeseries_ax_right.set_ylabel('Task Response', color='r')
                    timeseries_ax_right.tick_params(axis='y', labelcolor='r')
                else:
                    print(f"No condition.dfnc_reponse found. Available condition keys: {list(self.gatv_setting.get('condition', {}).__dict__.keys()) if 'condition' in self.gatv_setting else 'No condition'}")
                    # Still create the right axis but don't plot anything
                    timeseries_ax_right.set_ylabel('No Task Response', color='r')
                    timeseries_ax_right.tick_params(axis='y', labelcolor='r')
                    
            else:
                print("Plotting without condition overlay")
                # MATLAB: plot(app.UIAxes_Timeseries,1:length(temp_data),temp_data,'b','LineWidth',2);
                x_range = np.arange(1, len(temp_data) + 1)  # MATLAB 1-based indexing: 1:length(temp_data)
                print(f"Simple plot x_range: {x_range[:5]}...{x_range[-5:]} (length={len(x_range)})")
                
                self.timeseries_ax.plot(x_range, temp_data, 'b', linewidth=2, label='Network Property')
                # MATLAB: ylim(app.UIAxes_Timeseries,'auto') - use relim/autoscale_view for matplotlib
                self.timeseries_ax.relim()
                self.timeseries_ax.autoscale_view()
                self.timeseries_ax.set_ylabel('Network Measure', color='b')
                self.timeseries_ax.tick_params(axis='y', labelcolor='b')
            
            # Set title and labels
            self.timeseries_ax.set_title('Network Property Timeseries')
            self.timeseries_ax.set_xlabel('Time (frames)')
            self.timeseries_ax.grid(True, alpha=0.3)
            
            # Refresh the canvas
            self.timeseries_canvas.draw()
            
            print("Timeseries plot updated successfully")
            
        except Exception as e:
            print(f"Error in gatv_update_timeseries: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to update timeseries: {e}")

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

        # UIAxes_Timeseries with Navigation Toolbar
        timeseries_widget = QWidget()
        timeseries_layout = QVBoxLayout(timeseries_widget)
        
        self.timeseries_canvas = FigureCanvas(Figure())
        self.timeseries_ax = self.timeseries_canvas.figure.add_subplot(111)
        self.timeseries_ax.set_title("Network Properties Timeseries")
        
        # Add navigation toolbar for zoom/pan functionality
        self.timeseries_toolbar = NavigationToolbar(self.timeseries_canvas, timeseries_widget)
        
        timeseries_layout.addWidget(self.timeseries_toolbar)
        timeseries_layout.addWidget(self.timeseries_canvas)
        
        layout.addWidget(timeseries_widget, 3, 0, 1, 3)

        # LoadTemporalMaskButton
        self.load_temporal_button = QPushButton("Load Temporal Mask")
        self.load_temporal_button.clicked.connect(self.gatv_load_design)
        layout.addWidget(self.load_temporal_button, 2, 0)

        # LoadNetworkPropertyFileButton
        self.load_property_button = QPushButton("Load Network Property File")
        self.load_property_button.clicked.connect(self.gatv_load_measure)
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
        self.update_button.clicked.connect(self.gatv_update_timeseries)
        layout.addWidget(self.update_button, 8, 0)

        self.show()