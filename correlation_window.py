import sys
import os
import numpy as np
from scipy.io import loadmat, savemat
from scipy.stats import pearsonr
import pandas as pd
import h5py
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QMessageBox, QTableWidget, QTableWidgetItem,
                            QComboBox, QTextEdit, QSplitter, QScrollArea)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

class CorrWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data1 = None
        self.data2 = None
        self.file1_path = None
        self.file2_path = None
        self.setupUI()
        
    def setupUI(self):
        self.setWindowTitle('Correlation Comparison Tool')
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Title
        title_label = QLabel('Dataset Correlation Comparison Tool')
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        main_layout.addWidget(title_label)
        
        # File selection section
        file_section = QWidget()
        file_layout = QVBoxLayout(file_section)
        
        # File 1 selection
        file1_layout = QHBoxLayout()
        self.file1_label = QLabel('Dataset 1: No file selected')
        self.file1_button = QPushButton('Select Dataset 1 (.mat file)')
        self.file1_button.clicked.connect(self.select_file1)
        file1_layout.addWidget(self.file1_label)
        file1_layout.addWidget(self.file1_button)
        file_layout.addLayout(file1_layout)
        
        # Data path selection for file 1
        path1_layout = QHBoxLayout()
        path1_layout.addWidget(QLabel('Data path in file 1:'))
        self.path1_combo = QComboBox()
        self.path1_combo.setEditable(True)
        self.path1_combo.addItems(['subj_data.d_atlas', 'subj_data.d_corr', 'subj_data.func_data', 'subj_data.func_sample', 'd_atlas', 'd_corr', 'func_data', 'func_sample'])
        path1_layout.addWidget(self.path1_combo)
        file_layout.addLayout(path1_layout)
        
        # File 2 selection
        file2_layout = QHBoxLayout()
        self.file2_label = QLabel('Dataset 2: No file selected')
        self.file2_button = QPushButton('Select Dataset 2 (.mat file)')
        self.file2_button.clicked.connect(self.select_file2)
        file2_layout.addWidget(self.file2_label)
        file2_layout.addWidget(self.file2_button)
        file_layout.addLayout(file2_layout)
        
        # Data path selection for file 2
        path2_layout = QHBoxLayout()
        path2_layout.addWidget(QLabel('Data path in file 2:'))
        self.path2_combo = QComboBox()
        self.path2_combo.setEditable(True)
        self.path2_combo.addItems(['subj_data.d_atlas', 'subj_data.d_corr', 'subj_data.func_data', 'subj_data.func_sample', 'd_atlas', 'd_corr', 'func_data', 'func_sample'])
        path2_layout.addWidget(self.path2_combo)
        file_layout.addLayout(path2_layout)
        
        main_layout.addWidget(file_section)
        
        # Control buttons
        button_layout = QHBoxLayout()
        self.load_button = QPushButton('Load Data')
        self.load_button.clicked.connect(self.load_data)
        self.run_button = QPushButton('Calculate Correlations')
        self.run_button.clicked.connect(self.calculate_correlations)
        self.run_button.setEnabled(False)
        self.save_button = QPushButton('Save Results (.mat)')
        self.save_button.clicked.connect(self.save_results)
        self.save_button.setEnabled(False)
        
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.save_button)
        main_layout.addLayout(button_layout)
        
        # Results section
        results_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Data info
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(QLabel('Dataset Information:'))
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(200)
        left_layout.addWidget(self.info_text)
        
        # Right panel - Results table
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(QLabel('Correlation Results:'))
        
        # Scroll area for the table
        scroll_area = QScrollArea()
        self.results_table = QTableWidget()
        scroll_area.setWidget(self.results_table)
        scroll_area.setWidgetResizable(True)
        right_layout.addWidget(scroll_area)
        
        results_splitter.addWidget(left_panel)
        results_splitter.addWidget(right_panel)
        results_splitter.setSizes([400, 800])
        
        main_layout.addWidget(results_splitter)
        
        # Status info
        self.status_label = QLabel('Ready. Select two .mat files to compare (supports 2D, 3D, and 4D functional data).')
        main_layout.addWidget(self.status_label)
        
    def select_file1(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Select Dataset 1 (.mat file)', '', 'MATLAB Files (*.mat);;All Files (*)')
        if file_path:
            self.file1_path = file_path
            self.file1_label.setText(f'Dataset 1: {os.path.basename(file_path)}')
            self.status_label.setText('Dataset 1 selected. Select Dataset 2.')
            
    def select_file2(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Select Dataset 2 (.mat file)', '', 'MATLAB Files (*.mat);;All Files (*)')
        if file_path:
            self.file2_path = file_path
            self.file2_label.setText(f'Dataset 2: {os.path.basename(file_path)}')
            self.status_label.setText('Both datasets selected. Click "Load Data" to proceed.')
            
    def load_matlab_file(self, file_path):
        """Load MATLAB file, handling both v7.3 (HDF5) and older formats."""
        try:
            # First try with scipy.io.loadmat (for older MATLAB files)
            try:
                return loadmat(file_path)
            except NotImplementedError:
                with h5py.File(file_path, 'r') as f:
                    return self._h5py_to_dict(f)
        except Exception as e:
            raise Exception(f"Failed to load MATLAB file: {str(e)}")
    
    def _h5py_to_dict(self, h5_group):
        """Convert h5py group to dictionary recursively."""
        result = {}
        for key in h5_group.keys():
            if key.startswith('#'):  # Skip HDF5 metadata
                continue
            item = h5_group[key]
            if isinstance(item, h5py.Group):
                result[key] = self._h5py_to_dict(item)
            elif isinstance(item, h5py.Dataset):
                # Convert dataset to numpy array
                data = np.array(item)
                # Handle MATLAB's column-major order vs Python's row-major
                if data.ndim > 1:
                    data = data.T
                result[key] = data
        return result
            
    def get_nested_data(self, mat_data, path_string):
        """Extract data from nested structure using dot notation."""
        try:
            parts = path_string.split('.')
            current_data = mat_data
            
            for part in parts:
                if isinstance(current_data, dict):
                    if part in current_data:
                        current_data = current_data[part]
                    else:
                        return None
                elif hasattr(current_data, 'dtype') and hasattr(current_data.dtype, 'names') and current_data.dtype.names:
                    # Structured array (from scipy.io.loadmat)
                    if part in current_data.dtype.names:
                        current_data = current_data[part][0, 0]
                    else:
                        return None
                elif hasattr(current_data, '__getitem__'):
                    # Try direct indexing (for h5py-loaded data)
                    try:
                        current_data = current_data[part]
                    except (KeyError, TypeError):
                        return None
                else:
                    return None
                    
            return current_data
        except Exception as e:
            print(f"Error extracting data from path '{path_string}': {e}")
            return None
            
    def load_data(self):
        if not self.file1_path or not self.file2_path:
            QMessageBox.warning(self, 'Warning', 'Please select both dataset files first.')
            return
            
        try:
            # Load MATLAB files using the new method
            mat1 = self.load_matlab_file(self.file1_path)
            mat2 = self.load_matlab_file(self.file2_path)
            
            # Debug: Print available keys
            print(f"Dataset 1 keys: {list(mat1.keys())}")
            print(f"Dataset 2 keys: {list(mat2.keys())}")
            
            # Get data paths
            path1 = self.path1_combo.currentText()
            path2 = self.path2_combo.currentText()
            
            # Extract data using specified paths
            data1 = self.get_nested_data(mat1, path1)
            data2 = self.get_nested_data(mat2, path2)
            
            if data1 is None:
                # Try to provide helpful error message
                available_paths = self._get_available_paths(mat1)
                error_msg = f'Could not find data at path "{path1}" in Dataset 1.\n\nAvailable paths:\n' + '\n'.join(available_paths[:10])
                QMessageBox.critical(self, 'Error', error_msg)
                return
            if data2 is None:
                # Try to provide helpful error message
                available_paths = self._get_available_paths(mat2)
                error_msg = f'Could not find data at path "{path2}" in Dataset 2.\n\nAvailable paths:\n' + '\n'.join(available_paths[:10])
                QMessageBox.critical(self, 'Error', error_msg)
                return
                
            # Convert to numpy arrays
            self.data1 = np.array(data1, dtype=np.float64)
            self.data2 = np.array(data2, dtype=np.float64)
            
            # Validate data shapes - support 2D, 3D, and 4D data
            if self.data1.ndim not in [2, 3, 4] or self.data2.ndim not in [2, 3, 4]:
                QMessageBox.critical(self, 'Error', 'Both datasets must be 2D (time x regions), 3D (time x regions x subjects), or 4D arrays (time x regions x subjects x samples)')
                return
            
            # Handle dimension mismatch
            if self.data1.ndim != self.data2.ndim:
                QMessageBox.critical(self, 'Error', f'Dimension mismatch: Dataset 1 is {self.data1.ndim}D, Dataset 2 is {self.data2.ndim}D')
                return
                
            # For 2D data: shape should be (time, regions)
            # For 3D data: shape should be (time, regions, subjects)
            # For 4D data: shape should be (time, regions, subjects, samples)
            if self.data1.ndim == 2:
                if self.data1.shape[1] != self.data2.shape[1]:
                    QMessageBox.warning(self, 'Warning', 
                        f'Number of regions differ: Dataset 1 has {self.data1.shape[1]} regions, '
                        f'Dataset 2 has {self.data2.shape[1]} regions. Will use minimum.')
                    min_cols = min(self.data1.shape[1], self.data2.shape[1])
                    self.data1 = self.data1[:, :min_cols]
                    self.data2 = self.data2[:, :min_cols]
            elif self.data1.ndim == 3:  # 3D data
                # Check if regions dimension matches
                if self.data1.shape[1] != self.data2.shape[1]:
                    QMessageBox.warning(self, 'Warning', 
                        f'Number of regions differ: Dataset 1 has {self.data1.shape[1]} regions, '
                        f'Dataset 2 has {self.data2.shape[1]} regions. Will use minimum.')
                    min_regions = min(self.data1.shape[1], self.data2.shape[1])
                    self.data1 = self.data1[:, :min_regions, :]
                    self.data2 = self.data2[:, :min_regions, :]
                
                # Check if subjects/samples dimension matches
                if self.data1.shape[2] != self.data2.shape[2]:
                    QMessageBox.warning(self, 'Warning', 
                        f'Number of subjects/samples differ: Dataset 1 has {self.data1.shape[2]} subjects, '
                        f'Dataset 2 has {self.data2.shape[2]} subjects. Will use minimum.')
                    min_subjects = min(self.data1.shape[2], self.data2.shape[2])
                    self.data1 = self.data1[:, :, :min_subjects]
                    self.data2 = self.data2[:, :, :min_subjects]
            else:  # 4D data
                # Check if regions dimension matches
                if self.data1.shape[1] != self.data2.shape[1]:
                    QMessageBox.warning(self, 'Warning', 
                        f'Number of regions differ: Dataset 1 has {self.data1.shape[1]} regions, '
                        f'Dataset 2 has {self.data2.shape[1]} regions. Will use minimum.')
                    min_regions = min(self.data1.shape[1], self.data2.shape[1])
                    self.data1 = self.data1[:, :min_regions, :, :]
                    self.data2 = self.data2[:, :min_regions, :, :]
                
                # Check if subjects dimension matches
                if self.data1.shape[2] != self.data2.shape[2]:
                    QMessageBox.warning(self, 'Warning', 
                        f'Number of subjects differ: Dataset 1 has {self.data1.shape[2]} subjects, '
                        f'Dataset 2 has {self.data2.shape[2]} subjects. Will use minimum.')
                    min_subjects = min(self.data1.shape[2], self.data2.shape[2])
                    self.data1 = self.data1[:, :, :min_subjects, :]
                    self.data2 = self.data2[:, :, :min_subjects, :]
                
                # Check if samples dimension matches
                if self.data1.shape[3] != self.data2.shape[3]:
                    QMessageBox.warning(self, 'Warning', 
                        f'Number of samples differ: Dataset 1 has {self.data1.shape[3]} samples, '
                        f'Dataset 2 has {self.data2.shape[3]} samples. Will use minimum.')
                    min_samples = min(self.data1.shape[3], self.data2.shape[3])
                    self.data1 = self.data1[:, :, :, :min_samples]
                    self.data2 = self.data2[:, :, :, :min_samples]
            
            # Update info display
            if self.data1.ndim == 2:
                data_desc = "(time x regions)"
            elif self.data1.ndim == 3:
                data_desc = "(time x regions x subjects)"
            else:
                data_desc = "(time x regions x subjects x samples)"
                
            info_text = f"""Dataset 1 Information:
                        File: {os.path.basename(self.file1_path)}
                        Path: {path1}
                        Shape: {self.data1.shape} {data_desc}
                        Data type: {self.data1.dtype}
                        Min: {np.min(self.data1):.6f}, Max: {np.max(self.data1):.6f}
                        Mean: {np.mean(self.data1):.6f}, Std: {np.std(self.data1):.6f}

                        Dataset 2 Information:
                        File: {os.path.basename(self.file2_path)}
                        Path: {path2}
                        Shape: {self.data2.shape} {data_desc}
                        Data type: {self.data2.dtype}
                        Min: {np.min(self.data2):.6f}, Max: {np.max(self.data2):.6f}
                        Mean: {np.mean(self.data2):.6f}, Std: {np.std(self.data2):.6f}

                        Ready for correlation analysis."""
            
            self.info_text.setText(info_text)
            self.run_button.setEnabled(True)
            self.status_label.setText('Data loaded successfully. Click "Calculate Correlations" to proceed.')
            
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Error loading data: {str(e)}')
            
    def calculate_correlations(self):
        if self.data1 is None or self.data2 is None:
            QMessageBox.warning(self, 'Warning', 'Please load data first.')
            return
            
        try:
            if self.data1.ndim == 2:
                # 2D data: (time x regions)
                self._calculate_correlations_2d()
            elif self.data1.ndim == 3:
                # 3D data: (time x regions x subjects)
                self._calculate_correlations_3d()
            else:
                # 4D data: (time x regions x subjects x samples)
                self._calculate_correlations_4d()
                
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Error calculating correlations: {str(e)}')
    
    def _calculate_correlations_2d(self):
        """Calculate correlations for 2D data (time x regions)."""
        n_cols = self.data1.shape[1]
        
        # Initialize results arrays
        correlations = np.zeros(n_cols)
        p_values = np.zeros(n_cols)
        
        # Calculate column-wise correlations
        for i in range(n_cols):
            col1 = self.data1[:, i]
            col2 = self.data2[:, i]
            
            # Remove NaN values
            mask = ~(np.isnan(col1) | np.isnan(col2))
            if np.sum(mask) < 3:  # Need at least 3 points for correlation
                correlations[i] = np.nan
                p_values[i] = np.nan
            else:
                corr, p_val = pearsonr(col1[mask], col2[mask])
                correlations[i] = corr
                p_values[i] = p_val
        
        # Store results
        self.correlations = correlations
        self.p_values = p_values
        self.correlation_type = "Region-wise"
        
        # Display results and calculate summary
        self._display_and_summarize_results()
    
    def _calculate_correlations_3d(self):
        """Calculate correlations for 3D data (time x regions x subjects)."""
        n_regions = self.data1.shape[1]
        n_subjects = self.data1.shape[2]
        
        # Initialize results arrays
        correlations = np.zeros((n_regions, n_subjects))
        p_values = np.zeros((n_regions, n_subjects))
        
        # Calculate correlations for each region and subject
        for region in range(n_regions):
            for subject in range(n_subjects):
                data1_subj = self.data1[:, region, subject]
                data2_subj = self.data2[:, region, subject]
                
                # Remove NaN values
                mask = ~(np.isnan(data1_subj) | np.isnan(data2_subj))
                if np.sum(mask) < 3:  # Need at least 3 points for correlation
                    correlations[region, subject] = np.nan
                    p_values[region, subject] = np.nan
                else:
                    corr, p_val = pearsonr(data1_subj[mask], data2_subj[mask])
                    correlations[region, subject] = corr
                    p_values[region, subject] = p_val
        
        # Store results
        self.correlations = correlations
        self.p_values = p_values
        self.correlation_type = "Region x Subject"
        
        # Display results and calculate summary
        self._display_and_summarize_results()
    
    def _calculate_correlations_4d(self):
        """Calculate correlations for 4D data (time x regions x subjects x samples)."""
        n_regions = self.data1.shape[1]
        n_subjects = self.data1.shape[2]
        n_samples = self.data1.shape[3]
        
        # Initialize results arrays
        correlations = np.zeros((n_regions, n_subjects, n_samples))
        p_values = np.zeros((n_regions, n_subjects, n_samples))
        
        # Calculate correlations for each region, subject, and sample
        for region in range(n_regions):
            for subject in range(n_subjects):
                for sample in range(n_samples):
                    data1_sample = self.data1[:, region, subject, sample]
                    data2_sample = self.data2[:, region, subject, sample]
                    
                    # Remove NaN values
                    mask = ~(np.isnan(data1_sample) | np.isnan(data2_sample))
                    if np.sum(mask) < 3:  # Need at least 3 points for correlation
                        correlations[region, subject, sample] = np.nan
                        p_values[region, subject, sample] = np.nan
                    else:
                        corr, p_val = pearsonr(data1_sample[mask], data2_sample[mask])
                        correlations[region, subject, sample] = corr
                        p_values[region, subject, sample] = p_val
        
        # Store results
        self.correlations = correlations
        self.p_values = p_values
        self.correlation_type = "Region x Subject x Sample"
        
        # Display results and calculate summary
        self._display_and_summarize_results()
    
    def _display_and_summarize_results(self):
        """Display results in table and calculate summary statistics."""
        # Display results in table
        self.display_results()
        
        # Calculate summary statistics
        if self.correlations.ndim == 1:
            valid_corrs = self.correlations[~np.isnan(self.correlations)]
            n_comparisons = len(self.correlations)
        else:
            valid_corrs = self.correlations[~np.isnan(self.correlations)]
            n_comparisons = self.correlations.size
        
        if len(valid_corrs) > 0:
            if self.correlations.ndim == 1:
                summary = f"""
Correlation Summary ({self.correlation_type}):
Number of regions compared: {n_comparisons}
Valid correlations: {len(valid_corrs)}
Mean correlation: {np.mean(valid_corrs):.6f}
Std correlation: {np.std(valid_corrs):.6f}
Min correlation: {np.min(valid_corrs):.6f}
Max correlation: {np.max(valid_corrs):.6f}
Correlations > 0.9: {np.sum(valid_corrs > 0.9)}
Correlations > 0.95: {np.sum(valid_corrs > 0.95)}
Correlations > 0.99: {np.sum(valid_corrs > 0.99)}"""
            elif self.correlations.ndim == 2:
                summary = f"""
Correlation Summary ({self.correlation_type}):
Total comparisons: {n_comparisons} ({self.correlations.shape[0]} regions × {self.correlations.shape[1]} subjects)
Valid correlations: {len(valid_corrs)}
Mean correlation: {np.mean(valid_corrs):.6f}
Std correlation: {np.std(valid_corrs):.6f}
Min correlation: {np.min(valid_corrs):.6f}
Max correlation: {np.max(valid_corrs):.6f}
Correlations > 0.9: {np.sum(valid_corrs > 0.9)}
Correlations > 0.95: {np.sum(valid_corrs > 0.95)}
Correlations > 0.99: {np.sum(valid_corrs > 0.99)}"""
            else:  # 3D results (4D input data)
                summary = f"""
Correlation Summary ({self.correlation_type}):
Total comparisons: {n_comparisons} ({self.correlations.shape[0]} regions × {self.correlations.shape[1]} subjects × {self.correlations.shape[2]} samples)
Valid correlations: {len(valid_corrs)}
Mean correlation: {np.mean(valid_corrs):.6f}
Std correlation: {np.std(valid_corrs):.6f}
Min correlation: {np.min(valid_corrs):.6f}
Max correlation: {np.max(valid_corrs):.6f}
Correlations > 0.9: {np.sum(valid_corrs > 0.9)}
Correlations > 0.95: {np.sum(valid_corrs > 0.95)}
Correlations > 0.99: {np.sum(valid_corrs > 0.99)}"""
            
            current_text = self.info_text.toPlainText()
            self.info_text.setText(current_text + summary)
        
        self.save_button.setEnabled(True)
        self.status_label.setText('Correlations calculated successfully. Results shown in table.')
            
    def display_results(self):
        if self.correlations.ndim == 1:
            self._display_results_2d()
        elif self.correlations.ndim == 2:
            self._display_results_3d()
        else:
            self._display_results_4d()
    
    def _display_results_2d(self):
        """Display results for 2D correlation analysis."""
        n_cols = len(self.correlations)
        
        # Setup table
        self.results_table.setRowCount(n_cols)
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(['Region', 'Correlation', 'P-value', 'Significance'])
        
        # Fill table
        for i in range(n_cols):
            # Region number
            self.results_table.setItem(i, 0, QTableWidgetItem(f'Region {i+1}'))
            
            # Correlation
            if np.isnan(self.correlations[i]):
                corr_text = 'NaN'
            else:
                corr_text = f'{self.correlations[i]:.6f}'
            self.results_table.setItem(i, 1, QTableWidgetItem(corr_text))
            
            # P-value
            if np.isnan(self.p_values[i]):
                p_text = 'NaN'
            else:
                p_text = f'{self.p_values[i]:.6e}'
            self.results_table.setItem(i, 2, QTableWidgetItem(p_text))
            
            # Significance
            if np.isnan(self.p_values[i]) or np.isnan(self.correlations[i]):
                sig_text = 'N/A'
            elif self.p_values[i] < 0.001:
                sig_text = '***'
            elif self.p_values[i] < 0.01:
                sig_text = '**'
            elif self.p_values[i] < 0.05:
                sig_text = '*'
            else:
                sig_text = 'ns'
            self.results_table.setItem(i, 3, QTableWidgetItem(sig_text))
        
        # Auto-resize columns
        self.results_table.resizeColumnsToContents()
    
    def _display_results_3d(self):
        """Display results for 3D correlation analysis."""
        n_regions, n_subjects = self.correlations.shape
        total_rows = n_regions * n_subjects
        
        # Setup table
        self.results_table.setRowCount(total_rows)
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels(['Region', 'Subject', 'Correlation', 'P-value', 'Significance'])
        
        # Fill table
        row = 0
        for region in range(n_regions):
            for subject in range(n_subjects):
                # Region number
                self.results_table.setItem(row, 0, QTableWidgetItem(f'Region {region+1}'))
                
                # Subject number
                self.results_table.setItem(row, 1, QTableWidgetItem(f'Subject {subject+1}'))
                
                # Correlation
                corr_val = self.correlations[region, subject]
                if np.isnan(corr_val):
                    corr_text = 'NaN'
                else:
                    corr_text = f'{corr_val:.6f}'
                self.results_table.setItem(row, 2, QTableWidgetItem(corr_text))
                
                # P-value
                p_val = self.p_values[region, subject]
                if np.isnan(p_val):
                    p_text = 'NaN'
                else:
                    p_text = f'{p_val:.6e}'
                self.results_table.setItem(row, 3, QTableWidgetItem(p_text))
                
                # Significance
                if np.isnan(p_val) or np.isnan(corr_val):
                    sig_text = 'N/A'
                elif p_val < 0.001:
                    sig_text = '***'
                elif p_val < 0.01:
                    sig_text = '**'
                elif p_val < 0.05:
                    sig_text = '*'
                else:
                    sig_text = 'ns'
                self.results_table.setItem(row, 4, QTableWidgetItem(sig_text))
                
                row += 1
        
        # Auto-resize columns
        self.results_table.resizeColumnsToContents()
        
    def _display_results_4d(self):
        """Display results for 4D correlation analysis."""
        n_regions, n_subjects, n_samples = self.correlations.shape
        total_rows = n_regions * n_subjects * n_samples
        
        # Setup table
        self.results_table.setRowCount(total_rows)
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels(['Region', 'Subject', 'Sample', 'Correlation', 'P-value', 'Significance'])
        
        # Fill table
        row = 0
        for region in range(n_regions):
            for subject in range(n_subjects):
                for sample in range(n_samples):
                    # Region number
                    self.results_table.setItem(row, 0, QTableWidgetItem(f'Region {region+1}'))
                    
                    # Subject number
                    self.results_table.setItem(row, 1, QTableWidgetItem(f'Subject {subject+1}'))
                    
                    # Sample number
                    self.results_table.setItem(row, 2, QTableWidgetItem(f'Sample {sample+1}'))
                    
                    # Correlation
                    corr_val = self.correlations[region, subject, sample]
                    if np.isnan(corr_val):
                        corr_text = 'NaN'
                    else:
                        corr_text = f'{corr_val:.6f}'
                    self.results_table.setItem(row, 3, QTableWidgetItem(corr_text))
                    
                    # P-value
                    p_val = self.p_values[region, subject, sample]
                    if np.isnan(p_val):
                        p_text = 'NaN'
                    else:
                        p_text = f'{p_val:.6e}'
                    self.results_table.setItem(row, 4, QTableWidgetItem(p_text))
                    
                    # Significance
                    if np.isnan(p_val) or np.isnan(corr_val):
                        sig_text = 'N/A'
                    elif p_val < 0.001:
                        sig_text = '***'
                    elif p_val < 0.01:
                        sig_text = '**'
                    elif p_val < 0.05:
                        sig_text = '*'
                    else:
                        sig_text = 'ns'
                    self.results_table.setItem(row, 5, QTableWidgetItem(sig_text))
                    
                    row += 1
        
        # Auto-resize columns
        self.results_table.resizeColumnsToContents()
        
    def save_results(self):
        if not hasattr(self, 'correlations'):
            QMessageBox.warning(self, 'Warning', 'No results to save. Calculate correlations first.')
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, 'Save Correlation Results', 'correlation_results.mat', 'MATLAB Files (*.mat)')
        
        if file_path:
            try:
                # Prepare results for saving
                results = {
                    'correlations': self.correlations,
                    'p_values': self.p_values,
                    'correlation_type': self.correlation_type,
                    'data_dimensions': f"{self.data1.ndim}D",
                    'dataset1_info': {
                        'file': os.path.basename(self.file1_path),
                        'path': self.path1_combo.currentText(),
                        'shape': self.data1.shape
                    },
                    'dataset2_info': {
                        'file': os.path.basename(self.file2_path),
                        'path': self.path2_combo.currentText(),
                        'shape': self.data2.shape
                    }
                }
                
                # Add appropriate indexing based on data dimensions
                if self.correlations.ndim == 1:
                    results['region_numbers'] = np.arange(1, len(self.correlations) + 1)
                    results['summary_stats'] = {
                        'n_regions': len(self.correlations),
                        'n_valid': np.sum(~np.isnan(self.correlations)),
                        'mean_correlation': np.nanmean(self.correlations),
                        'std_correlation': np.nanstd(self.correlations),
                        'min_correlation': np.nanmin(self.correlations),
                        'max_correlation': np.nanmax(self.correlations)
                    }
                elif self.correlations.ndim == 2:
                    # For 3D results, create region and subject indices
                    n_regions, n_subjects = self.correlations.shape
                    region_idx, subject_idx = np.meshgrid(
                        np.arange(1, n_regions + 1), 
                        np.arange(1, n_subjects + 1), 
                        indexing='ij'
                    )
                    results['region_indices'] = region_idx
                    results['subject_indices'] = subject_idx
                    results['summary_stats'] = {
                        'n_regions': n_regions,
                        'n_subjects': n_subjects,
                        'total_comparisons': self.correlations.size,
                        'n_valid': np.sum(~np.isnan(self.correlations)),
                        'mean_correlation': np.nanmean(self.correlations),
                        'std_correlation': np.nanstd(self.correlations),
                        'min_correlation': np.nanmin(self.correlations),
                        'max_correlation': np.nanmax(self.correlations)
                    }
                else:
                    # For 4D results, create region, subject, and sample indices
                    n_regions, n_subjects, n_samples = self.correlations.shape
                    region_idx = np.zeros(self.correlations.shape, dtype=int)
                    subject_idx = np.zeros(self.correlations.shape, dtype=int)
                    sample_idx = np.zeros(self.correlations.shape, dtype=int)
                    
                    for r in range(n_regions):
                        for s in range(n_subjects):
                            for sa in range(n_samples):
                                region_idx[r, s, sa] = r + 1
                                subject_idx[r, s, sa] = s + 1
                                sample_idx[r, s, sa] = sa + 1
                    
                    results['region_indices'] = region_idx
                    results['subject_indices'] = subject_idx
                    results['sample_indices'] = sample_idx
                    results['summary_stats'] = {
                        'n_regions': n_regions,
                        'n_subjects': n_subjects,
                        'n_samples': n_samples,
                        'total_comparisons': self.correlations.size,
                        'n_valid': np.sum(~np.isnan(self.correlations)),
                        'mean_correlation': np.nanmean(self.correlations),
                        'std_correlation': np.nanstd(self.correlations),
                        'min_correlation': np.nanmin(self.correlations),
                        'max_correlation': np.nanmax(self.correlations)
                    }
                
                # Save to MATLAB format
                savemat(file_path, {'correlation_results': results})
                
                QMessageBox.information(self, 'Success', f'Results saved to:\n{file_path}')
                self.status_label.setText(f'Results saved to {os.path.basename(file_path)}')
                
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Error saving results: {str(e)}')
    
    def _get_available_paths(self, mat_data, prefix='', max_depth=3):
        """Get all available data paths in a MATLAB structure."""
        paths = []
        
        if max_depth <= 0:
            return paths
            
        if isinstance(mat_data, dict):
            for key, value in mat_data.items():
                if key.startswith('__'):  # Skip MATLAB metadata
                    continue
                current_path = f"{prefix}.{key}" if prefix else key
                paths.append(current_path)
                
                # Recursively explore nested structures
                if isinstance(value, (dict, np.ndarray)) and hasattr(value, 'dtype'):
                    if hasattr(value, 'dtype') and hasattr(value.dtype, 'names') and value.dtype.names:
                        # Structured array
                        nested_paths = self._get_available_paths(value, current_path, max_depth-1)
                        paths.extend(nested_paths)
                    elif isinstance(value, dict):
                        nested_paths = self._get_available_paths(value, current_path, max_depth-1)
                        paths.extend(nested_paths)
        elif hasattr(mat_data, 'dtype') and hasattr(mat_data.dtype, 'names') and mat_data.dtype.names:
            # Structured array from scipy.io.loadmat
            for field_name in mat_data.dtype.names:
                current_path = f"{prefix}.{field_name}" if prefix else field_name
                paths.append(current_path)
                try:
                    field_data = mat_data[field_name][0, 0]
                    nested_paths = self._get_available_paths(field_data, current_path, max_depth-1)
                    paths.extend(nested_paths)
                except (IndexError, AttributeError):
                    pass
        
        return paths


