#!/usr/bin/env python3
"""
Debug script to test the timeseries plotting functionality
"""

import sys
import os
import numpy as np
import scipy.io
import h5py
import matplotlib.pyplot as plt

def test_data_loading(mat_file_path):
    """Test loading and understanding the data structure"""
    print(f"Loading file: {mat_file_path}")
    
    try:
        # First try to load with scipy.io.loadmat (for older MATLAB formats)
        mat_contents = scipy.io.loadmat(mat_file_path, squeeze_me=True, struct_as_record=False)
        print("Loaded with scipy.io.loadmat")
        
        # Try to load the expected data structures
        dnet_data_data_mat_global = mat_contents.get('dnet_data_data_mat_global', None)
        dnet_data_threshold_list = mat_contents.get('dnet_data_threshold_list', None)
        dnet_data_files = mat_contents.get('dnet_data_files', None)
        dnet_data_measures_glob = mat_contents.get('dnet_data_measures_glob', None)
        
    except NotImplementedError:
        # If that fails, try h5py for MATLAB v7.3 files
        print("Detected MATLAB v7.3 format, using HDF5 reader...")
        try:
            with h5py.File(mat_file_path, 'r') as f:
                print(f"HDF5 keys: {list(f.keys())}")
                # Navigate the HDF5 structure to find the data
                dnet_data_data_mat_global = np.array(f['dnet_data_data_mat_global']) if 'dnet_data_data_mat_global' in f else None
                dnet_data_threshold_list = np.array(f['dnet_data_threshold_list']) if 'dnet_data_threshold_list' in f else None
                dnet_data_files = np.array(f['dnet_data_files']) if 'dnet_data_files' in f else None
                dnet_data_measures_glob = np.array(f['dnet_data_measures_glob']) if 'dnet_data_measures_glob' in f else None
                
        except Exception as hdf5_error:
            print(f"Failed to load HDF5 file: {hdf5_error}")
            return
    except Exception as general_error:
        print(f"Failed to load file: {general_error}")
        return
    
    if dnet_data_data_mat_global is None:
        print("No global measures data found")
        return
        
    print(f"Raw data shape: {dnet_data_data_mat_global.shape}")
    print(f"Raw data type: {dnet_data_data_mat_global.dtype}")
    print(f"Raw data min/max: {np.min(dnet_data_data_mat_global):.6f} / {np.max(dnet_data_data_mat_global):.6f}")
    print(f"Raw data mean/std: {np.mean(dnet_data_data_mat_global):.6f} / {np.std(dnet_data_data_mat_global):.6f}")
    
    # Process the data similar to MATLAB version
    temp_data_size = dnet_data_data_mat_global.shape
    print(f"Original data shape: {temp_data_size}")
    
    if len(temp_data_size) == 4:
        # Create expanded data array with means (MATLAB: temp_data_size(3)+1, temp_data_size(4)+1)
        expanded_shape = (temp_data_size[0], temp_data_size[1], temp_data_size[2] + 1, temp_data_size[3] + 1)
        expanded_data = np.zeros(expanded_shape)
        
        # Copy original data (MATLAB: app.gatv_data.data(:,:,2:end,2:end)=temp_data.dnet_data_data_mat_global;)
        expanded_data[:, :, 1:, 1:] = dnet_data_data_mat_global
        
        # Calculate means (MATLAB: app.gatv_data.data(:,:,1,2:end)=mean(temp_data.dnet_data_data_mat_global,3);)
        expanded_data[:, :, 0, 1:] = np.mean(dnet_data_data_mat_global, axis=2)
        # MATLAB: app.gatv_data.data(:,:,:,1)=mean(app.gatv_data.data(:,:,:,2:end),4);
        expanded_data[:, :, :, 0] = np.mean(expanded_data[:, :, :, 1:], axis=3)
        print(f"Expanded data shape: {expanded_data.shape}")
        
        # Test extracting a specific timeseries (equivalent to MATLAB indexing)
        measure_idx = 0  # First measure
        threshold_idx = 0  # Mean threshold (first index)
        subject_idx = 0   # Group average (first index)
        
        temp_data = expanded_data[:, measure_idx, threshold_idx, subject_idx]
        print(f"Extracted timeseries shape: {temp_data.shape}")
        print(f"Extracted timeseries min/max: {np.min(temp_data):.6f} / {np.max(temp_data):.6f}")
        print(f"Extracted timeseries mean/std: {np.mean(temp_data):.6f} / {np.std(temp_data):.6f}")
        
        # Create a simple plot
        plt.figure(figsize=(12, 6))
        x_range = np.arange(1, len(temp_data) + 1)  # MATLAB 1-based indexing
        plt.plot(x_range, temp_data, 'b-', linewidth=2, label='Network Property')
        plt.xlabel('Time (frames)')
        plt.ylabel('Network Measure')
        plt.title('Network Property Timeseries - Debug Plot')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        return expanded_data
    
    else:
        print(f"Unexpected data dimensions: {len(temp_data_size)}")
        return dnet_data_data_mat_global

if __name__ == "__main__":
    if len(sys.argv) > 1:
        mat_file_path = sys.argv[1]
    else:
        print("Usage: python debug_timeseries.py <path_to_mat_file>")
        print("Please provide the path to your network property MAT file")
        sys.exit(1)
    
    if not os.path.exists(mat_file_path):
        print(f"File not found: {mat_file_path}")
        sys.exit(1)
    
    test_data_loading(mat_file_path)
