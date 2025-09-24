"""
Python implementation of MATLAB's imwarp function for 3D medical images.
Focus on replicating the exact behavior of images.internal.builtins.warp3d
"""

import numpy as np
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
import warnings

class ImRef3D:
    """Python equivalent of MATLAB's imref3d spatial referencing object"""
    
    def __init__(self, image_size, world_limits=None, pixel_extents=None):
        self.ImageSize = np.array(image_size)
        
        if world_limits is None:
            # Default: pixel centers at 1,2,3... (MATLAB 1-based indexing)
            self.XWorldLimits = np.array([0.5, image_size[1] + 0.5])
            self.YWorldLimits = np.array([0.5, image_size[0] + 0.5]) 
            self.ZWorldLimits = np.array([0.5, image_size[2] + 0.5])
        else:
            self.XWorldLimits = world_limits[0]
            self.YWorldLimits = world_limits[1]
            self.ZWorldLimits = world_limits[2]
            
        # Calculate pixel extents
        self.PixelExtentInWorldX = (self.XWorldLimits[1] - self.XWorldLimits[0]) / image_size[1]
        self.PixelExtentInWorldY = (self.YWorldLimits[1] - self.YWorldLimits[0]) / image_size[0]
        self.PixelExtentInWorldZ = (self.ZWorldLimits[1] - self.ZWorldLimits[0]) / image_size[2]
        
        # Create transformation matrices (MATLAB style)
        self._create_transform_matrices()
    
    def _create_transform_matrices(self):
        """Create intrinsic-to-world and world-to-intrinsic transformation matrices"""
        # MATLAB uses 1-based indexing, so intrinsic coordinates start at 1
        # Transform from intrinsic (1-based) to world coordinates
        self.TransformIntrinsicToWorld = np.array([
            [self.PixelExtentInWorldX, 0, 0, self.XWorldLimits[0] - self.PixelExtentInWorldX],
            [0, self.PixelExtentInWorldY, 0, self.YWorldLimits[0] - self.PixelExtentInWorldY],
            [0, 0, self.PixelExtentInWorldZ, self.ZWorldLimits[0] - self.PixelExtentInWorldZ], 
            [0, 0, 0, 1]
        ])
        
        # Inverse transformation (world to intrinsic)
        self.TransformWorldToIntrinsic = np.linalg.inv(self.TransformIntrinsicToWorld)
    
    def intrinsic_to_world_algo(self, x_intrinsic, y_intrinsic, z_intrinsic):
        """Convert intrinsic coordinates to world coordinates (MATLAB equivalent)"""
        # Stack coordinates for matrix multiplication
        coords = np.stack([x_intrinsic.flatten(), y_intrinsic.flatten(), 
                          z_intrinsic.flatten(), np.ones_like(x_intrinsic.flatten())])
        
        # Apply transformation
        world_coords = self.TransformIntrinsicToWorld @ coords
        
        # Reshape back to original shape
        x_world = world_coords[0].reshape(x_intrinsic.shape)
        y_world = world_coords[1].reshape(y_intrinsic.shape)
        z_world = world_coords[2].reshape(z_intrinsic.shape)
        
        return x_world, y_world, z_world
    
    def world_to_intrinsic_algo(self, x_world, y_world, z_world):
        """Convert world coordinates to intrinsic coordinates (MATLAB equivalent)"""
        # Stack coordinates for matrix multiplication
        coords = np.stack([x_world.flatten(), y_world.flatten(), 
                          z_world.flatten(), np.ones_like(x_world.flatten())])
        
        # Apply transformation
        intrinsic_coords = self.TransformWorldToIntrinsic @ coords
        
        # Reshape back to original shape
        x_intrinsic = intrinsic_coords[0].reshape(x_world.shape)
        y_intrinsic = intrinsic_coords[1].reshape(y_world.shape)
        z_intrinsic = intrinsic_coords[2].reshape(z_world.shape)
        
        return x_intrinsic, y_intrinsic, z_intrinsic


def warp3d(input_image, tform_matrix, output_size, fill_values=0, method='linear'):
    """
    Python implementation of MATLAB's images.internal.builtins.warp3d
    
    Parameters:
    -----------
    input_image : ndarray
        3D input image
    tform_matrix : ndarray
        4x4 transformation matrix (MATLAB tComp format)
    output_size : tuple
        (height, width, depth) of output image
    fill_values : float
        Value to use for pixels outside input image bounds
    method : str
        Interpolation method ('linear', 'nearest', 'cubic')
        
    Returns:
    --------
    output_image : ndarray
        Transformed 3D image
    """
    
    print(f"warp3d: Input shape {input_image.shape}, Output size {output_size}")
    print(f"warp3d: Transform matrix shape {tform_matrix.shape}")
    print(f"warp3d: Method: {method}, Fill value: {fill_values}")
    
    # Ensure input is float32 to match MATLAB's single precision
    input_image = input_image.astype(np.float32)
    
    # Create output coordinate grids (MATLAB 1-based indexing style)
    # MATLAB meshgrid creates coordinates starting from 1
    y_out, x_out, z_out = np.meshgrid(
        np.arange(1, output_size[0] + 1, dtype=np.float32),  # 1 to height
        np.arange(1, output_size[1] + 1, dtype=np.float32),  # 1 to width  
        np.arange(1, output_size[2] + 1, dtype=np.float32),  # 1 to depth
        indexing='ij'
    )
    
    # Stack coordinates for transformation (homogeneous coordinates)
    coords_out = np.stack([
        x_out.flatten(),
        y_out.flatten(), 
        z_out.flatten(),
        np.ones_like(x_out.flatten())
    ])
    
    print(f"warp3d: Output coordinates shape: {coords_out.shape}")
    
    # Apply inverse transformation to map output coordinates to input coordinates
    # MATLAB uses tComp in post-multiply form, so we apply it directly
    coords_in = tform_matrix @ coords_out
    
    # Extract input coordinates (convert back to 0-based for Python indexing)
    x_in = coords_in[0].reshape(output_size) - 1  # Convert to 0-based
    y_in = coords_in[1].reshape(output_size) - 1  # Convert to 0-based
    z_in = coords_in[2].reshape(output_size) - 1  # Convert to 0-based
    
    print(f"warp3d: Input coordinate ranges:")
    print(f"  X: {np.min(x_in):.3f} to {np.max(x_in):.3f}")
    print(f"  Y: {np.min(y_in):.3f} to {np.max(y_in):.3f}")
    print(f"  Z: {np.min(z_in):.3f} to {np.max(z_in):.3f}")
    
    # Perform interpolation using scipy's map_coordinates (matches MATLAB's behavior)
    # Stack coordinates for map_coordinates
    coordinates = np.stack([y_in, x_in, z_in])  # Note: y,x,z order for map_coordinates
    
    # Map interpolation methods
    order_map = {
        'nearest': 0,
        'linear': 1, 
        'cubic': 3
    }
    order = order_map.get(method, 1)
    
    print(f"warp3d: Using scipy.ndimage.map_coordinates with order={order}")
    
    # Apply transformation with proper boundary handling
    output_image = ndimage.map_coordinates(
        input_image,
        coordinates,
        order=order,
        cval=fill_values,
        mode='constant',
        prefilter=True  # Important: matches MATLAB's behavior
    )
    
    # Reshape to output size
    output_image = output_image.reshape(output_size).astype(np.float32)
    
    print(f"warp3d: Output image shape: {output_image.shape}")
    print(f"warp3d: Output value range: {np.min(output_image):.6f} to {np.max(output_image):.6f}")
    
    return output_image


def remap_and_resample_invertible_3d(input_image, R_in, tform, R_out, method='linear', 
                                   fill_values=0, smooth_edges=False):
    """
    Python implementation of MATLAB's remapAndResampleInvertible3d
    
    This is the main function that gets called for 3D affine transformations
    """
    
    print("=== remap_and_resample_invertible_3d ===")
    print(f"Input image shape: {input_image.shape}")
    print(f"Output image size: {R_out.ImageSize}")
    print(f"Method: {method}")
    
    # Get transformation matrices from spatial referencing objects
    t_intrinsic_to_world_output = R_out.TransformIntrinsicToWorld
    t_world_to_intrinsic_input = R_in.TransformWorldToIntrinsic
    
    print("Transform matrices:")
    print(f"Output intrinsic->world:\n{t_intrinsic_to_world_output}")
    print(f"Input world->intrinsic:\n{t_world_to_intrinsic_input}")
    print(f"Geometric transform matrix:\n{tform}")
    
    # Form composite transformation (MATLAB: tComp = tIntrinsictoWorldOutput / tform.A' * tWorldToIntrinsicInput)
    # Note: MATLAB uses post-multiply form
    tform_inv = np.linalg.inv(tform)
    t_comp = t_intrinsic_to_world_output @ tform_inv @ t_world_to_intrinsic_input
    
    # Avoid round-off issues (MATLAB: tComp(:,4)=[0 0 0 1])
    t_comp[:, 3] = [0, 0, 0, 1]
    
    print(f"Composite transformation matrix:\n{t_comp}")
    
    # Check if we should use fast path (equivalent to MATLAB's condition check)
    use_fast_path = (not smooth_edges and 
                    method != 'cubic' and
                    input_image.dtype in [np.uint8, np.int16, np.uint16, np.float32, np.float64])
    
    if use_fast_path:
        print("Using fast path (warp3d)")
        # Call our warp3d implementation (equivalent to MATLAB's images.internal.builtins.warp3d)
        output_image = warp3d(input_image, t_comp, tuple(R_out.ImageSize), fill_values, method)
    else:
        print("Using generic path (coordinate mapping)")
        # Generic path using coordinate mapping (equivalent to MATLAB's fallback)
        output_image = _generic_3d_warp(input_image, t_comp, R_out, method, fill_values)
    
    return output_image


def _generic_3d_warp(input_image, t_comp, R_out, method, fill_values):
    """Generic 3D warping using coordinate mapping (MATLAB fallback path)"""
    
    # Create destination coordinate grid (MATLAB 1-based)
    dst_y, dst_x, dst_z = np.meshgrid(
        np.arange(1, R_out.ImageSize[0] + 1),
        np.arange(1, R_out.ImageSize[1] + 1), 
        np.arange(1, R_out.ImageSize[2] + 1),
        indexing='ij'
    )
    
    # Apply forward transformation to get source coordinates
    dst_coords = np.stack([
        dst_x.flatten(),
        dst_y.flatten(),
        dst_z.flatten(), 
        np.ones_like(dst_x.flatten())
    ])
    
    # Transform using composite matrix
    src_coords = t_comp @ dst_coords
    
    # Extract source coordinates (convert to 0-based)
    src_x = src_coords[0].reshape(R_out.ImageSize) - 1
    src_y = src_coords[1].reshape(R_out.ImageSize) - 1
    src_z = src_coords[2].reshape(R_out.ImageSize) - 1
    
    # Interpolate
    coordinates = np.stack([src_y, src_x, src_z])
    order_map = {'nearest': 0, 'linear': 1, 'cubic': 3}
    order = order_map.get(method, 1)
    
    output_image = ndimage.map_coordinates(
        input_image.astype(np.float32),
        coordinates,
        order=order,
        cval=fill_values,
        mode='constant',
        prefilter=True
    )
    
    return output_image.reshape(R_out.ImageSize).astype(np.float32)


def matlab_imwarp_3d(input_image, tform_matrix, output_view=None, method='linear', fill_values=0):
    """
    Main interface function - Python equivalent of MATLAB's imwarp for 3D
    
    Parameters:
    -----------
    input_image : ndarray
        3D input image
    tform_matrix : ndarray  
        4x4 transformation matrix (affine transform)
    output_view : ImRef3D, optional
        Output spatial referencing. If None, calculated automatically
    method : str
        Interpolation method ('linear', 'nearest', 'cubic')
    fill_values : float
        Fill value for pixels outside input bounds
        
    Returns:
    --------
    output_image : ndarray
        Transformed image
    output_ref : ImRef3D
        Output spatial referencing object
    """
    
    print("=== MATLAB imwarp 3D Implementation ===")
    print(f"Input image shape: {input_image.shape}")
    print(f"Input image dtype: {input_image.dtype}")
    print(f"Transform matrix:\n{tform_matrix}")
    
    # Create input spatial referencing (default: 1-based indexing)
    R_in = ImRef3D(input_image.shape)
    
    if output_view is None:
        # Calculate output spatial referencing automatically
        # For now, use same size as input (you may need to adjust this)
        R_out = ImRef3D(input_image.shape)
        print("Using default output view (same as input)")
    else:
        R_out = output_view
        print(f"Using provided output view: {R_out.ImageSize}")
    
    # Call the main transformation function
    output_image = remap_and_resample_invertible_3d(
        input_image, R_in, tform_matrix, R_out, method, fill_values
    )
    
    print(f"Final output shape: {output_image.shape}")
    print(f"Final output dtype: {output_image.dtype}")
    print(f"Final output range: {np.min(output_image):.6f} to {np.max(output_image):.6f}")
    
    return output_image, R_out


# Test function to verify against your MATLAB results
def test_matlab_imwarp_compatibility():
    """Test function to verify our implementation matches MATLAB"""
    
    # Create a simple test image
    test_image = np.random.rand(64, 64, 32).astype(np.float32)
    
    # Create a simple affine transformation (identity matrix)  
    tform = np.eye(4, dtype=np.float32)
    
    print("Testing with identity transform...")
    output, output_ref = matlab_imwarp_3d(test_image, tform)
    
    # Should be nearly identical for identity transform
    diff = np.abs(output - test_image)
    print(f"Max difference with identity: {np.max(diff):.10f}")
    
    return output, output_ref

if __name__ == "__main__":
    # Run basic test
    test_matlab_imwarp_compatibility()
