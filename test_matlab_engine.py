#!/usr/bin/env python3

import matlab.engine
import numpy as np

def test_matlab_engine():
    try:
        print("Starting MATLAB Engine...")
        eng = matlab.engine.start_matlab()
        print("MATLAB Engine started successfully!")

        # Basic test
        result = eng.eval("2 + 3")
        print(f"Basic test: 2 + 3 = {result}")

        # Affine2D test (for 2D image warping)
        affine_mat = np.eye(3)  # 3x3 identity matrix
        affine_matlab = matlab.double(affine_mat.tolist(), size=(3,3))
        eng.workspace['test_affine'] = affine_matlab

        eng.eval("tform = affine2d(test_affine);", nargout=0)
        print("affine2d function test: SUCCESS")

        # Generate 2D random image
        test_data = np.random.rand(50, 50)
        test_data_matlab = matlab.double(test_data.tolist(), size=test_data.shape)
        eng.workspace['test_data'] = test_data_matlab

        eng.eval("warped = imwarp(test_data, tform);", nargout=0)
        warped = eng.workspace['warped']  # fetch result
        print(f"imwarp function: SUCCESS, output size = {np.array(warped).shape}")

        eng.quit()
        print("All tests passed!")
        return True

    except Exception as e:
        print(f"MATLAB Engine test failed: {e}")
        return False

if __name__ == "__main__":
    test_matlab_engine()
