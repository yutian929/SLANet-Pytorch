import numpy as np

def compare_npy_files(file1, file2, rtol=1e-5, atol=1e-8):
    # Load the data from the .npy files
    try:
        data1 = np.load(file1)
        data2 = np.load(file2)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # Print the shapes of the two arrays
    print(f"Shape of {file1}: {data1.shape}")
    print(f"Shape of {file2}: {data2.shape}")

    # Compare shapes
    if data1.shape == data2.shape:
        print("Shapes are consistent.")
        # Compare the data for numerical closeness
        is_close = np.allclose(data1, data2, rtol=rtol, atol=atol)
        if is_close:
            print("The data in both files are numerically close enough. ✅")
        else:
            print("The data in the files are not close enough: ❌")
            # Optionally, find and report some statistics about the differences
            diff = np.abs(data1 - data2)
            max_diff = np.max(diff)
            num_mismatches = np.count_nonzero(diff > atol + rtol * np.abs(data2))

            print(f"Mismatched elements: {num_mismatches}")
            print(f"Max absolute difference: {max_diff}")
            # Find indices of max difference
            max_diff_indices = np.unravel_index(np.argmax(diff, axis=None), diff.shape)
            print(f"Indices of max difference: {max_diff_indices}, values: {data1[max_diff_indices]}, {data2[max_diff_indices]}")
    else:
        print("Shapes are not consistent. Cannot compare the data numerically.")

if __name__ == "__main__":
    DEBUG_DATA_DIR = "./debug_data"
    PADDLE_DEBUG_DATA_DIR = DEBUG_DATA_DIR + "/paddle"
    PYTORCH_DEBUG_DATA_DIR = DEBUG_DATA_DIR + "/pytorch"
    # Example usage:
    paddle_files = ['/input.npy', '/backbone_output_0.npy', '/backbone_output_1.npy', '/backbone_output_2.npy', '/backbone_output_3.npy',
            '/neck_output_0.npy', '/neck_output_1.npy', '/neck_output_2.npy', '/neck_output_3.npy', '/head_output_structure_probs.npy', '/head_output_loc_preds.npy']
    pytorch_files = ['/input.npy', '/backbone_output_0.npy', '/backbone_output_1.npy', '/backbone_output_2.npy', '/backbone_output_3.npy',
            '/neck_output_0.npy', '/neck_output_1.npy', '/neck_output_2.npy', '/neck_output_3.npy', '/head_output_structure_probs.npy', '/head_output_loc_preds.npy']
    assert len(paddle_files) == len(pytorch_files)
    for idx in range(len(paddle_files)):
        paddle_file = PADDLE_DEBUG_DATA_DIR + paddle_files[idx]
        pytorch_file = PYTORCH_DEBUG_DATA_DIR + pytorch_files[idx]
        print(f"Comparing {paddle_file} and {pytorch_file}")
        compare_npy_files(paddle_file, pytorch_file, rtol=0, atol=0.001)