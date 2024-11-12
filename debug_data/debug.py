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

# Example usage:
file1s = ['./paddle/input.npy', './paddle/backbone_output_0.npy.npy', './paddle/backbone_output_1.npy.npy', './paddle/backbone_output_2.npy.npy', './paddle/backbone_output_3.npy.npy',
          './paddle/neck_output_0.npy.npy', './paddle/neck_output_1.npy.npy', './paddle/neck_output_2.npy.npy', './paddle/neck_output_3.npy.npy', './paddle/head_output_structure_probs.npy.npy', './paddle/head_output_loc_preds.npy.npy']
file2s = ['./pytorch/input.npy', './pytorch/backbone_output_0.npy', './pytorch/backbone_output_1.npy', './pytorch/backbone_output_2.npy', './pytorch/backbone_output_3.npy',
          './pytorch/neck_output_0.npy', './pytorch/neck_output_1.npy', './pytorch/neck_output_2.npy', './pytorch/neck_output_3.npy', './pytorch/head_output_0.npy', './pytorch/head_output_1.npy']
assert len(file1s) == len(file2s)
for idx in range(len(file1s)):
    file1 = file1s[idx]
    file2 = file2s[idx]
    print(f"Comparing {file1} and {file2}")
    compare_npy_files(file1, file2, rtol=0, atol=0.001)