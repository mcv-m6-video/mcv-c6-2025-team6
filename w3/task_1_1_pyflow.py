from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
import numpy as np
from PIL import Image
import time
import argparse
import cv2
import os
import png

import argparse

def compute_flow(model, im1, im2, output_dir=None, viz=False):
    '''Computes the optical flow between two images using the specified model.
    
    Args:
        model (str): Optical flow model to use ('pyflow', 'perceiveio', or 'flowformer').
        im1 (numpy.ndarray): First input image.
        im2 (numpy.ndarray): Second input image.
        output_dir (str, optional): Directory to save the output flow. Defaults to None.
        viz (bool, optional): Whether to visualize the optical flow. Defaults to False.

    Raises:
        ValueError: If an invalid model is provided.

    Returns:
        numpy.ndarray: Computed optical flow.
    '''
    # List of supported models
    valid_models = {"pyflow", "flowformer", "perceiveio"}
    
    # Validate the input model
    if model not in valid_models:
        raise ValueError(f"Invalid model: {model}. Must be one of {valid_models}")
    
    # Import and execute the corresponding function based on the selected model
    if model == "pyflow":
        import pyflow
        flow, _ = compute_flow_pyflow(im1, im2, output_dir, viz)
    elif model == "perceiveio":
        # compute_flow_perceiveio()
        pass
    elif model == "flowformer":
        # compute_flow_flowformer()
        pass
    
    return flow

def compute_flow_pyflow(im1, im2, output_dir=None, viz=False):
    '''Compute optical flow using the pyflow method from the pyflow library.

    Args:
        im1 (ndarray): The first input image.
        im2 (ndarray): The second input image.
        output_dir (str, optional): Directory to save the flow result. Defaults to None.
        viz (bool, optional): Whether to visualize and save the optical flow output. Defaults to False.

    Returns:
        tuple: The computed optical flow (u, v) and the time taken for computation.

    Notes:
        This function is adapted from the PyFlow demo script at:
        https://github.com/pathak22/pyflow/blob/master/demo.py  
    '''
    # Create output dir
    if output_dir is not None:
        if os.path.splitext(output_dir)[1]:
            # Get the directory path from the given output path
            dir_path = os.path.dirname(output_dir)
        else:
            # If already a directory, use it as is
            dir_path = output_dir
        os.makedirs(dir_path, exist_ok=True)

    # Normalize images
    im1 = im1.astype(float) / 255.
    im2 = im2.astype(float) / 255.

    if im1.ndim < 3:
        im1 = np.expand_dims(im1, axis=-1)  # Dims need to be (h, w, 1)
        im2 = np.expand_dims(im2, axis=-1)  # Dims need to be (h, w, 1)
        colType = 1  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
    elif im1.ndim == 3:
        colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
    
    # Flow Options:
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30

    # Calc flow
    s = time.time()
    u, v, im2W = pyflow.coarse2fine_flow(im1, 
                                         im2, 
                                         alpha, 
                                         ratio, 
                                         minWidth, 
                                         nOuterFPIterations, 
                                         nInnerFPIterations, 
                                         nSORIterations, 
                                         colType
                                         )
    e = time.time()
    print(f"Time Taken: {e-s:.2f} seconds for image of size {im1.shape}")

    # Save flow in an npy file
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    
    if output_dir is not None:
        if not os.path.splitext(output_dir)[1]:
            flow_file_path = os.path.join(dir_path, "flow_pred_pyflow")
        else:
            flow_file_path = output_dir.split('.')[0] + "_flow_pred_pyflow"      
        np.save(f"{flow_file_path}.npy", flow)

    if viz:
        # hsv = np.zeros(im1.shape, dtype=np.uint8)
        hsv = np.zeros((im1.shape[0], im1.shape[1], 3), dtype=np.uint8)
        hsv[:, :, 0] = 255
        hsv[:, :, 1] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        if output_dir is not None:
            if not os.path.splitext(output_dir)[1]:
                rgb_filename = os.path.join(dir_path, "flow_visual_pyflow")
                warped_filename = os.path.join(dir_path, "flow_warped_pyflow")
            else:
                rgb_filename = output_dir.split('.')[0] + "_flow_visual_pyflow"
                warped_filename = output_dir.split('.')[0] + "_flow_warped_pyflow"
            cv2.imwrite(f"{rgb_filename}.png", rgb)
            cv2.imwrite(f"{warped_filename}.png", im2W[:, :, ::-1] * 255)

    return flow, e-s

def load_flow_gt(gt_path):
    '''Load ground truth optical flow data from a .png file.

    Args:
        gt_path (str): The path to the ground truth optical flow file.

    Returns:
        ndarray: The ground truth optical flow data.

    Notes:
        This function assumes the flow file is in the format used by the KITTI benchmark.
    '''
    flow_gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype(np.float64)
    flow_gt = flow_gt[:, :, :2]  # Keep only the first two channels
    flow_gt /= 64.0  # Convert to real flow values if necessary
    return flow_gt

def read_png_file(flow_file):
    '''Read optical flow data from a KITTI-style .png file.

    Args:
        flow_file (str): The path to the flow file.

    Returns:
        ndarray: Optical flow data in the form of a matrix.

    Notes:
        This function is extracted from the OpticalFlowToolkit repository:
        https://github.com/liruoteng/OpticalFlowToolkit/blob/master/lib/flowlib.py
    '''
    flow_object = png.Reader(filename=flow_file)
    flow_direct = flow_object.asDirect()
    flow_data = list(flow_direct[2])
    (w, h) = flow_direct[3]['size']
    print("Reading %d x %d flow file in .png format" % (h, w))
    flow = np.zeros((h, w, 3), dtype=np.float64)
    for i in range(len(flow_data)):
        flow[i, :, 0] = flow_data[i][0::3]
        flow[i, :, 1] = flow_data[i][1::3]
        flow[i, :, 2] = flow_data[i][2::3]

    invalid_idx = (flow[:, :, 2] == 0)
    flow[:, :, 0:2] = (flow[:, :, 0:2] - 2 ** 15) / 64.0
    flow[invalid_idx, 0] = 0
    flow[invalid_idx, 1] = 0
    return flow

def compute_msen(flow_pred, flow_gt, valid_mask):
    '''Compute the mean squared error (MSEN) between predicted and ground truth optical flow.

    Args:
        flow_pred (ndarray): The predicted optical flow.
        flow_gt (ndarray): The ground truth optical flow.
        valid_mask (ndarray): A mask indicating valid flow values.

    Returns:
        float: The MSEN value.
    '''
    diff = flow_pred - flow_gt  # Difference between flows
    error = np.linalg.norm(diff, axis=2)  # Euclidean norm per pixel
    msen = np.mean(error[valid_mask])  # Mean over valid pixels
    return msen

def compute_pepn(flow_pred, flow_gt, valid_mask, threshold=3.0):
    '''Compute the percentage of erroneous pixels (PEPN) between predicted and ground truth optical flow.

    Args:
        flow_pred (ndarray): The predicted optical flow.
        flow_gt (ndarray): The ground truth optical flow.
        valid_mask (ndarray): A mask indicating valid flow values.
        threshold (float, optional): The threshold for classifying a pixel as erroneous. Defaults to 3.0.

    Returns:
        float: The PEPN value.
    '''
    diff = flow_pred - flow_gt
    error = np.linalg.norm(diff, axis=2)
    erroneous_pixels = (error > threshold) & valid_mask # Pixels with error > threshold
    pepn = np.sum(erroneous_pixels) / np.sum(valid_mask) * 100  # Percentage
    return pepn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "This script computes optical flow between two images using various models.\n"
            "Supported models include 'pyflow', 'flowformer', and 'perceiveio'.\n"
            "It can also compute performance metrics like MSEN and PEPN based on ground truth flow."
        )
    )
    
    parser.add_argument("-inp", dest="input_dir", type=str, default="./data", help="Path to the directory containing the input images (default: ./data).")
    parser.add_argument("-out", dest="output_dir", type=str, default="./results", help="Directory where the results (flow and visualizations) will be saved (default: ./results).")

    parser.add_argument('-img1', dest='img1_path', type=str, default="data/000045_10.png", help='Path to the first image for optical flow computation (default: data/000045_10.png).')
    parser.add_argument('-img2', dest='img2_path', type=str, default="data/000045_11.png", help='Path to the second image for optical flow computation (default: data/000045_11.png.')
    parser.add_argument("-gt", dest="gt_path", type=str, default="data/000045_gt.png", help="Path to the ground truth optical flow file (default: data/000045_gt.png.")

    parser.add_argument("-model", dest="model", type=str, default="pyflow", choices=["pyflow", "flowformer", "perceiveio"], help="Model to use for optical flow computation. Choose from:\n"
                            "  'pyflow'    - Uses the PyFlow method.\n"
                            "  'flowformer' - Placeholder for the Flowformer method (future implementation).\n"
                            "  'perceiveio' - Placeholder for the PerceiveIO method (future implementation).\n"
                            "(default: pyflow).")
    parser.add_argument("-force_compute", dest="force_compute", action="store_true", help="Force optical flow computation (ignore existing flow file).")

    parser.add_argument('-viz', dest='viz', action='store_true', help='Enable visualization of the optical flow result. If set, flow visualization and warped images will be saved.')

    args = parser.parse_args()

    viz = True if args.viz else False

    flow_pred_file = os.path.join(args.input_dir, f"flow_pred_{args.model}.npy")

    if os.path.exists(flow_pred_file) and not args.force_compute:
        flow_pred = np.load(flow_pred_file)
    else:
        img1 = np.array(Image.open(args.img1_path))
        img2 = np.array(Image.open(args.img2_path))
        flow_pred = compute_flow(args.model, img1, img2, args.output_dir, viz=viz)
    
    flow_gt = read_png_file(args.gt_path) # function from flowlib
    valid_mask = flow_gt[..., 2] > 0
    
    msen = compute_msen(flow_pred, flow_gt[..., :2], valid_mask)
    pepn = compute_pepn(flow_pred, flow_gt[..., :2], valid_mask)

    print(f"MSEN: {msen:.2f}")
    print(f"PEPN: {pepn:.2f}%")