"""
Numerosity Analysis and Object Detection Utilities

This script contains utility functions and methods for analyzing numerosity confusion matrices,
assessing human-likeness of behavior, and evaluating scalar variability. Additionally, it includes
tools for image processing, detection result filtering, and working with GroundingDINO models.

Key Components:
- Image loading and preprocessing (`load_image`).
- GroundingDINO model loading (`load_GDINO`).
- Confusion matrix analysis (`MWE`, `knower_level`, `human_likeness`, `scalar_variability`).
- Filtering detection results based on logits, NMS, and bounding box area (`filter_results_by_logit`).

Author: Kuinan Hou
License: CC0 1.0 Universal
"""
import numpy as np
import statsmodels.api as sm
import scipy as sci
from torchvision.ops import nms
import torch
import warnings
from huggingface_hub import hf_hub_download
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from GroundingDINO.groundingdino.models import build_model
import GroundingDINO.groundingdino.datasets.transforms as T

from typing import Tuple, Union
from PIL import Image

def load_image(input_data: Union[str, np.array, Image.Image, torch.Tensor]) -> Tuple[np.array, torch.Tensor]:
    """
    Load and transform an image from a file path, numpy array, PIL Image, or torch Tensor.

    Args:
        input_data (Union[str, np.array, Image.Image, torch.Tensor]): The input image, which can be a file path, a numpy array, a PIL Image, or a torch Tensor.

    Returns:
        Tuple[np.array, torch.Tensor]:
            - Original image as a numpy array.
            - Transformed image as a torch.Tensor.
    """
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Handle input type
    if isinstance(input_data, str):
        # Input is a file path
        image_source = Image.open(input_data).convert("RGB")
    elif isinstance(input_data, np.ndarray):
        # Input is a numpy array
        image_source = Image.fromarray(input_data).convert("RGB")
    elif isinstance(input_data, Image.Image):
        # Input is a PIL Image
        image_source = input_data.convert("RGB")
    elif isinstance(input_data, torch.Tensor):
        # Input is a torch Tensor
        if input_data.dim() == 3 and input_data.size(0) in {1, 3}:  # C x H x W format
            input_data = input_data.permute(1, 2, 0).numpy()  # Convert to H x W x C
            image_source = Image.fromarray((input_data * 255).astype(np.uint8)).convert("RGB")
        else:
            raise ValueError("Torch Tensor must have shape [C, H, W] with 1 or 3 channels.")
    else:
        raise ValueError("Input must be a file path, numpy array, PIL Image, or torch Tensor.")

    # Convert the PIL Image to a numpy array
    image = np.asarray(image_source)

    # Apply transformations
    image_transformed, _ = transform(image_source, None)

    return image, image_transformed



# G-DINO model loading function
def load_GDINO(repo_id, filename, ckpt_config_filename, device='cpu'):
    """
    A customized function to load Grounding DINO model
    """
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    #log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    #print(f"Model loaded from {cache_file} \n => {log}")
    
    _ = model.eval()
    #print('Model in eval mode')
    return model  

def MWE(confusion_matrix):
    """
    Calculate the mean weighted error (MWE) for a given confusion matrix.
    
    By default, this function assumes that the zero point (the origin of indexing for the generated numerosity)
    of the confusion matrix is at the bottom-left. However, it will automatically attempt to detect whether
    the zero point is actually at the top-left or bottom-left by examining which diagonal (top-left to 
    bottom-right or bottom-left to top-right) contains the majority of the data.

    Logic for zero point detection:
    - If the majority of the data lies along the top-left to bottom-right diagonal, 
      then the zero point is at the top-left.
    - If the majority of the data lies along the bottom-left to top-right diagonal,
      then the zero point is at the bottom-left (default).

    If the zero point is detected to be at the top-left (contrary to the default assumption),
    a warning will be raised and the confusion matrix will be printed.

    Args:
        confusion_matrix (np.ndarray): 
            A 2D NumPy array representing the confusion matrix.
            The rows and columns correspond to generated and target numerosities, respectively.
            Dimensions:
              - Rows: 20 (generated)
              - Columns: 10 (target)

    Returns:
        float: The mean weighted error.
    """
    # Get dimensions
    num_generated = confusion_matrix.shape[0]  # Should be 20
    num_target = confusion_matrix.shape[1]     # Should be 10

    # Compute sums along the two possible main diagonals:
    # 1. Top-left to bottom-right (tl_br)
    # 2. Bottom-left to top-right (bl_tr)
    min_dim = min(num_generated, num_target)
    diag_tl_br = sum(confusion_matrix[i, i] for i in range(min_dim))
    diag_bl_tr = sum(confusion_matrix[num_generated - 1 - i, i] for i in range(min_dim))

    # Detect zero point
    # Default is bottom-left, so if top-left diagonal is stronger, we raise a warning
    if diag_tl_br > diag_bl_tr:
        zero_point = "top-left"
    else:
        zero_point = "bottom-left"

    if zero_point == "top-left":
        warnings.warn("Detected zero point at top-left, but default is bottom-left. Using top-left indexing.")
        print("Confusion Matrix:\n", confusion_matrix)
    print("Confusion Matrix:\n", confusion_matrix)
    # Initialize accumulators
    total_weighted_error = 0.0
    total_counts = 0
    
    # Calculate mean weighted error
    # If zero point is bottom-left:
    #   The bottom row of the matrix corresponds to generated numerosity = 1
    #   and the top row corresponds to generated numerosity = num_generated.
    #   Since numpy indexing starts at the top, we need to invert the indexing for generated.
    #
    # If zero point is top-left:
    #   The top row corresponds to generated numerosity = 1, as in the original code.
    
    for generated_idx in range(num_generated):
        for target_idx in range(num_target):
            count = confusion_matrix[generated_idx, target_idx]
            if count > 0:
                if zero_point == "bottom-left":
                    # Invert the row index for generated
                    generated_n = num_generated - generated_idx
                else:
                    # top-left zero point as in original code
                    generated_n = generated_idx + 1
                
                # Target indexing is always left-to-right
                target_n = target_idx + 1

                error = abs(generated_n - target_n) / target_n
                total_weighted_error += error * count
                total_counts += count

    mean_weighted_error = total_weighted_error / total_counts if total_counts > 0 else 0.0

    return mean_weighted_error


def filter_results_by_logit(
    filtered_results,
    logit_threshold=0.40,
    apply_nms=True,
    nms_iou_threshold=0.95,
    remove_large_boxes=False,
    max_area_ratio=0.95):
    """
    Filters detection results based on a logit threshold, applies NMS suppression,
    and removes large bounding boxes that cover a specified area of the image.

    Parameters:
    - filtered_results (dict): The detection results loaded from the .pkl file.
                               The expected format is:
                               {
                                   'image_id1': {
                                       'boxes': [...],
                                       'logits': [...],
                                       'phrases': [...],
                                       'prompts': [...]
                                   },
                                   'image_id2': {
                                       ...
                                   },
                                   ...
                               }
    - logit_threshold (float, optional): The threshold for filtering logits.
                                         Detections with logits below this
                                         threshold will be discarded.
                                         Default is 0.35.
    - apply_nms (bool, optional): Whether to apply Non-Maximum Suppression (NMS)
                                  to the remaining bounding boxes. Default is False.
    - nms_iou_threshold (float, optional): The IoU threshold for NMS. Boxes with
                                           IoU greater than this value will be suppressed.
                                           Default is 0.95.
    - remove_large_boxes (bool, optional): Whether to remove bounding boxes that
                                           cover more than `max_area_ratio` of the
                                           image area. Default is False.
    - max_area_ratio (float, optional): The maximum allowed area ratio of a bounding
                                        box relative to the image area. Boxes exceeding
                                        this ratio will be removed. Default is 0.95.

    Returns:
    - filtered_filtered_results (dict): The filtered detection results in the
                                        same format as the input, containing only
                                        detections that meet the specified criteria.
    """
    # Initialize a new dictionary to store the filtered results
    filtered_filtered_results = {}

    # Loop over each image in the results
    for filename, data in filtered_results.items():
        if data is None:
            continue
        boxes = data['boxes']     # List of boxes in (cx, cy, w, h) format, normalized
        logits = data['logits']   # List of logits
        phrases = data['phrases'] # List of phrases
        prompts = data['prompts'] # List of prompts

        # Check if there are any detections
        if not boxes:
            continue

        # Filter indices where logits are above or equal to the threshold
        filtered_indices = [i for i, logit in enumerate(logits) if logit >= logit_threshold]

        # If no boxes meet the threshold, skip this image
        if not filtered_indices:
            continue

        # Filter the boxes and associated data
        filtered_boxes = [boxes[i] for i in filtered_indices]
        filtered_logits = [logits[i] for i in filtered_indices]
        filtered_phrases = [phrases[i] for i in filtered_indices]
        

        # Remove bounding boxes that cover too much of the image area
        if remove_large_boxes:
            new_filtered_boxes = []
            new_filtered_logits = []
            new_filtered_phrases = []
            
            for i, (cx, cy, w, h) in enumerate(filtered_boxes):
                box_area_ratio = w * h  # Normalized area as a fraction of the image
                if box_area_ratio <= max_area_ratio:
                    new_filtered_boxes.append((cx, cy, w, h))
                    new_filtered_logits.append(filtered_logits[i])
                    new_filtered_phrases.append(filtered_phrases[i])
                    #new_filtered_prompts.append(filtered_prompts[i])
            filtered_boxes = new_filtered_boxes
            filtered_logits = new_filtered_logits
            filtered_phrases = new_filtered_phrases
            

            # If no boxes remain after filtering, skip this image
            if not filtered_boxes:
                continue

        # Apply Non-Maximum Suppression if requested
        if apply_nms:
            # Convert (cx, cy, w, h) to (x1, y1, x2, y2) format for NMS
            boxes_tensor = torch.tensor([
                [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
                for cx, cy, w, h in filtered_boxes
            ])
            logits_tensor = torch.tensor(filtered_logits)
            
            # Perform NMS
            keep_indices = nms(boxes_tensor, logits_tensor, nms_iou_threshold).tolist()

            # Filter the boxes and associated data based on NMS results
            filtered_boxes = [filtered_boxes[i] for i in keep_indices]
            filtered_logits = [filtered_logits[i] for i in keep_indices]
            filtered_phrases = [filtered_phrases[i] for i in keep_indices]
            

            # If no boxes remain after NMS, skip this image
            if not filtered_boxes:
                continue

        # Store the filtered data back into the new dictionary
        filtered_filtered_results[filename] = {
            'boxes': filtered_boxes,
            'logits': filtered_logits,
            'phrases': filtered_phrases,
            'prompts': prompts
        }

    return filtered_filtered_results

def knower_level(CM):
    """
    Evaluates the knower-level given a response confusion matrix

    Parameters:
    - CM: confusion matrix for the response, generated from the create_CM() function

    Returns:
    - Knower-level
    """
    average_CM = np.round(np.mean(CM[:,1:,:], axis = 0)) # do not count invalid answers (set to 20)
    #print(average_CM)
    stim_per_num = sum(average_CM)
    #print(stim_per_num)

    # @title
    average_CM = np.round(np.mean(CM[:,1:,:], axis = 0)) # do not count invalid answers (set to 20)
    #print(average_CM)
    stim_per_num = sum(average_CM)
    #print(stim_per_num)
    knower = 0
    # 1-knower?
    CM_1 = np.flip(average_CM[:,0])
    CM_2 = np.flip(average_CM[:,1])
    CM_3 = np.flip(average_CM[:,2])
    CM_4 = np.flip(average_CM[:,3])
    CM_5 = np.flip(average_CM[:,4])
    if CM_1[0] > 0.67*stim_per_num[0]:
        if CM_2[0] < 0.5*stim_per_num[1]:
            knower = 1

    if CM_2[1] > 0.67*stim_per_num[1]:
        if CM_3[1] < 0.5*stim_per_num[2]:
            knower = 2

    if CM_3[2] > 0.67*stim_per_num[2]:
        if CM_4[2] < 0.5*stim_per_num[3]:
            knower = 3

    if CM_4[3] > 0.67*stim_per_num[3]:
        if CM_5[3] < 0.5*stim_per_num[4]:
            knower = 4

    if CM_4[4] > 0.67*stim_per_num[4]:
        if CM_5[4] < 0.5*stim_per_num[5]:
            knower = 5
    print(f'{knower} Knower!')
    return knower

def human_likeness(confusion_matrix):
    """
    Correlation between the agent's response matrix and the reponses from human which follows the Weber's law
    and with random choice baseline.

    Parameters:
    - CM: confusion matrix for the response, generated from the create_CM() function

    Returns:
    - Pearson Correaltion with random choice
    - Pearson Correaltion with human response that follows the Weber's law
    """
    CM_avg = np.mean(confusion_matrix, axis=0) # average confusion matrix
    np.random.seed(13)  # for reproducibility
    weber = 0.15
    # Parameters
    targets = np.arange(1, 11)   # target numerosity: 1 to 10
    n_per_target = 100
    n_total = len(targets) * n_per_target

    # Generate target array
    target_arr = np.repeat(targets, n_per_target)

    # 1) Uniform distribution of generated numerosity
    # Generated numerosity drawn uniformly from [1, 20]
    gen_uniform = np.random.randint(1, 21, size=n_total)

    # 2) Normal distribution of generated numerosity
    # For targets in 1-4 (subitizing range), use no variance (sd = 0).
    gen_normal = []
    for t in targets:
        if 1 <= t <= 4:
            # No variability: all samples are exactly the target number
            samples = np.full(n_per_target, t)
        else:
            # For targets > 4, generate from N(mean=t, sd=0.15 * t) but only valid [1,20]
            samples = []
            while len(samples) < n_per_target:
                # Generate a batch of samples
                batch = np.random.normal(loc=t, scale=weber * t, size=n_per_target)
                # Keep only those in the valid range [1, 20]
                valid_samples = batch[(batch >= 1) & (batch <= 20)]
                samples.extend(valid_samples[:n_per_target - len(samples)])
            # Round to integers
            samples = np.round(samples).astype(int)
        gen_normal.append(samples)

    gen_normal = np.concatenate(gen_normal)

    # Build confusion matrices
    # Rows = generated numerosity (1 to 20), Cols = target numerosity (1 to 10)
    confusion_uniform = np.zeros((20, 10), dtype=int)
    confusion_normal = np.zeros((20, 10), dtype=int)

    for i in range(n_total):
        t = target_arr[i]       # target
        g_u = gen_uniform[i]    # generated from uniform
        g_n = gen_normal[i]     # generated from normal

        confusion_uniform[g_u - 1, t - 1] += 1
        confusion_normal[g_n - 1, t - 1] += 1

    # Suppose CM_avg is defined elsewhere
    corr_norm = np.corrcoef(np.matrix.flatten(CM_avg[1:,:]), np.matrix.flatten(confusion_normal[::-1,:]))[0, 1]
    corr_uniform = np.corrcoef(np.matrix.flatten(CM_avg[1:,:]), np.matrix.flatten(confusion_uniform[::-1,:]))[0, 1]
    #print(f'Correlation with Ideal Human Observer: {corr_norm:.4f}')
    #print(f'Correlation with Random Selection: {corr_uniform:.4f}')
    return corr_norm, corr_uniform

def scalar_variability(confusion_matrix):
    """
    Correlation between the agent's response matrix and the reponses from human which follows the Weber's law
    and with random choice baseline.

    Parameters:
    - CM: confusion matrix for the response, generated from the create_CM() function

    Returns:
    Scalar variability statistic test results
    """
    scalar = False
    # Initialize arrays
    cv = np.zeros((1,10))
    sd = np.zeros((1,10))
    mu = np.zeros((1,10))

    # Compute mu, sd, and cv for each number from 1 to 10.
    for j in range(10):
        freq = np.around(np.flip(confusion_matrix[:, j]))
        freq = freq.astype(int)
        
        # If sum(freq) == 0, no responses. Handle that case by skipping or setting default values.
        if np.sum(freq) == 0:
            # If no responses, set mu=NaN, sd=NaN, cv=NaN or handle as needed.
            # But let's just set them to something neutral:
            mu[0,j] = np.nan
            sd[0,j] = np.nan
            cv[0,j] = np.nan
            continue

        responses = np.repeat(np.arange(1,21), freq)
        mu[0,j] = np.average(responses)
        var = np.var(responses)
        
        # Original sd
        sd_original = np.sqrt(var) / np.sqrt(np.sum(freq))
        # Check if sd_original is zero before adjusting:
        if sd_original == 0:
            # To avoid divide by zero later, replace by a very small number
            sd[0,j] = 1e-10
        else:
            sd[0,j] = sd_original

        cv[0,j] = sd_original / mu[0,j] if mu[0,j] != 0 else np.nan

    # Now we have mu, sd, and cv. 
    # According to the rule:
    # We must remove initial points if their cv is zero and they lie within numbers 1 to 4, and all smaller ones are also zero.
    # Note: j=0 corresponds to number=1, j=1->number=2, etc.
    cv_array = cv[0,:]

    # Determine how many initial points (up to the first four) are zero.
    remove_count = 0
    for idx in range(min(4, 10)):  # only check the first 4 points (or fewer if less than 4 exist)
        if np.isnan(cv_array[idx]):
            # If cv is NaN, treat it as non-zero to stop removal.
            break
        if cv_array[idx] == 0:
            remove_count += 1
        else:
            # Found a non-zero cv within first four points, stop removing here.
            break

    # remove_count now tells how many of the leading points (from the first four) are zero and should be removed.
    # Apply removal to mu and sd arrays before taking logs.
    mu_valid = mu[0,remove_count:]
    sd_valid = sd[0,remove_count:]

    # Take logs
    mu_log = np.log(mu_valid).reshape(-1, 1)
    sd_log = np.log(sd_valid)

    # Perform regression: sd_log vs mu_log
    mu_log_const = sm.add_constant(mu_log)
    ols = sm.OLS(sd_log, mu_log_const)
    ols_result = ols.fit()
    slope = ols_result.params[1]

    f_pvalue = ols_result.f_pvalue
    adj_rsq = ols_result.rsquared_adj
    print('F-test p-value=%.4f, Adjusted R^2=%.4f' % (f_pvalue, adj_rsq))

    t_value = (abs(slope - 1)/ols_result.bse[1]) # test if slope is different from 1
    p_value = sci.stats.t.sf(t_value, ols_result.df_resid)
    print('\nNull hyp: the slope s = %.4f is not different than 1:' % slope)
    print('R-squared=%.2f p=%.4f\n' % (ols_result.rsquared, p_value))

    if p_value > 0.05:
        print('Power or Scalar Variability: Scalar')
        scalar = True
    else:
        print('Power or Scalar Variability: Power')

    return f_pvalue, adj_rsq, t_value, p_value, ols_result._rsquared, scalar