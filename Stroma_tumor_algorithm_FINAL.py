"""
Author: Yuri Tolkach

For non-commercial, academic research use only

(c) Yuri Tolkach
"""

import os
import statistics

import numpy as np
from PIL import Image, ImageOps
import cv2
import statistics as st

###PATHES
DIR_PATH_MASK = './mask_tsr_final/' #png 1-layer pixel-wise class maps
DIR_PATH_ORIGINAL = './overlay_tsr_final/' #overlay image from model inference
DIR_OUTPUT = './01_1000_TU0.1_TIS_0.95_BACK0.1_NECR0.2_MUC0.3/'
FILE_OUTPUT = 'BATCH_NAME'

'''
All masks with pixel-wise class information are standardized with regard to MPP (=1.0).
It is good as we do not need to account for different mpps. 

Overlays are generated with downsample of 8x.
Therefore they are approximately 5x, ca. MPP=2.0
Based on this we can calculate ideal size of region from overlays.
E.g., good size of patch will be 500 px in overlay (at mpp=2.0)
This is 1000 mkm or 1 mm.

At mpp=1.0 in mask the patch size will be 1000 px.
'''

### PARAMETERS
P_S = 1000 #It is region size for analysis of tumor/stroma ratio.
OVERLAP = 0.3 #Overlap of single patches to make smooth transitions.
THR_PATCH_TU = 0.1 #Threshold to trigger patch analysis (content of tumor).
THR_PATCH_TARG_TIS = 0.95 #Threshold to trigger patch analysis (content of tumor and stroma, patch should normally contain
# only these two classes).
THR_BACK = 0.1
THR_NECR = 0.2
THR_MUCIN = 0.3

### CODE
CODE_TU = 1
CODE_STR = 3
CODE_MUCIN = 10
CODE_BACK = 11
CODE_NECR = 8

### FLAGS
CIRCLE_FLAG = True
OVERLAY_ON_TUMOR_FLAG = False
OVERLAY_ON_TUMOR_STR_FLAG = True

##############################################################
#SCRIPT


#Prepare circle

circle = Image.open('Circle_CROP.png')
circe_res = circle.resize((P_S,P_S))
circle_res_np = np.array(circe_res)
circle_res_np_bool = circle_res_np > 0

test_D_L = Image.open('Test_D_L.png')
test_D_L = test_D_L.resize((P_S,P_S))
test_D_L = np.array(test_D_L)
test_D_L = test_D_L > 0

test_D_R = Image.open('Test_D_R.png')
test_D_R = test_D_R.resize((P_S,P_S))
test_D_R = np.array(test_D_R)
test_D_R = test_D_R > 0

test_U_L = Image.open('Test_U_L.png')
test_U_L = test_U_L.resize((P_S,P_S))
test_U_L = np.array(test_U_L)
test_U_L = test_U_L > 0

test_U_R = Image.open('Test_U_R.png')
test_U_R = test_U_R.resize((P_S,P_S))
test_U_R = np.array(test_U_R)
test_U_R = test_U_R > 0


#Read mask images from the folder
image_names = sorted(os.listdir(DIR_PATH_MASK))

#Create dirs
dir_output_overlay = DIR_OUTPUT + 'overlay/'
dir_ouput_mask = DIR_OUTPUT + 'mask/'

try:
    os.mkdir(dir_output_overlay)
    os.mkdir(dir_ouput_mask)
except:
    print('Output dirs are already there ...')

file_name = DIR_OUTPUT + FILE_OUTPUT + str(P_S) + '_' + str(OVERLAP) + '.txt'

with open(file_name, "a") as f:
    f.write('FILENAME' + "\t" + 'PATCHES_ANALYZED' + "\t" +
            'MIN' + "\t" + 'MAX' + "\t" + 'MED' + "\t"  + "\n")

for y, image_name in enumerate(image_names):

    print ('Processing', y + 1, '/', len(image_names), ' ', image_name)

    file_name = DIR_OUTPUT + FILE_OUTPUT + str(P_S) + '_' + str(OVERLAP) + '.txt'

    with open(file_name, "a") as f:
        f.write(image_name + "\t")

    image = np.array(Image.open(DIR_PATH_MASK + image_name))

    #####
    #####
    ##### ONLY FOR INITIAL RESIZED MAPS (TO OPTIMIZE ANALYSIS SPEED)
    #image = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    #####
    #####
    #####

    original = Image.open(DIR_PATH_ORIGINAL + image_name [:-14] + 'overlay_final.jpg')
    original = original.resize((image.shape[1], image.shape[0]), Image.ANTIALIAS)
    ##width>height
    height,width = image.shape


    ''' For later:
    if width < P_S or height < P_S:
            continue
    '''

    p_s_ov = int(P_S * OVERLAP)  # OVERLAP 0.5 = 500 px.
    wi_n = (width - P_S) // p_s_ov # Here one patch less, one full patch at the end is not accounted for
    he_n = (height - P_S) // p_s_ov

    image_full = image [0:he_n*p_s_ov + P_S, 0:wi_n*p_s_ov + P_S]
    original = np.array(original) [0:he_n*p_s_ov + P_S, 0:wi_n*p_s_ov + P_S]


    mask_ratio_image_full = np.zeros((he_n*p_s_ov + P_S, wi_n*p_s_ov + P_S))

    counter_over_threshold = 0
    list_ratios = []

    for h in range (he_n+1): # "+1" for last full patch
        for w in range(wi_n+1): # "+1" for last full patch
            if w == 0 and h == 0:
                #image_work = image_full.crop((w * P_S, h * P_S, (w + 1) * P_S, (h + 1) * P_S))
                h_start = h * P_S
                h_end = (h + 1) * P_S
                w_start = w * P_S
                w_end = (w + 1) * P_S
            elif w == 0 and h != 0:
                #image_work = image_full.crop((w * P_S, h * p_s_ov, (w + 1) * P_S, h * p_s_ov + P_S))
                h_start = h * p_s_ov
                h_end = h * p_s_ov + P_S
                w_start = w * P_S
                w_end = (w + 1) * P_S
            elif w != 0 and h == 0:
                #image_work = image_full.crop((w * p_s_ov, h * P_S, w * p_s_ov + P_S, (h + 1) * P_S))
                h_start = h * P_S
                h_end = (h + 1) * P_S
                w_start = w * p_s_ov
                w_end = w * p_s_ov + P_S
            else:
                #image_work = image_full.crop((w * p_s_ov, h * p_s_ov, w * p_s_ov + P_S, h * p_s_ov + P_S))
                h_start = h * p_s_ov
                h_end = h * p_s_ov + P_S
                w_start = w * p_s_ov
                w_end = w * p_s_ov + P_S

            image_work = image_full[h_start: h_end, w_start: w_end]

            #Now check for patch content
            #1. BACKGROUND < threshold
            back_content = np.sum(image_work == CODE_BACK) / np.sum(image_work < CODE_BACK + 1)

            if back_content > THR_BACK:
                continue

            #2. Tumor > threshold (all classes excluding BACK)
            try:
                tumor_content = np.sum(image_work == CODE_TU) / np.sum(image_work < CODE_BACK)
            except: #Dividing by zero - when only background (== 11) is present
                continue

            necrosis_content = np.sum(image_work == CODE_NECR) / np.sum(image_work < CODE_BACK)
            if necrosis_content > THR_NECR:
                continue

            mucin_content = np.sum(image_work == CODE_MUCIN) / np.sum(image_work < CODE_BACK)
            if mucin_content > THR_MUCIN:
                continue

            #3. There are only tumor-associated classes present > threshold (mucin, necrosis, stroma, tumor).
            target_tissue_content = (np.sum(image_work == CODE_TU) + np.sum(image_work == CODE_STR) + np.sum(image_work == CODE_MUCIN)
                                     + np.sum(image_work == CODE_NECR)) / np.sum(image_work < CODE_BACK)


            if  tumor_content > THR_PATCH_TU and target_tissue_content > THR_PATCH_TARG_TIS:
                #print (">threshold")
                D_L = np.where(test_D_L [:,:,1] == True, 0, image_work)
                U_L = np.where(test_U_L [:,:,1] == True, 0, image_work)
                U_R = np.where(test_U_R [:,:,1] == True, 0, image_work)
                D_R = np.where(test_D_R [:,:,1] == True, 0, image_work)

                if np.sum(D_L == CODE_TU) < 10000 or \
                        np.sum(U_L == CODE_TU) < 10000 or np.sum(U_R == CODE_TU) < 10000 or np.sum(D_R == CODE_TU) < 10000:
                    continue

                counter_over_threshold += 1

                '''
                Ratio would be calculated as tumor / (tumor + stroma)
                '''
                ratio_tu_str = round(np.sum(image_work == CODE_STR) /
                                     (np.sum(image_work == CODE_TU) + np.sum(image_work == CODE_STR)), 2)

                list_ratios.append(ratio_tu_str)

                mask_ratio_image_work = np.full(image_work.shape, ratio_tu_str)

                if CIRCLE_FLAG == True:
                    mask_ratio_image_work = np.where(circle_res_np_bool [:,:,0] == True, 0, mask_ratio_image_work)


                ### Test of content



                #reduce ratio mappings to tumor and stroma parts of the patch

                '''
                #Heatmap visualized for tumor and stroma pixels
                mask_ratio_image_work = np.where(np.logical_and(image_work != CODE_TU, image_work != CODE_STR),
                                                 0, mask_ratio_image_work)
                '''

                #Heatmap visualized for tumor pixels only
                if OVERLAY_ON_TUMOR_FLAG == True:
                    mask_ratio_image_work = np.where(image_work != CODE_TU, 0, mask_ratio_image_work)

                if OVERLAY_ON_TUMOR_STR_FLAG == True:
                    mask_ratio_image_work = np.where(np.logical_and(image_work != CODE_TU, image_work != CODE_STR), 0, mask_ratio_image_work)


                subsample = mask_ratio_image_full[h_start: h_end, w_start: w_end]

                subsample = np.where(np.logical_and(subsample != 0, mask_ratio_image_work != 0),
                                     (subsample + mask_ratio_image_work) / 2, subsample)

                subsample = np.where(np.logical_and(subsample == 0, mask_ratio_image_work != 0),
                                     mask_ratio_image_work, subsample)

                mask_ratio_image_full[h_start: h_end, w_start: w_end] = subsample

    #Now making heatmap from ratios. It would be overlaid onto original.
    heatmap = cv2.applyColorMap(np.uint8(mask_ratio_image_full*255), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    '''
    heatmap [:,:,0] = np.where(image_full != 1, 128, heatmap[:,:,0])
    heatmap [:,:,1] = np.where(image_full != 1, 128, heatmap[:,:,1])
    heatmap [:,:,2] = np.where(image_full != 1, 128, heatmap[:,:,2])
    '''

    # Overlay heatmap on image
    overlay = cv2.addWeighted(original, 0.5, heatmap, 0.5, 0)
    overlay = Image.fromarray(overlay)
    overlay.save(dir_output_overlay + image_name[:-8] + 'tu_str_overlay_' + str(P_S) + '_' + str(OVERLAP) + '.jpg')

    # Save mask as npy
    #np.savetxt(dir_ouput_mask + image_name[:-8] + 'tu_str_mask_' + str(P_S) + '_' + str(OVERLAP) + '.npy', mask_ratio_image_full)

    # Save mask as tif
    #imsave(dir_ouput_mask + image_name[:-8] + 'tu_str_mask_' + str(P_S) + '_' + str(OVERLAP) + '.tif', mask_ratio_image_full)

    #Calculate metrics

    if sum(list_ratios) > 0:
        ratio_min = round(min(list_ratios), 2)
        ratio_max = round(max(list_ratios), 2)
        ratio_med = round(statistics.median(list_ratios), 2)


    else:
        ratio_min = 'NA'
        ratio_max = 'NA'
        ratio_med = 'NA'




    #Write the file with ratios
    with open(file_name, "a") as f:
        f.write(str(counter_over_threshold) + "\t")
        f.write(str(ratio_min) + "\t")
        f.write(str(ratio_max) + "\t")
        f.write(str(ratio_med) + "\t")
        f.write("\t".join(map(str, list_ratios)) + "\n")