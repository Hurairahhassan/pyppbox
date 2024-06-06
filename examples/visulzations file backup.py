# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                           #
#   pyppbox: Toolbox for people detecting, tracking, and re-identifying.    #
#   Copyright (C) 2022 UMONS-Numediart                                      #
#                                                                           #
#   This program is free software: you can redistribute it and/or modify    #
#   it under the terms of the GNU General Public License as published by    #
#   the Free Software Foundation, either version 3 of the License, or       #
#   (at your option) any later version.                                     #
#                                                                           #
#   This program is distributed in the hope that it will be useful,         #
#   but WITHOUT ANY WARRANTY; without even the implied warranty of          #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           #
#   GNU General Public License for more details.                            #
#                                                                           #
#   You should have received a copy of the GNU General Public License       #
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.  #
#                                                                           #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


import cv2
import numpy as np
import time
from .persontools import Person
from .commontools import getCVMat
from .logtools import add_error_log, add_warning_log

# For ultralytics's skeleton
has_ultralytics = True
try:
    from ultralytics.utils.plotting import Colors
    colors = Colors()
    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], 
                [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    limb_color = colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    kpt_color = colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
except ImportError as e:
    has_ultralytics = False
    add_warning_log("visualizetools: ultralytics or pyppbox-ultralytics is not installed.")

# For cid
cid_col = (0, 0, 255)
cid_font_thickness = 2
cid_font = cv2.FONT_HERSHEY_COMPLEX_SMALL

# For faceid
faceid_col = (255, 0, 255)
faceid_font_thickness = 2
faceid_footnote_text = "faceid"
faceid_font = cv2.FONT_HERSHEY_COMPLEX_SMALL

# For deepid
deepid_col = (0, 255, 255)
deepid_font_thickness = 2
deepid_footnote_text = "deepid"
deepid_font = cv2.FONT_HERSHEY_COMPLEX_SMALL

# For foot note
footnote_font_scale = 1
footnote_font_thickness = 2
footnote_font = cv2.FONT_HERSHEY_COMPLEX_SMALL

# For reid
reid_pos = (125, 30)
reid_col = (0, 255, 255)
reid_dup_col = (0, 0, 255)
reid_status_font = cv2.FONT_HERSHEY_COMPLEX_SMALL


def __addSKL__(img, kpts, radius=5, kpt_line=True):
    # Ultralytics YOLO
    h, w, c = img.shape
    shape = (h, w)
    nkpt, ndim = kpts.shape
    is_pose = nkpt == 17 and ndim == 3
    kpt_line &= is_pose
    for i, k in enumerate(kpts):
        color_k = [int(x) for x in kpt_color[i]] if is_pose else colors(i)
        x_coord, y_coord = k[0], k[1]
        if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
            if len(k) == 3:
                conf = k[2]
                if conf < 0.5: continue
            cv2.circle(img, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)
    if kpt_line:
        ndim = kpts.shape[-1]
        for i, sk in enumerate(skeleton):
            pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
            pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
            if ndim == 3:
                conf1 = kpts[(sk[0] - 1), 2]
                conf2 = kpts[(sk[1] - 1), 2]
                if conf1 < 0.5 or conf2 < 0.5: continue
            if (pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or 
                pos1[0] < 0 or pos1[1] < 0): continue
            if (pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or 
                pos2[0] < 0 or pos2[1] < 0): continue
            cv2.line(img, pos1, pos2, [int(x) for x in limb_color[i]], 
                     thickness=2, lineType=cv2.LINE_AA)
    return img

def visualizePeople(img, people, tracking_dict, show_box=True, show_skl=(True,True,5), show_ids=(True,True,True), 
                    show_reid=(0,0), show_repspoint=True, img_is_mat=True):
    if img_is_mat and isinstance(img, str): 
        img_is_mat = False
    if not img_is_mat: 
        img = getCVMat(img)

    if isinstance(people, list):
        if len(people) > 0:
            if isinstance(people[0], Person):
                (h, w, c) = img.shape

                for p in people:
                    x, y = p.repspoint
                    deepid = p.deepid
                    deepid_col = (0, 0, 255) if deepid == "Unknown" else (0, 255, 0)
                    
                    # Retrieve and calculate elapsed time in minutes
                    if deepid in tracking_dict:
                        elapsed_time_seconds = int(time.time() - tracking_dict[deepid]['start_time'])
                        elapsed_minutes = elapsed_time_seconds // 60
                        remaining_seconds = elapsed_time_seconds % 60
                        deepid_text = f"{deepid} ({elapsed_minutes}m {remaining_seconds}s)"
                    else:
                        deepid_text = deepid

                    if show_box:
                        cv2.rectangle(img, (p.box_xyxy[0], p.box_xyxy[1]), 
                                      (p.box_xyxy[2], p.box_xyxy[3]), (255, 255, 0), 2)
                    if show_repspoint:
                        cv2.circle(img, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
                    if has_ultralytics and isinstance(show_skl, tuple) and len(p.keypoints) >= 15:
                        if show_skl[0]: img = __addSKL__(img, p.keypoints, radius=show_skl[2], kpt_line=show_skl[1])
                    if isinstance(show_ids, tuple):
                        if show_ids[0]:
                            cv2.putText(img, str(p.cid), (x - 10, y - 65), cid_font, 
                                        1, cid_col, cid_font_thickness)
                        if show_ids[1]:
                            cv2.putText(img, deepid_text, 
                                        (x - 90, y - 90), deepid_font, 1, deepid_col, deepid_font_thickness)
                        if show_ids[2]:
                            cv2.putText(img, str(p.faceid), 
                                        (x - 90, y - 350), faceid_font, 1, faceid_col, faceid_font_thickness)
                    if isinstance(show_reid, tuple):
                        if show_reid[0] > 0:
                            cv2.putText(img, "REIDING", reid_pos, reid_status_font, 1, reid_col, 1, cv2.LINE_AA)
                        if show_reid[1] > 0:
                            cv2.putText(img, "DEDUPLICATING", reid_pos, reid_status_font, 1, reid_dup_col, 1, cv2.LINE_AA)
            else:
                add_error_log("visualizePeople() -> Input 'people' list has unsupported element.")
                raise ValueError("Input 'people' list has unsupported element.")
    return img
