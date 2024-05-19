import os
import cv2
import math
import numpy as np

from PIL import Image


def largest_rotated_rect(w, h, angle):
    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return bb_w - 2 * x, bb_h - 2 * y

def get_rotated_image(image, angle):
    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]
    # affine_mat = (np.matrix(rot_mat))[0:2, :]

    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result

def get_rotated_lines(image, angle, pts):
    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]
    # affine_mat = (np.matrix(rot_mat))[0:2, :]

    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    pts = np.float32(pts)
    if len(pts) != 0:
        src_pt1 = pts[:, [0, 1]]
        src_pt2 = pts[:, [2, 3]]
        pts[:, [0, 1]] = src_pt1 * rot_mat_notranslate
        pts[:, [2, 3]] = src_pt2 * rot_mat_notranslate

        ones = np.ones(shape=(len(src_pt1), 1))  # N, 1
        pt1_ones = np.hstack([src_pt1, ones])    # N, 3
        dst_pt1 = affine_mat.dot(pt1_ones.T).T

        ones = np.ones(shape=(len(src_pt2), 1))  # N, 1
        pt2_ones = np.hstack([src_pt2, ones])    # N, 3
        dst_pt2 = affine_mat.dot(pt2_ones.T).T

        pts[:, [0, 1]] = dst_pt1
        pts[:, [2, 3]] = dst_pt2

    return result, pts

def crop_square_around_center(image, width, height, gt_pts, vis=False):
    """
    Given an image, crops it to the given width and height around its center point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if width > image_size[0]:
        width = image_size[0]

    if height > image_size[1]:
        height = image_size[1]

    width, height = min(width, height), min(width, height)  ###

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    # Calculate the coordinates of the rotated gt lines
    dst_pts = []
    for i in range(gt_pts.shape[0]):
        pts = gt_pts[i]
        gt_x1, gt_y1, gt_x2, gt_y2 = int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3])

        if gt_x1 == gt_x2:
            is_cross = (gt_x1 < x2) and (gt_x1 > x1)
            if is_cross:
                new_line = [gt_x1, y1, gt_x1, y2]
                dst_pts.append(new_line)
        else:
            gt_slope = (gt_y2 - gt_y1) / (gt_x2 - gt_x1)
            is_cross1 = (gt_slope * (x1 - gt_x1) + gt_y1 - y1) * (gt_slope * (x1 - gt_x1) + gt_y1 - y2) < 0     # cross left line   : (x1, y1)-(x1, y2)  --> cx = x1
            is_cross2 = (gt_slope * (x1 - gt_x1) + gt_y1 - y1) * (gt_slope * (x2 - gt_x1) + gt_y1 - y1) < 0     # cross upper line  : (x1, y1)-(x2, y1)  --> cy = y1
            is_cross3 = (gt_slope * (x2 - gt_x1) + gt_y1 - y2) * (gt_slope * (x2 - gt_x1) + gt_y1 - y1) < 0     # cross right line  : (x2, y2)-(x2, y1)  --> cx = x2
            is_cross4 = (gt_slope * (x2 - gt_x1) + gt_y1 - y2) * (gt_slope * (x1 - gt_x1) + gt_y1 - y2) < 0     # cross bottom line : (x2, y2)-(x1, y2)  --> cy = y2
            is_cross = (is_cross1*1 + is_cross2*1 + is_cross3*1 + is_cross4*1) == 2               # if the gt line crosses crop region, it must cross two crop lines
            if is_cross:
                new_line = []
                if is_cross1:
                    cx, cy = x1, gt_slope * (x1 - gt_x1) + gt_y1
                    new_line.append(cx)
                    new_line.append(cy)
                if is_cross2:
                    cx, cy = (y1 - gt_y1) / gt_slope + gt_x1, y1
                    new_line.append(cx)
                    new_line.append(cy)
                if is_cross3:
                    cx, cy = x2, gt_slope * (x2 - gt_x1) + gt_y1
                    new_line.append(cx)
                    new_line.append(cy)
                if is_cross4:
                    cx, cy = (y2 - gt_y1) / gt_slope + gt_x1, y2
                    new_line.append(cx)
                    new_line.append(cy)
                dst_pts.append(new_line)

    dst_pts = np.array(dst_pts)
    translate = np.array([-x1, -y1, -x1, -y1])
    if vis:
        new_image = np.ascontiguousarray(np.copy(image))
        save_dir = f'./result/Multi_task/'
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(save_dir + 'input.jpg', image)

        for i in range(gt_pts.shape[0]):
            pts = gt_pts[i]
            pt_1 = (int(pts[0]), int(pts[1]))
            pt_2 = (int(pts[2]), int(pts[3]))
            image = cv2.line(image, pt_1, pt_2, color=(255, 255, 0), thickness=3)

        for i in range(dst_pts.shape[0]):
            pts = dst_pts[i]
            pt_1 = (int(pts[0]), int(pts[1]))
            pt_2 = (int(pts[2]), int(pts[3]))
            new_image = cv2.line(new_image, pt_1, pt_2, color=(255, 255, 0), thickness=3)
            dst_pts[i] = dst_pts[i] + translate

        cv2.imwrite(save_dir + 'input_with_gt_lines.jpg', image)
        cv2.imwrite(save_dir + 'input_with_new_gt_lines.jpg', new_image)
        cv2.imwrite(save_dir + 'cropped.jpg', new_image[y1:y2, x1:x2])
    else:
        for i in range(dst_pts.shape[0]):
            dst_pts[i] = dst_pts[i] + translate

    return image[y1:y2, x1:x2], dst_pts

def random_rotate_with_line(cfg, image, rotate_angle, gt_lines):
    img_size = cfg.image_size
    image_height, image_width = image.shape[0:2]

    x1 = gt_lines[:, 0] / (img_size - 1) * (image_width - 1)
    y1 = gt_lines[:, 1] / (img_size - 1) * (image_height - 1)
    x2 = gt_lines[:, 2] / (img_size - 1) * (image_width - 1)
    y2 = gt_lines[:, 3] / (img_size - 1) * (image_height - 1)
    gt_lines = np.stack([x1, y1, x2, y2], axis=1)

    image_rotated, gt_line_pts = get_rotated_lines(image, rotate_angle, gt_lines)
    image_rotated_cropped, gt_line_pts = crop_square_around_center(
        image_rotated,
        *largest_rotated_rect(image_width, image_height, math.radians(rotate_angle)),
        gt_line_pts,
        vis=False
    )

    if len(gt_line_pts) == 0:
        return image, gt_lines

    return image_rotated_cropped, gt_line_pts


def random_rotate_just_img(image, rotate_angle):
    image_rotated = get_rotated_image(image, rotate_angle)
    return image_rotated

