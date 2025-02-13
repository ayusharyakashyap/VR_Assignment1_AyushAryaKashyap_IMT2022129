import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os

# Create a directory to save images
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

def solution(left_img, center_img, right_img):
    print("[INFO] Converting images to grayscale...")
    
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    center_gray = cv2.cvtColor(center_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    show_and_save(left_gray, "left_grayscale.jpg", "Left Grayscale Image")
    show_and_save(center_gray, "center_grayscale.jpg", "Center Grayscale Image")
    show_and_save(right_gray, "right_grayscale.jpg", "Right Grayscale Image")

    print("[INFO] Stitching left and center images...")
    stitched_left_center = stitch_pair(left_img, center_img)

    print("[INFO] Stitching result with right image...")
    final_panorama = stitch_pair(stitched_left_center, right_img)

    print("[INFO] Saving and displaying final panorama...")
    show_and_save(final_panorama, "stitched_panorama.jpg", "Final Stitched Panorama")

    return final_panorama

def stitch_pair(img1, img2):
    print("[INFO] Extracting keypoints and descriptors...")
    key_points1, descriptor1, key_points2, descriptor2 = get_keypoints(img1, img2)

    print("[INFO] Matching keypoints...")
    good_matches = match_keypoints(key_points1, key_points2, descriptor1, descriptor2)

    print("[INFO] Computing homography using RANSAC...")
    final_H = ransac(good_matches)

    rows1, cols1 = img2.shape[:2]
    rows2, cols2 = img1.shape[:2]

    points1 = np.float32([[0,0], [0,rows1], [cols1,rows1], [cols1,0]]).reshape(-1,1,2)
    points2 = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)
    points2 = cv2.perspectiveTransform(points2, final_H)

    list_of_points = np.concatenate((points1, points2), axis=0)
    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

    H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]]) @ final_H
    output_img = cv2.warpPerspective(img1, H_translation, (x_max - x_min, y_max - y_min))
    output_img[-y_min:rows1-y_min, -x_min:cols1-x_min] = img2


    return output_img

def get_keypoints(img1, img2):
    sift = cv2.SIFT_create()
    key_points1, descriptor1 = sift.detectAndCompute(img1, None)
    key_points2, descriptor2 = sift.detectAndCompute(img2, None)

    print(f"[INFO] Found {len(key_points1)} keypoints in first image.")
    print(f"[INFO] Found {len(key_points2)} keypoints in second image.")

    # Draw and save keypoints
    img1_keypoints = cv2.drawKeypoints(img1, key_points1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2_keypoints = cv2.drawKeypoints(img2, key_points2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    show_and_save(img1_keypoints, "image1_keypoints.jpg", "Keypoints in First Image")
    show_and_save(img2_keypoints, "image2_keypoints.jpg", "Keypoints in Second Image")

    return key_points1, descriptor1, key_points2, descriptor2

def match_keypoints(key_points1, key_points2, descriptor1, descriptor2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptor1, descriptor2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            left_pt = key_points1[m.queryIdx].pt
            right_pt = key_points2[m.trainIdx].pt
            good_matches.append([left_pt[0], left_pt[1], right_pt[0], right_pt[1]])

    print(f"[INFO] {len(good_matches)} good matches found.")
    return good_matches

def homography(points):
    A = []
    for pt in points:
        x, y, X, Y = pt
        A.append([x, y, 1, 0, 0, 0, -X*x, -X*y, -X])
        A.append([0, 0, 0, x, y, 1, -Y*x, -Y*y, -Y])
    A = np.array(A)
    _, _, vh = np.linalg.svd(A)
    H = vh[-1, :].reshape(3, 3)
    return H / H[2, 2]

def ransac(good_pts, iterations=5000, threshold=5):
    best_inliers = []
    final_H = None

    for _ in range(iterations):
        random_pts = random.choices(good_pts, k=4)
        H = homography(random_pts)
        inliers = []

        for pt in good_pts:
            p = np.array([pt[0], pt[1], 1]).reshape(3, 1)
            p_1 = np.array([pt[2], pt[3], 1]).reshape(3, 1)
            Hp = np.dot(H, p)
            Hp /= Hp[2]
            dist = np.linalg.norm(p_1 - Hp)
            if dist < threshold:
                inliers.append(pt)

        if len(inliers) > len(best_inliers):
            best_inliers, final_H = inliers, H

    print(f"[INFO] RANSAC selected {len(best_inliers)} inliers.")
    return final_H

def show_and_save(image, filename, title):
    """ Save and display an image. """
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, image)
    print(f"[INFO] Saved: {filepath}")

    plt.figure(figsize=(6, 6))
    if len(image.shape) == 2:  # Grayscale image
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    left_img = cv2.imread('image_1.jpg')
    center_img = cv2.imread('image_2.jpg')
    right_img = cv2.imread('image_3.jpg')

    if left_img is None or center_img is None or right_img is None:
        print("Error: Could not load images. Check the file paths.")
    else:
        print("[INFO] Running panorama stitching...")
        result_img = solution(left_img, center_img, right_img)
        print("[INFO] All images have been saved in the 'output_images' folder.")
