import cv2
import glob
import os
import numpy as np

class PanaromaStitcher:
    def __init__(self):
        """
        Initializes the PanaromaStitcher class.
        This class helps stitch multiple images together into a beautiful panorama.
        """
        pass

    def make_panaroma_for_images_in(self, path):
        """
        This method takes a directory path and stitches the images found in that directory into a single panorama.

        :param path: str
            The folder where your images are stored.

        :return: tuple
            - stitched_image: The final panorama image created by stitching.
            - homography_matrix_list: A list of transformation matrices that show how each image relates to the next.
        """
        # Prepare to look for images in the provided path
        imf = path
        # Grab all the image files from the directory and sort them
        all_images = sorted(glob.glob(imf + os.sep + '*'))
        print(f'Found {len(all_images)} Images for stitching.')

        # Load images into a list using OpenCV, ready for processing
        images = [cv2.imread(img_path) for img_path in all_images]

        # Check if we actually found any images
        if len(images) == 0:
            print("It seems there are no images in the specified directory.")
            return None, []  # Return None and an empty list if no images were found

        # Create an OpenCV stitcher object to help us combine the images
        stitcher = cv2.Stitcher_create()

        # Try to stitch the images together and see what we get
        status, stitched_image = stitcher.stitch(images)

        # Did the stitching go smoothly?
        if status != cv2.Stitcher_OK:
            print(f"There was an error during stitching. OpenCV Stitcher returned status code: {status}")
            return None, []  # If it didn't work, return None and an empty list

        # Now let's compute the homography matrices for the images
        homography_matrix_list = self.compute_homographies(images)

        # Return the stitched image along with the list of homography matrices
        return stitched_image, homography_matrix_list

    def compute_homographies(self, images):
        """
        This method calculates the homography matrices between pairs of consecutive images using feature matching.

        :param images: list
            A list of images for which we want to find the homography matrices.

        :return: list
            A collection of homography matrices that describe how each image aligns with the next.
        """
        # We’ll store our homography matrices here
        homographies = []

        # Let’s use ORB (Oriented FAST and Rotated BRIEF) to find interesting points in our images
        orb = cv2.ORB_create()

        # Loop through each pair of consecutive images
        for i in range(len(images) - 1):
            # Detect keypoints and compute their descriptors for the first image
            kp1, des1 = orb.detectAndCompute(images[i], None)
            # Do the same for the second image
            kp2, des2 = orb.detectAndCompute(images[i + 1], None)

            # Create a Brute Force Matcher to find the best matches between the keypoints
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            # Match the descriptors from both images
            matches = bf.match(des1, des2)

            # Sort the matches based on how good they are (distance)
            matches = sorted(matches, key=lambda x: x.distance)

            # Extract the locations of the good matches
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Compute the homography matrix that relates the two images
            H = self.estimate_homography(src_pts, dst_pts)
            homographies.append(H)  # Add this matrix to our list

        # Return the list of homography matrices for further processing
        return homographies

    def estimate_homography(self, src_pts, dst_pts):
        """
        This method estimates the homography matrix based on the source and destination points.

        :param src_pts: array
            Source points from the first image.
        :param dst_pts: array
            Destination points from the second image.

        :return: np.array
            The estimated homography matrix.
        """
        # Prepare the matrix for the Direct Linear Transformation (DLT)
        A = []

        for i in range(len(src_pts)):
            x1, y1 = src_pts[i][0]
            x2, y2 = dst_pts[i][0]
            A.append([-x1, -y1, -1, 0, 0, 0, x2 * x1, x2 * y1, x2])
            A.append([0, 0, 0, -x1, -y1, -1, y2 * x1, y2 * y1, y2])

        A = np.array(A)

        # Apply Singular Value Decomposition (SVD)
        _, _, VT = np.linalg.svd(A)
        H = VT[-1].reshape(3, 3)

        return H

# Usage Example
# if __name__ == "__main__":
#     stitcher = PanaromaStitcher()
#     path_to_images = "/home/shruti/assignment-03/ES666-Assignment3/Images"  
#     stitched_image, homography_matrices = stitcher.make_panaroma_for_images_in(path_to_images)

#     # Save the stitched image to the results folder
#     if stitched_image is not None:
#         output_path = './results/stitched_image.jpg'
#         cv2.imwrite(output_path, stitched_image)
#         print(f'Successfully saved the stitched image to {output_path}')
#     else:
#         print('Stitching was unsuccessful. Check the input images and try again.')
