import cv2
import numpy as np

MIN_MATCHES = 20
detector = cv2.ORB_create(nfeatures=5000) # oriented Brief with the maximum number of features to retain

FLANN_INDEX_KDTREE = 1
index_parameters = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_parameters = dict(checks=100)
flann = cv2.FlannBasedMatcher(index_parameters, search_parameters) # matching features, Approximate Nearest Neighbors


def load_input():

    img = cv2.imread("Resources/camera_img.jpg")
    img_aug = cv2.imread("Resources/mask.jpg")

    img = cv2.resize(img, (300, 400), interpolation=cv2.INTER_AREA) # resampling using pixel area relation
    img_aug = cv2.resize(img_aug, (300, 400))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # finding the keypoints
    keypoints, descriptors = detector.detectAndCompute(img_gray, None)

    return img_gray, img_aug, keypoints, descriptors


def compute_matches(descriptors_input, descriptors_output):

    if (len(descriptors_output) != 0 and len(descriptors_input) != 0):

        matches = flann.knnMatch(np.asarray(descriptors_input, np.float32), np.asarray(descriptors_output, np.float32),
                                 k=2) #finds the k best matches for each descriptor

        good = [] # store all the good matches
        for m, n in matches:
            if m.distance < 0.69 * n.distance:  # It is viewed as a match only when closest/second closest is less than the threshold (0.69)
                                                # because it is assumed that the matching is one-to-one (ideally, it would be 0)
                good.append(m)
        return good
    else:
        return None


if __name__ == '__main__':

    # Getting information from the input image
    img, img_aug, input_keypoints, input_descriptors = load_input()

    camera = cv2.VideoCapture(0)
    ret, frame = camera.read()

    while (ret):
        ret, frame = camera.read()
        if (len(input_keypoints) < MIN_MATCHES):
            continue
        frame = cv2.resize(frame, (600, 450))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        output_keypoints, output_descriptors = detector.detectAndCompute(frame_gray, None)
        matches = compute_matches(input_descriptors, output_descriptors)
        if (matches != None):
            if (len(matches) > 10):

                # If enough matches are found, extract the locations of matched keypoints
                source_points = np.float32([input_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                destination_points= np.float32([output_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                # Finally find the homography matrix
                M, mask = cv2.findHomography(source_points, destination_points, cv2.RANSAC, 5.0)

                points = np.float32([[0, 0], [0, 399], [299, 399], [299, 0]]).reshape(-1, 1, 2)
                destination = cv2.perspectiveTransform(points, M)
                M_aug = cv2.warpPerspective(img_aug, M, (600, 450))

                # getting the frame ready for addition operation with Mask Image
                frame2 = cv2.fillConvexPoly(frame, destination.astype(int), 0) #draw a filled convex polygon
                Final = frame2 + M_aug

                output_final = cv2.polylines(frame,[np.int32(destination)],True,255,3, cv2.LINE_AA)
                cv2.imshow('Final Output', Final)
            else:
                cv2.imshow('Final Output', frame)
        else:
            cv2.imshow('Final Output', frame)

        key = cv2.waitKey(15)
        if (key == 27): #Esc key to stop
            break

