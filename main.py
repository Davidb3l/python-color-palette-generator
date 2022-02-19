import cv2
import numpy as np
from sklearn.cluster import KMeans
# from vlc_player import Player
import sys
import os
import argparse
from datetime import datetime

# Global variables
framesTaken = 0
framesTakenList = []
framesSpecified = 0
clusters = 5
margin = 5
borderSize = 60
offset = 2


if __name__=='__main__':
    parser=argparse.ArgumentParser(description="Scene Extractor")

    parser.add_argument('-vid',
                        '--video',
                        help="File path to an MP4 video",
                        type=str)

    parser.add_argument('-fs',
                        '--frames',
                        help="Frame skip to extract scenes",
                        type=float)

    parser.add_argument('-f',
                        '--frame',
                        help="Specific frame to extract, in HH:MM:SS format",
                        type=str)

    # Courtesy of https://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/
    def centroid_histogram(clt):
        # grab the number of different clusters and create a histogram
        # based on the number of pixels assigned to each cluster
        numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        (hist, _) = np.histogram(clt.labels_, bins=numLabels)
        # normalize the histogram, such that it sums to one
        hist = hist.astype("float")
        hist /= hist.sum()
        # return the histogram
        return hist


    # Courtesy of https://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/
    def plot_colors(hist, centroids):
        # initialize the bar chart representing the relative frequency
        # of each of the colors
        bar = np.zeros((50, 300, 3), dtype="uint8")
        startX = 0

        # Sort the centroids to form a gradient color look
        centroids = sorted(centroids, key=lambda x: sum(x))

        # loop over the percentage of each cluster and the color of
        # each cluster
        for (percent, color) in zip(hist, centroids[offset:]):
            # plot the relative percentage of each cluster
            # endX = startX + (percent * 300)

            # Instead of plotting the relative percentage,
            # we will make a n=clusters number of color rectangles
            # we will also seperate them by a margin
            new_length = 300 - margin * (clusters - 1)
            endX = startX + new_length/clusters
            cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                        color.astype("uint8").tolist(), -1)
            cv2.rectangle(bar, (int(endX), 0), (int(endX + margin), 50),
                        (255, 255, 255), -1)
            startX = endX + margin

        # return the bar chart
        return bar


    # A helper function to resize images
    def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]
        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))
        resized = cv2.resize(image, dim, interpolation=inter)
        return resized

    def color_map(img):
        image_copy = image_resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width=400)

        # Since the K-means algorithm we're about to do,
        # is very labour intensive, we will do it on a smaller image copy
        # This will not affect the quality of the algorithm
        pixelImage = image_copy.reshape((image_copy.shape[0] * image_copy.shape[1], 3))

        # We use the sklearn K-Means algorithm to find the color histogram
        # from our small size image copy
        clt = KMeans(n_clusters=clusters+offset)
        clt.fit(pixelImage)

        # build a histogram of clusters and then create a figure
        # representing the number of pixels labeled to each color
        hist = centroid_histogram(clt)

        # Let's plot the retrieved colors. See the plot_colors function
        # for more details
        bar = plot_colors(hist, clt.cluster_centers_)

        # Resize the color bar to be even width with the video frame
        barImage = image_resize(
            cv2.cvtColor(bar, cv2.COLOR_RGB2BGR),
            width=int(img.shape[0]))
        return barImage


    args=parser.parse_args()

    cap = cv2.VideoCapture(args.video) # open video for capture

    success, image = cap.read()

    if args.frame != "":
        # Scroll to the specific frame
       
        timestamp = datetime.strptime(args.frame, '%H:%M:%S').time()
        seconds = (timestamp.hour * 60 + timestamp.minute) * 60 + timestamp.second

        fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count/fps

        # print(duration, seconds)

        if duration < seconds:
            raise Exception("Cant extract frame past video length")
        
        frame_count = 0
        # Scroll to the specific frame
        while(cap.isOpened()):
            frame_count += 1
            frame_exists, curr_frame = cap.read()
            
            if frame_count / fps > seconds:  # specific millisecond 
                cv2.imwrite("extracted_frame.png", curr_frame)
                image = color_map(curr_frame)
                cv2.imwrite("color_map.png", image)
                break
    elif args.frames != 0:
        frame_count = 0
        frames = []
        while(cap.isOpened()):
            frame_count += 1
            frame_exists, curr_frame = cap.read()
            
            if frame_count % args.frames == 0:  # specific millisecond 
                image = color_map(curr_frame)
                frames.append(image)
        


    # while success:
    #     if counter % args.frames

    #     cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
    #     success,image = vidcap.read()
    #     print('Read a new frame: ', success)
    #     count += 1

    






# # Initialize the QtGUI Application instance
# app = QtGui.QApplication(sys.argv)

# # Initialize our custom VLC Player instance
# vlc = Custom_VLC_Player()
# vlc.show()
# # Let's change the size of the window. We will make it 660px by 530px
# vlc.resize(660, 530)

# if sys.argv[1:]:
#     vlc.OpenFile(sys.argv[1])
# sys.exit(app.exec_())
