import cv2
import numpy as np
from sklearn.cluster import DBSCAN


def preprocess_img(img):
    """
    Helper function that preprocesses image files for line/edge detection
    :param img: numpy ndarray, image
    :return: numpy ndarray, preprocessed image
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Make image greyscale
    gray = cv2.bitwise_not(gray)  # Invert colors

    # Make the greyscale image binary
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, -2)

    return gray


def find_straight_lines(img):
    """
    Finds horizontal and vertical edges in image using the Hough transform.
    :param img: numpy ndarray, pixel values in a greyscale image.
    :return: tuple, list of end points for horizontal and vertical edges.
    """
    edges = preprocess_img(img)

    # Take the Sobel derivatives for x and y direction to get vertical and horizontal lines
    # horizontal = cv2.Sobel(edges, cv2.CV_8U, dx=0, dy=1, ksize=3)
    # vertical = cv2.Sobel(edges, cv2.CV_8U, dx=1, dy=0, ksize=3)

    # Detect straight lines in the edge map
    RHO = 1  # Distance resolution of the accumulator in pixels.
    THETA = np.pi / 2  # Angle resolution of the accumulator in radians.
    THRESHOLD = 150  # Accumulator threshold parameter.
    MIN_LINE_LENGTH = 20  # Minimum line length. Line segments shorter than this are rejected.
    MAX_LINE_GAP = 0  # Maximum allowed gap between points on the same line to link them.

    hough_args = (RHO, THETA, THRESHOLD, np.array([]), MIN_LINE_LENGTH, MAX_LINE_GAP)

    # Use Hough transform to find straight lines in the edge map
    # v_lines = cv2.HoughLinesP(vertical, *hough_args)
    # h_lines = cv2.HoughLinesP(horizontal, *hough_args)
    lines = cv2.HoughLinesP(edges, *hough_args)
    return lines


def draw_linesp(lines, img):
    """
    Draws lines on an overlay from list of distance and angle.
    :param lines: list, end points for lines
    :param img: numpy ndarray, image to overlay the lines
    :return: numpy ndarray, image overlay.
    """
    canvas = np.copy(img) * 0  # Initialize an empty canvas in the shape of img to draw on

    for line in lines:
        x0, y0, x1, y1 = line[0]
        pt1 = (x0, y0)
        pt2 = (x1, y1)
        cv2.line(canvas, pt1, pt2, (0, 0, 255), 1, 8)

    return canvas


def draw_lines(lines, img):
    """
    Draws lines on an overlay from list of theta and rho.
    :param lines: List, lines in polar coordinate form.
    :param img: numpy ndarray, image the lines will be overlayed on.
    :return: numpy ndarray, image overlay.
    """
    canvas = np.copy(img) * 0  # Initialize an empty canvas in the shape of img to draw on

    for line in lines:
        rho = line[0][0]
        theta = line[0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        # Stretch the line through the whole image
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
        cv2.line(canvas, pt1, pt2, (0, 0, 255), 1, 8)

    return canvas


def find_orthogonal_lines(img):
    """
    Finds horizontal and vertical lines in an image.
    :param img: numpy ndarray, image
    :return: (numpy ndarray, numpy ndarray), contains horizontal and vertical line maps respectively.
    """
    gray = preprocess_img(img)

    # Initialize canvases to draw lines to
    horizontal = np.copy(gray)
    vertical = np.copy(gray)

    # Get size of the x axis
    cols = horizontal.shape[1]
    h_size = cols // 30

    # Create structure element for extracting horizontal lines
    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (h_size, 1))

    # Apply morphological operations
    horizontal = cv2.erode(horizontal, horizontal_structure)
    horizontal = cv2.dilate(horizontal, horizontal_structure)

    # Get size of the y axis
    rows = vertical.shape[0]
    v_size = rows // 30

    # Create structure element for extracting horizontal lines
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))

    # Apply morphological operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    # Dilate lines to catch external corners
    v_kernel = np.array([[2, 2], [2, 2]]).astype(np.uint8)
    h_kernel = np.array([[2, 2], [2, 2]]).astype(np.uint8)
    vertical = cv2.dilate(vertical, v_kernel)
    horizontal = cv2.dilate(horizontal, h_kernel)
    return horizontal, vertical


def find_cross(horizontal, vertical):
    """
    Finds crossing points of horizontal and vertical lines. Averages close points so that each crossing point
    should be represented by a single point.
    :param horizontal: numpy ndarray, map of horizontal lines.
    :param vertical: numpy ndarray, map of horizontal lines.
    :return: list, crossing points.
    """
    crosses = np.where(horizontal * vertical > 0, 1, 0).astype('uint8')
    y, x = np.nonzero(crosses)
    points = np.concatenate([x[:, np.newaxis], y[:, np.newaxis]], axis=1)

    # Group points and find the average
    clusters = group_points(points)
    avg = []
    for cluster in clusters:
        avg_x = int(np.average(cluster[:,0]))
        avg_y = int(np.average(cluster[:,1]))
        avg.append((avg_x, avg_y))
    return avg


def group_points(points, dist=10):
    """
    Clusters set of points
    :param points: numpy ndarray, array of points in the format (x, y)
    :param dist: int, maximum distance between points in a single cluster
    :return: list, list of numpy arrays containing points, each array represents a cluster.
    """
    db = DBSCAN(eps=dist, min_samples=1).fit(points)
    labels = db.labels_     # group labels for each point
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # number of clusters
    clusters = [points[labels == i] for i in range(n_clusters)]  # list of clusters
    return clusters


def draw_dots(dots, img):
    """
    Helper function draws circles on image.
    :param dots: list, center points for images.
    :param img: numpy ndarray, image the circles will be overlayed on..
    :return: numpy ndarray, image overlay.
    """
    canvas = np.copy(img) * 0

    for dot in dots:
        cv2.circle(canvas, (dot[0], dot[1]), 1, (0, 0, 255), 1, 1)

    return canvas


def find_rectangles(crossing_points, line_map):
    """
    Finds smallest rectangles in a set of crossing points
    :param crossing_points: list, crossing points
    :param line_map: numpy ndarray, image of lines considered valid
    :return: list, rectangle corner points.
    """
    rectangles = []

    # Sort crossing points from top to bottom, left to right.
    raw_points = np.array(sorted(crossing_points, key=lambda pt: (pt[1], pt[0])))

    for i, point in enumerate(raw_points):
        # Remove the current leftmost point from the set
        points = np.delete(raw_points, i, axis=0)

        # Get points that are on the same x and y coordinates
        x_points = points[points[:, 0] == point[0]]
        y_points = points[points[:, 1] == point[1]]

        for x_point in x_points:

            # Get pixels between points and replace actual pixel values with binary
            h_edge = line_map[point[1]:x_point[1], point[0]]
            h_zero_edge = h_edge != 0

            # Check if there is any empty pixels in the line
            if h_zero_edge.all():

                for y_point in y_points:
                    w_edge = line_map[point[1], point[0]:y_point[0]]
                    w_zero_edge = w_edge != 0

                    if w_zero_edge.all():
                        btm_right_point = (y_point[0], x_point[1])

                        # Check if there is continuous line from each corner to the bottom right point
                        btm_edge_x = line_map[x_point[1], y_point[0]: x_point[0]] != 0
                        btm_edge_y = line_map[x_point[1]:y_point[1], y_point[0]] != 0

                        if btm_right_point in points and btm_edge_x.all() and btm_edge_y.all():
                            rectangles.append((tuple(point), tuple(btm_right_point), tuple(x_point), tuple(y_point)))

    return rectangles


def group_rectangles(input_rectangles):
    """
    Groups rectangles into complete tables using bfs. Notice this method is a bit slow at the moment
    because neighbours are unknown and have to be checked from the complete set.
    :param input_rectangles: list, rectangle corner points.
    :return: list, sets of points which belong to the same table.
    """
    tables = []
    rectangles = input_rectangles.copy()
    while rectangles:
        rectangle = rectangles.pop()
        table = {rectangle}
        queue = [rectangle]
        while queue:
            cell = queue.pop()
            points = set(cell)

            for i, other_rectangle in enumerate(rectangles):
                other_points = set(other_rectangle)
                if other_points.intersection(points) and not(other_points in table):
                    neighbour = rectangles.pop(i)
                    table.add(neighbour)
                    queue.append(neighbour)

        tables.append(table)

    return tables


def find_point_neighbours(rectangles):
    # TODO might have to make a method that builds hashed neighbourhoods for each corner point if group_rectangles
    #  proves to be too slow for large files.

    pass


def draw_rectangles(rectangles, img, color=(0, 0, 255)):
    """
    Helper function, draws rectangles.
    :param rectangles: list, contains tuples of rectangle top left corner and bottom right corner coordinates.
    :param img: numpy ndarray, image to overlay.
    :param color: tuple, BGR color
    :return: numpy ndarray, overlay with rectangles drawn on it.
    """
    canvas = np.copy(img) * 0
    for rect in rectangles:
        x1, y1 = rect[0][0], rect[0][1]
        x2, y2 = rect[1][0], rect[1][1]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    return canvas


def find_tables(img_path):
    """
    Finds tables in an image file.
    :param img_path: string, path to the image file
    :return: list, tables with cell coordinates for each table found.
    """
    # Load image.
    load_img = cv2.imread(img_path)
    # Find orthogonal lines.
    h, v = find_orthogonal_lines(load_img)
    # Find crossing points for those lines.
    pts = find_cross(h, v)
    # Find cells. h+v is to give complete line map to the function i.e. both horizontal and vertical lines.
    r = find_rectangles(pts, h + v)
    # Finally group cells into tables.
    tables = group_rectangles(r)
    return tables


def demo():
    # img_path = 'sample.png'
    img_path = input('Give name of image: ')
    color_list = [[255, 0, 0],
                  [255, 165, 0],
                  [255, 255, 0],
                  [0, 255, 0],
                  [0, 0, 255],
                  [75, 0, 130],
                  [238, 130, 238]]
    load_img = cv2.imread(img_path)
    tables = find_tables(img_path)
    pic = np.copy(load_img)
    for color, table in zip(color_list[::-1], tables):
        rect_map = draw_rectangles(table, pic, color)
        pic = cv2.addWeighted(pic, 0.8, rect_map, 1, 0)
    # dots = draw_dots(pts, load_img)

    cv2.imshow('pic', pic)
    cv2.waitKey()


if __name__ == '__main__':
    demo()
