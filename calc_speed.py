import numpy as np


BEGIN_LINES = [(0, 360), (0, 433), (0, 529), (0, 691), (0, 991)]
END_LINES = [(1920, 303), (1920, 365), (1920, 468), (1920, 624), (1920, 925)]

# parameters for calculating vehicle speed
MS2KMH = 3.6
VIRTUAL_LINE_DISTANCE = 24  # distance between VDLs
FPS = 30  # input video's FPS
WIDTH_AREAS = [73, 96, 162, 300]  # width of the areas


def find_distance_between_line_point(p3, p1, p2):
    """
    Find the perpendicular distance from the point `p3` to the line (`p1`, `p2`)
    Parameters
    ----------
    p3: point outside the line
    p1: point inside the line
    p2: point inside the line
    Returns
    -------
        distance
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    return np.abs(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)


def check_line(p1, p2, w):
    """
    This is a function to check where the point w is based on a straight line.
    Args:
        p1: left point on a straight line
        p2: right point on a straight line
        w: point you want to know where you are in a straight line
    Returns:
        x |- x > 0: w point is above a straight line
          |- x = 0: w point is on a straight line
          |- x < 0: w point is below a straight line
    """
    x = (p2[0] - p1[0]) * (w[1] - p1[1]) - (w[0] - p1[0]) * (p2[1] - p1[1])
    return x


def define_point_in_area(point, begin_lines, end_lines):
    """
    Determine area which point is in
    Parameters
    ----------
    point: (x, y)
    begin_lines: left point on a straight line
    end_lines: right point on a straight line
    Returns
    -------
    area
    """
    num_lines = len(begin_lines)
    up_val = check_line(begin_lines[0], end_lines[0], point)
    bottom_val = check_line(begin_lines[-1], end_lines[-1], point)
    # if point outside limited area
    if up_val <= 0 or bottom_val >= 0:
        return -1
    for idx in range(num_lines - 1):
        up_val = check_line(begin_lines[idx], end_lines[idx], point)
        bottom_val = check_line(begin_lines[idx + 1], end_lines[idx + 1], point)
        # if point inside area limited by 2 nearest lines
        if up_val >= 0 >= bottom_val:
            return idx


def estimate_distance(previous_coords, current_coords, VIRTUAL_LINE_DISTANCE, width_area):
    """
    Estimate actual distance of object in specific area
    Parameters
    ----------
    previous_coords: previous position of object
    current_coords: current position of object
    VIRTUAL_LINE_DISTANCE: actual distance between two virtual distance lines
    width_area: distances of area limited by two lines
    Returns
    -------
    distance
    """
    dist_bbox_move = ((previous_coords[0] - current_coords[0]) ** 2 + (
            previous_coords[1] - current_coords[1]) ** 2) ** 0.5
    actual_dist = dist_bbox_move / width_area * VIRTUAL_LINE_DISTANCE
    return actual_dist


def calcSpeed(track, bbox, frame_idx, FPS):
    """
    This is a function to calculate vehicle speed  when vehicles pass the two VDL line((luLine, ldLine) or (ruLine, rdLine))
    Returns:
        track
    Parameters
    ----------
    track: Track class(Deep Sort reference code: https://github.com/nwojke/deep_sort/tree/master/deep_sort)
    * track.py has been updated to match our code
    bbox: bounding box(bbox) obtain from Deep Sort. bbox format is (min x, min y, max x, max y).
    frame_idx: frame number of an input video
    FPS: frames per second
    """

    # We use bottom center point to calculate vehicle speed
    bottom_center = (int(((bbox[0]) + (bbox[2])) / 2), int(bbox[3]))
    area = define_point_in_area(bottom_center, BEGIN_LINES, END_LINES)

    if area < 0:
        track.speed_update = False
    else:
        track.speed_update = True
    # Save the time when object in the area
    if track.speed_update:
        if not track.begin_count_time:  # if not count beginning time
            if track.area != area:
                track.time_passing_vline_start = frame_idx
                track.begin_count_time = True
                track.area = area
                track.coord_obj = bottom_center
        else:  # counted beginning time
            # update speed of object in the same area after half of video frame
            # Calculate the time that passed the two lines
            track.time_passing_vline_end = frame_idx
            num_frames = track.time_passing_vline_end - track.time_passing_vline_start
            if track.area == area:
                half_frames = round(FPS / 2)
                # after half (FPS) frame object update speed once in each area
                if num_frames % half_frames == 0:
                    actual_distance = estimate_distance(track.coord_obj, bottom_center, VIRTUAL_LINE_DISTANCE,
                                                        WIDTH_AREAS[track.area])
                    track.speed_time = num_frames * 1 / FPS
                    track.speed = actual_distance / track.speed_time * MS2KMH
            else:  # object move to the different area
                track.begin_count_time = False
                track.speed_update = False
                actual_distance = estimate_distance(track.coord_obj, bottom_center, VIRTUAL_LINE_DISTANCE,
                                                    WIDTH_AREAS[track.area])
                track.speed_time = num_frames * 1 / FPS
                track.speed = actual_distance / track.speed_time * MS2KMH
                track.coord_obj = bottom_center  # update coordinates when object moves different area
                track.area = area

    return track
