import cv2
import numpy as np
from scipy.ndimage import interpolation as inter
import utils
from classes import Note, Staff, StaffLine


def convert_to_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None)
    utils.save_and_show("gray.jpg", gray)
    return gray


def binarize(img, block_size, offset, filter):
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, offset)
    thresh = cv2.bilateralFilter(thresh, *filter)
    utils.save_and_show("thresh.jpg", thresh)
    return thresh


def straighten(bin_img, delta, limit):
    def straighten_helper(arr, angle):
        data = inter.rotate(arr, angle, reshape = False, order = 0)
        hist = np.sum(data, axis = 1, dtype = float)
        score = np.sum((hist[1:] - hist[:-1])**2, dtype = float)
        return score

    bin_img = np.invert(bin_img)
    angles = np.arange(-limit, limit + delta, delta)
    scores = []
    for angle in angles:
        score = straighten_helper(bin_img, angle)
        scores.append(score)
    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]
    print('Best angle:', best_angle)
    data = inter.rotate(bin_img, best_angle, reshape = False, order = 0)
    img = np.invert(np.array(data).astype(np.uint8))

    return img


# Find horizontal lines (staff lines)
def detect_horizontal_lines(img):
    img = np.invert(img)

    horizontal_size = int(img.shape[1] / 30)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    detected_lines = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations = 1)

    utils.save_and_show("horizontal.jpg", detected_lines)
    return detected_lines


# Find vertical lines (helpful in notes repairing)
def detect_vertical_lines(img):
    img = np.invert(img)

    vertical_size = int(img.shape[0] / 30)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    detected_lines = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations = 1)

    utils.save_and_show("vertical.jpg", detected_lines)
    return detected_lines


# Remove horizontal lines and repair through vertical lines
def remove_lines(img, horizontal_lines, vertical_lines):
    img = cv2.bitwise_or(img, horizontal_lines)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    img = np.invert(cv2.bitwise_or(np.invert(img), vertical_lines))

    utils.save_and_show("erased.jpg", img)
    return img


# Find rectangles of objects that have area between min_area and max_area
def find_bounding_rectangles(img, min_area, max_area):
    count, labels, stats = cv2.connectedComponentsWithStats(np.invert(img))[:3]
    areas = stats[:, 4]

    for label in range(1, count):
        if areas[label] > max_area or areas[label] < min_area:
            labels[labels == label] = 0
    labels[labels > 0] = 255

    stats = cv2.connectedComponentsWithStats(labels.astype(np.uint8))[2]

    return [[x, y, w, h] for x, y, w, h, *_ in stats[1:]]


def get_staves(staff_lines, img):
    lines = []
    staves = []

    if len(staff_lines) == 0 or len(staff_lines) == 1:
        print("No stafflines detected")
        exit()

    # Sort lines from top to bottom
    staff_lines.sort(key = lambda x: x[1])

    avg_space = 0.0
    for i in range(len(staff_lines) - 1):
        _, y1, _, _ = staff_lines[i]
        _, y2, _, _ = staff_lines[i + 1]
        avg_space += y2 - y1
    avg_space /= len(staff_lines) - 1

    # Create Staff objects
    staff_nr = 0
    line_nr = 0
    for i, line in zip(range(len(staff_lines)), staff_lines):
        x, y, w, h = line

        lines.append(StaffLine(x, y, w, h, line_nr, img[y:y + h, x:x + w]))
        line_nr += 1

        if i == len(staff_lines) - 1 or abs(lines[-1].y - staff_lines[i + 1][1]) > 3 * avg_space:
            staff_height = (lines[-1].y + lines[-1].height) - lines[0].y
            staff_width = max([line.width for line in lines])
            nparray = img[lines[0].y:lines[0].y + staff_height, lines[0].x:lines[0].x + staff_width]
            staves.append(Staff(lines[0].x, lines[0].y, staff_width, staff_height, lines, staff_nr, nparray))
            lines = []
            line_nr = 0
            staff_nr += 1

    print("Number of stafflines: ", sum([len(staff.staff_lines) for staff in staves]))
    print("Number of staves:", len(staves))

    return staves


def get_notes(notes_rectangles, img):
    notes = []

    # Sort notes from left to right
    notes_rectangles.sort(key = lambda x: x[0])

    # Create Note objects
    for note_rectangle in notes_rectangles:
        x, y, w, h = note_rectangle
        notes.append(Note(x, y, w, h, img[y:y + h, x:x + w]))

    return notes


def classify(staves, notes):
    # Find proper staff for each note
    notes_on_staff = []

    for note in notes:
        for staff in staves:
            if (staff.x <= note.x_center <= staff.x + staff.width) and (staff.y <= note.y_center <=
                                                                        staff.y + staff.height):
                note.staff = staff.order
                notes_on_staff.append(note)
                break

    if (len(notes_on_staff) == 0):
        print("No notes on staff detected")
        exit()

    print("Number of notes:", len(notes_on_staff))
    print("Number of non-notes:", len(notes) - len(notes_on_staff))

    notes = notes_on_staff

    # Find proper note type for each note

    # First note on staff is always clef
    # for i in range(len(staves)):
    #     # Common g-clef is always higher than staff
    #     if (notes[i].height > staves[notes[i].staff].height):
    #         notes[i].type = "wiolin"
    #     else:
    #         notes[i].type = "bas"

    # # Other
    # for note in notes:
    #     if(note.type is not None): continue
    #     assert staves[note.staff].order == note.staff

    #     half_height = note.height // 2
    #     half_width = note.width // 2

    #     # "1" has the size of space between staff lines
    #     if note.height < 1.5 * staves[note.staff].avg_staffline_space:
    #         note.type = "1"
    #     # "1/2" has many white pixels in bottom-left corner
    #     elif np.count_nonzero(note.nparray[half_height:, :half_width]) / (half_height * half_width) > 0.8:
    #         note.type = "1/2"
    #     # "1/8" has many white pixels in bottom-right corner
    #     elif np.count_nonzero(note.nparray[half_height:, half_width:]) / (half_height * half_width) > 0.8:
    #         note.type = "1/8"
    #     # Otherwise it's "1/4"
    #     else:
    #         note.type = "1/4"

    # Find position on staff
    for note in notes:
        # It's known where clef is located
        if (note.type == "bas" or note.type == "wiolin"):
            note.position = "-"
        # Otherwise I'm looking for nearest staff line (distances between certain point in note and center of line)
        else:
            # For "1" I take y-center, for others point that is lower (0.7)
            k = 0.5 if note.type == "1" else 0.7

            distances_from_line = [(line.order, abs((note.y + k * note.height) - line.y_center))
                                   for line in staves[note.staff].staff_lines]
            distances_from_line.sort(key = lambda x: x[1])
            line_nr, min = distances_from_line[0]

            # Note crosses the line
            if (min < 0.3 * staves[note.staff].avg_staffline_space):
                note.position = str(line_nr + 1)
            # Note is between lines
            else:
                if len(distances_from_line) > 1:
                    note.position = "{}-{}".format(line_nr + 1, distances_from_line[1][0] + 1)

    # Check result
    flag_OK = True
    for note in notes:
        if (note.position is None or note.type is None or note.staff is None):
            flag_OK = False

    assert flag_OK
    return notes


def draw_result(img, notes):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for index, note in enumerate(notes):
        # Text label example "wiolin 2 (3)" means "g-clef is placed on second staff and third line of this staff"
        # text_label = "{} {} ({})".format(note.type, note.staff + 1, note.position)
        text_label = f"{index}"
        # if note.type == '@':
        #     continue
        if note.type == '1':
            color = (224, 217, 174)
        elif note.type == '1/2':
            color = (195, 160, 159)
        elif note.type == '1/4':
            color = (0, 86, 148)
        elif note.type == '1/8':
            color = (29, 176, 114)
        else:
            color = (29, 4, 106)

        cv2.rectangle(img, (note.x, note.y), (note.x + note.width, note.y + note.height), color, 2)
        cv2.putText(img,
                    text_label, (note.x, note.y),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.6, (0, 0, 255),
                    thickness = 1,
                    lineType = cv2.LINE_AA)

        print(f"{index:^4}", note)

    utils.save_and_show("result.jpg", img)


def __draw_note_parts__(img, notes: list[Note]):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for note in notes:
        if note.head is not None:
            head_x = note.x + note.head.x
            head_y = note.y + note.head.y
            cv2.rectangle(img, (head_x, head_y, note.head.width, note.head.height), (0, 255, 0), 1)
        if note.stem is not None:
            stem_x = note.x + note.stem.x
            stem_y = note.y + note.stem.y
            cv2.rectangle(img, (stem_x, stem_y, note.stem.width, note.stem.height), (255, 0, 0), 1)

    utils.save_and_show("parts.jpg", img)


######
img = utils.import_image(size = (1200, 500))
gray = convert_to_gray(img)
binary = binarize(gray, block_size = 51, offset = 10, filter = (9, 75, 75))
straight = straighten(binary, delta = 1, limit = 50)

horizontal_lines = detect_horizontal_lines(straight)
vertical_lines = detect_vertical_lines(straight)
erased = remove_lines(straight, horizontal_lines, vertical_lines)

notes_rectangles = find_bounding_rectangles(erased, min_area = 150, max_area = 5000)
staff_lines_rectangles = find_bounding_rectangles(np.invert(horizontal_lines), min_area = 500, max_area = 400000)

staves = get_staves(staff_lines_rectangles, np.invert(horizontal_lines))
notes = get_notes(notes_rectangles, erased)

notes = classify(staves, notes)
draw_result(straight, notes)
# __draw_note_parts__(straight, notes)
######
