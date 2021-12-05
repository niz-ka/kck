import sys
import cv2
import numpy as np
import utils
import config
from classes import Note, Staff, StaffLine

from score import filter, io, transform


def __configure__logger__():
    import yaml
    from logging.config import dictConfig
    from pathlib import Path

    path = Path(__file__).parent.joinpath("logger.config.yaml").resolve()
    with open(path) as config_file:
        config = yaml.safe_load(config_file.read())
        dictConfig(config)


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

    io.show_image('Detected notes parts', img)


def __draw_rectangles__(title, img, rects, thickness = 1):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    for x, y, w, h in rects:
        cv2.rectangle(img, (x, y, w, h), (144, 169, 85), thickness)

    io.show_image(title, img)
    return img


######
__configure__logger__()

filename = sys.argv[1]

img = io.import_image('img/' + filename, 1200)
if img is False:
    exit(-1)

gray = filter.desaturate(img)
io.save_image(config.OUTPUT_DIR + filename + '_1_desaturated.png', gray)
io.show_image(filename + ' | Desaturated', gray)

binary = filter.binarize(gray)
io.save_image(config.OUTPUT_DIR + filename + '_2_binarized.png', binary)
io.show_image(filename + ' | Binarized', binary)

rotation_angle = transform.detect_rotation_angle(binary)
straight = transform.rotate(binary, rotation_angle)
io.save_image(config.OUTPUT_DIR + filename + '_3_straightened.png', straight)
io.show_image(filename + ' | Straightened', straight)

horizontal_lines = filter.detect_horizontal_lines(straight)
io.save_image(config.OUTPUT_DIR + filename + '_4_horizontal_lines.png', horizontal_lines)
io.show_image(filename + ' | Horizontal Lines', horizontal_lines)

vertical_lines = filter.detect_vertical_lines(straight)
io.save_image(config.OUTPUT_DIR + filename + '_5_vertical_lines.png', vertical_lines)
io.show_image(filename + ' | Vertical Lines', vertical_lines)

erased = filter.remove_lines(straight, horizontal_lines, vertical_lines)
io.save_image(config.OUTPUT_DIR + filename + '_6_erased.png', erased)
io.show_image(filename + ' | Erased', erased)

notes_rectangles = filter.find_bounding_rectangles(erased, min_area = 150, max_area = 5000)
detected_objects = __draw_rectangles__('Detected objects', straight, notes_rectangles)
io.save_image('detected_objects.png', detected_objects)

staff_lines_rectangles = filter.find_bounding_rectangles(np.invert(horizontal_lines), min_area = 500, max_area = 400000)
detected_stafflines = __draw_rectangles__('Detected staff lines', straight, staff_lines_rectangles, -1)
io.save_image('detected_stafflines.png', detected_stafflines)

staves = get_staves(staff_lines_rectangles, np.invert(horizontal_lines))
notes = get_notes(notes_rectangles, erased)

notes = classify(staves, notes)
draw_result(straight, notes)
__draw_note_parts__(straight, notes)
######
