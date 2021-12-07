import sys
import cv2
import numpy as np
import utils
import config
from score.classes import Note, Staff, Point

from score import filter as filters, io, transform


def __configure__logger__():
    import yaml
    from logging.config import dictConfig
    from pathlib import Path

    path = Path(__file__).parent.joinpath("logger.config.yaml").resolve()
    with open(path) as config_file:
        config = yaml.safe_load(config_file.read())
        dictConfig(config)


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
    # notes_on_staff = []

    notes = list(filter(lambda note: note.type not in ['@', '-'], notes))
    print(f"Notes: {len(notes)}")
    for note in notes:
        center = Point(note.x + note.head.x + note.head.width // 2, note.y + note.head.y + note.head.height // 2)

        for staff in staves:
            if staff.contains(center):
                note.staff = staff
                break
        else:
            note.staff = None

    notes = list(filter(lambda note: note.staff is not None, notes))
    print(f"Notes filtered: {len(notes)}")

    for note in notes:
        center = Point(note.x + note.head.x + note.head.width // 2, note.y + note.head.y + note.head.height // 2)
        note.position = note.staff.position(center)

    # Naïve clef assignment, assuming every staff with some objects starts with a clef.
    for staff in staves:
        objects = sorted(filter(lambda note: note.staff.order == staff.order, notes), key = lambda note: note.x)
        if len(objects) == 0:
            continue
        if objects[0].height > staff.height:
            objects[0].type = 'V'
        else:
            objects[0].type = 'B'

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
        elif note.type == 'V':
            color = (151, 134, 56)
        elif note.type == 'B':
            color = (95, 122, 224)
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

    return img


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
    return img


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

gray = filters.desaturate(img)
io.save_image(config.OUTPUT_DIR + filename + '_1_desaturated.png', gray)
io.show_image(filename + ' | Desaturated', gray)

binary = filters.binarize(gray)
io.save_image(config.OUTPUT_DIR + filename + '_2_binarized.png', binary)
io.show_image(filename + ' | Binarized', binary)

rotation_angle = transform.detect_rotation_angle(binary)
straight = transform.rotate_img(binary, rotation_angle)
# straight = transform.straighten_by_weight(binary, 1, 50)
io.save_image(config.OUTPUT_DIR + filename + '_3_straightened.png', straight)
io.show_image(filename + ' | Straightened', straight)

horizontal_lines = filters.detect_horizontal_lines(straight)
io.save_image(config.OUTPUT_DIR + filename + '_4_horizontal_lines.png', horizontal_lines)
io.show_image(filename + ' | Horizontal Lines', horizontal_lines)

vertical_lines = filters.detect_vertical_lines(straight)
io.save_image(config.OUTPUT_DIR + filename + '_5_vertical_lines.png', vertical_lines)
io.show_image(filename + ' | Vertical Lines', vertical_lines)

erased = filters.remove_lines(straight, horizontal_lines, vertical_lines)
io.save_image(config.OUTPUT_DIR + filename + '_6_erased.png', erased)
io.show_image(filename + ' | Erased', erased)

notes_rectangles = filters.find_bounding_rectangles(erased, min_area = 150, max_area = 5000)
detected_objects = __draw_rectangles__('Detected objects', straight, notes_rectangles)
io.save_image(config.OUTPUT_DIR + filename + '_7_detected_objects.png', detected_objects)

staff_lines_rectangles = filters.find_bounding_rectangles(np.invert(horizontal_lines),
                                                          min_area = 500,
                                                          max_area = 400000)
detected_stafflines = __draw_rectangles__('Detected staff lines', straight, staff_lines_rectangles, -1)
io.save_image(config.OUTPUT_DIR + filename + '_8_detected_stafflines.png', detected_stafflines)

staves = filters.get_staves(staff_lines_rectangles, np.invert(horizontal_lines))
staves_rects = [(staff.x_start, int(staff.y), staff.x_end - staff.x_start, int(staff.height)) for staff in staves]
staves_rects = __draw_rectangles__('Detected staves', straight, staves_rects, -1)
io.save_image(config.OUTPUT_DIR + filename + '_9_staves.png', staves_rects)

notes = get_notes(notes_rectangles, erased)

notes = classify(staves, notes)
final = draw_result(straight, notes)
io.save_image(config.OUTPUT_DIR + filename + '_9_final.png', final)
io.show_image(filename + ' | Final', final)
heads = __draw_note_parts__(straight, notes)
io.save_image(config.OUTPUT_DIR + filename + '_10_heads.png', heads)
######
