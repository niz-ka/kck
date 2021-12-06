import numpy as np


class Staff:
    def __init__(self, x, y, width, height, staff_lines, order, nparray = None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.staff_lines = staff_lines
        self.order = order
        self.nparray = nparray
        self.y_center = y + (self.height // 2)
        self.x_center = x + (self.width // 2)
        self.avg_staffline_space = height / 4


class StaffLine:
    def __init__(self, x, y, width, height, order, nparray = None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.order = order
        self.nparray = nparray
        self.y_center = y + (height // 2)
        self.x_center = x + (width // 2)


class Note:
    class Head:
        def __init__(self, x, y, width, height, type) -> None:
            self.x = x
            self.y = y
            self.width = width
            self.height = height
            self.type = type

    class Stem:
        def __init__(self, x, y, width, height, type):
            self.x = x
            self.y = y
            self.width = width
            self.height = height
            self.type = type

    def __init__(self, x, y, width, height, nparray: np.ndarray):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.nparray = nparray
        self.y_center = y + (height // 2)
        self.x_center = x + (width // 2)
        self.staff = None
        self.position = None
        self.__detect_parts(nparray)
        if self.head is None:
            self.type = '@'
        elif self.stem is None:
            self.type = '1'
        elif self.stem.type == 'quaver':
            self.type = '1/8'
        elif self.stem.type == 'whole' and self.head.type == 'open':
            self.type = '1/2'
        elif self.stem.type == 'whole' and self.head.type == 'closed':
            self.type = '1/4'
        else:
            self.type = '-'

    def __str__(self):
        if self.head is None:
            head_type = '-'
        else:
            head_type = self.head.type

        if self.stem is None:
            stem_type = '-'
        else:
            stem_type = self.stem.type

        # yapf: disable
        return (
            f"Nuta: {self.type:^7} | "
            f"Wysokość: {self.height:^5} | "
            f"Szerokość: {self.width:^5} | "
            f"Nr pięciolinii: {self.staff + 1:^5} | "
            f"Położenie {self.position:^7} | "
            f"Head {head_type:^8} | "
            f"Stem {stem_type:^10}"
        )
        # yapf: enable

    def __detect_parts(self, img):
        # Characteristic feature of every considered note is it's build schema:
        # note consists of head on the lower part and (except semibreve) stem
        # on upper right part.
        #
        # Detection idea is simple:
        # - find stem horizontal offset from left edge and cut if off
        # - from rest of the image find note's head bound

        def detect_stem(img, width, height):
            img = img[:height, :width]
            columns = [(i, sum(img[:, i])) for i in range(width)]

            # Threshold above which we assume that column is part of main (vertical) stem.
            STEM_THRESHOLD = height * 0.5
            # Threshold above which we assume that column is part of additional (quaver) stem.
            # Only for right offset detection.
            VERTICAL_THRESHOLD = 5
            # Threshold above which we assume that row is part of stem.
            # Only for vertical offset detection.
            HORIZONTAL_THRESHOLD = 3

            # We assume that stem begins (from left) where STEM_THRESHOLD is exceeded.
            possible_stem = list(filter(lambda x: x[1] > STEM_THRESHOLD, columns))

            if len(possible_stem) == 0:
                # No stem was detected
                return None

            left_offset = possible_stem[0][0]
            # We need to remove head's part after detecting left offset.
            columns = list(filter(lambda x: x[0] > left_offset, columns))

            # Right offset will be the last element that exceeds THRESHOLD.
            # Because STEM_THRESHOLD gives a lot higher requirements so this definitely exists.
            right_offset = list(filter(lambda x: x[1] > VERTICAL_THRESHOLD, columns))[-1][0]

            rows = [(i, sum(img[i, left_offset:right_offset])) for i in range(height)]
            rows = list(filter(lambda x: x[1] >= HORIZONTAL_THRESHOLD, rows))

            if len(rows) == 0:
                return None

            upper_offset = rows[0][0]
            lower_offset = rows[-1][0]

            max_stem_girth = max(rows, key = lambda x: x[1])[1]
            min_stem_girth = min(rows, key = lambda x: x[1])[1]
            diff = max_stem_girth - min_stem_girth
            if diff > 2 * min_stem_girth:
                type = 'quaver'
            else:
                type = 'whole'

            stem_width = right_offset - left_offset
            stem_height = lower_offset - upper_offset

            if stem_height != 0 and stem_width / stem_height > 0.9:
                # If aspect ratio is not portrait-like then it's probably a semibreve.
                return None

            if stem_width > 0.7 * width:
                # If "stem" width take over 70% of note's width then it's probably a semibreve.
                return None

            # if np.sum(img[upper_offset:lower_offset, left_offset:right_offset]) < 0.5 * width * height:
            #     # If stem filling is less than 50% it's probably a semibreve.
            #     return None

            return Note.Stem(left_offset, upper_offset, stem_width, stem_height, type)

        def detect_head(img, width, height):
            img = img[:height, :width]

            THRESHOLD = 5

            columns = [(i, sum(img[:, i])) for i in range(width)]
            columns = list(filter(lambda x: x[1] > THRESHOLD, columns))

            rows = [(i, sum(img[i, :])) for i in range(height)]
            rows = list(filter(lambda x: x[1] > THRESHOLD, rows))

            if len(columns) == 0 or len(rows) == 0:
                # No head detected
                return None

            left_offset = columns[0][0]
            right_offset = columns[-1][0]
            upper_offset = rows[0][0]
            lower_offset = rows[-1][0]

            width = right_offset - left_offset
            height = lower_offset - upper_offset

            center_x = left_offset + (width // 2)
            center_y = upper_offset + (height // 2)

            x_center_sum = sum(img[center_y, :])
            y_center_sum = sum(img[:, center_x])

            x_threshold = width * 0.5
            y_threshold = height * 0.5
            if x_center_sum < x_threshold and y_center_sum < y_threshold:
                type = 'open'
            else:
                type = 'closed'

            return Note.Head(left_offset, upper_offset, width, height, type)

        height, width, *_ = img.shape
        img = np.invert(img) / 255.

        self.stem = detect_stem(img, width, height)

        if self.stem is not None:
            width = self.stem.x
        self.head = detect_head(img, width, height)
