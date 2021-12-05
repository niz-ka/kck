import numpy as np

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class HorizontalLine:
    def __init__(self, x_start, x_end, y) -> None:
        self.x_start = x_start
        self.x_end = x_end
        self.y = y

    def __str__(self) -> str:
        return f"HorizontalLine(start: {self.x_start}, end: {self.x_end}, y: {self.y})"


class HoughLine:
    def __init__(self, distance, angle) -> None:
        self.distance = distance
        self.angle = angle
        self.y = np.round(np.sin(angle) * distance + 1000 * np.cos(angle))

    def __str__(self) -> str:
        return f"Line(r: {self.distance}, theta: {self.angle}, y: {self.y})"


class Staff:
    def __init__(self, x_start, x_end, y, height, order):
        self.x_start = x_start
        self.x_end = x_end
        self.y = y
        self.height = height
        self.order = order
        self.y_center = y + (height // 2)
        self.line_span = height / 4

    def __str__(self) -> str:
        return f"Staff(y: {self.y}, height: {self.height}, order: {self.order}, space between: {self.line_span})"

    def contains(self, point) -> bool:
        # yapf:disable
        return (
            point.x >= self.x_start and
            point.x <= self.x_end and
            # Line span here is margin for notes just above or below bounding lines.
            point.y >= self.y - self.line_span and
            point.y <= self.y + self.height + self.line_span
        )
        # yapf:enable

    def position(self, point) -> str:
        positions = ['1', '1-2', '2', '2-3', '3', '3-4', '4', '4-5', '5']
        r = self.line_span / 2
        upper = self.y - r
        lower = self.y + r

        for position in positions:
            if point.y >= upper and point.y < lower:
                return position
            upper += r
            lower += r
        else:
            return '-'


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
            f"Nr pięciolinii: {self.staff.order + 1:^5} | "
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


# class Note:
#     class Head:
#         def __init__(self, x, y, width, height, type) -> None:
#             self.x = x
#             self.y = y
#             self.width = width
#             self.height = height
#             self.type = type
#             self.center = (x + width // 2, y + height // 2)

#     class Stem:
#         def __init__(self, x, y, width, height, type):
#             self.x = x
#             self.y = y
#             self.width = width
#             self.height = height
#             self.type = type

#     def __init__(self, x, y, width, height, nparray: np.ndarray):
#         self.x = x
#         self.y = y
#         self.width = width
#         self.height = height
#         self.nparray = nparray
#         self.y_center = y + (height // 2)
#         self.x_center = x + (width // 2)
#         self.staff = None
#         self.position = None
#         self.__detect_parts(nparray)
#         if self.head is None:
#             self.type = '-'
#         elif self.stem is None:
#             self.type = '1'
#         elif self.stem == 'quaver':
#             self.type == '1/8'
#         elif self.stem == 'whole' and self.head == 'open':
#             self.type = '1/2'
#         elif self.stem == 'whole' and self.head == 'closed':
#             self.type = '1/4'

#     def __str__(self):
#         # yapf: disable
#         return (
#             f"Nuta: {self.type:^7} | "
#             f"Wysokość: {self.height:^5} | "
#             f"Szerokość: {self.width:^5} | "
#             f"Nr pięciolinii: {self.staff + 1:^5} | "
#             f"Położenie {self.position:^7}"
#         )
#         # yapf: enable

#     def classify(self):
#         # head = self._find_head()

#         # # If head takes all space or note's aspect ratio is landscape
#         # # then it's probably a semibreve.
#         # if head.height == self.height or self.width >= self.height:
#         #     self.type = '1'
#         # # NOTE: Interesting case
#         # # If area covered by head is less than 50% then it's probably a minim.
#         # # But because quaver's stem takes horizontal space, its `head space` will also
#         # # have a lot of empty space, what leads us to quite low coverage of this area.
#         # # So we need to make additional check comparing horizontal halves of note's `head space`.
#         # elif head.coverage < 0.5:
#         #     half = head.width // 2
#         #     left_half = head.img[:half, :]
#         #     right_half = head.img[half:, :]
#         #     diff_threshold = head.width * head.height / 8

#         #     diff = np.sum(left_half) - np.sum(right_half)
#         #     if np.abs(diff) < diff_threshold:
#         #         self.type = '1/2'
#         #     else:
#         #         self.type = '1/8'
#         # # Otherwise it's probably a crotchet.
#         # else:
#         #     self.type = '1/4'

#         y = self.head.y + self.head.height // 2

#         self.position = self.staff.position(y)

#     def description(self) -> str:
#         if self.type is not None:
#             type = self.type
#         else:
#             type = "-"

#         if self.staff is not None:
#             staff = self.staff.order
#         else:
#             staff = "-"

#         if self.position is not None:
#             position = self.position
#         else:
#             position = "-"

#         return f"[{staff}]({position})|{type}|"

#     def __detect_parts(self, img):
#         # Characteristic feature of every considered note is it's build schema:
#         # note consists of head on the lower part and (except semibreve) stem
#         # on upper right part.
#         #
#         # Detection idea is simple:
#         # - find stem horizontal offset from left edge and cut if off
#         # - from rest of the image find note's head bound

#         def detect_stem(img, width, height):
#             img = img[:height, :width]
#             columns = [(i, sum(img[:, i])) for i in range(width)]

#             # Threshold above which we assume that column is part of main (vertical) stem.
#             STEM_THRESHOLD = height / 2
#             # Threshold above which we assume that column is part of additional (quaver) stem.
#             # Only for right offset detection.
#             VERTICAL_THRESHOLD = 5
#             # Threshold above which we assume that row is part of stem.
#             # Only for vertical offset detection.
#             HORIZONTAL_THRESHOLD = 5

#             # We assume that stem begins (from left) where STEM_THRESHOLD is exceeded.
#             possible_stem = list(filter(lambda x: x[1] > STEM_THRESHOLD, columns))

#             if len(possible_stem) == 0:
#                 # No stem was detected
#                 return None

#             left_offset = possible_stem[0][0]
#             # We need to remove head's part after detecting left offset.
#             columns = list(filter(lambda x: x[0] > left_offset, columns))

#             # Right offset will be the last element that exceeds THRESHOLD.
#             # Because STEM_THRESHOLD gives a lot higher requirements so this definitely exists.
#             right_offset = list(filter(lambda x: x[1] > VERTICAL_THRESHOLD, columns))[-1][0]

#             rows = [(i, sum(img[i, left_offset:right_offset])) for i in range(height)]
#             rows = list(filter(lambda x: x[1] > HORIZONTAL_THRESHOLD, rows))
#             upper_offset = rows[0][0]
#             lower_offset = rows[-1][0]

#             diff = max(rows) - min(rows)
#             if diff > HORIZONTAL_THRESHOLD:
#                 type = 'quaver'
#             else:
#                 type = 'whole'

#             width = right_offset - left_offset
#             height = lower_offset - upper_offset

#             return Note.Stem(left_offset, upper_offset, width, height, type)

#         def detect_head(img, width, height):
#             img = img[:height, :width]

#             THRESHOLD = 5

#             columns = [(i, sum(img[:, i])) for i in range(width)]
#             columns = list(filter(lambda x: x[1] > THRESHOLD, columns))

#             rows = [(i, sum(img[i, :])) for i in range(height)]
#             rows = list(filter(lambda x: x[1] > THRESHOLD, rows))

#             if len(columns) == 0 or len(rows) == 0:
#                 # No head detected
#                 return None

#             left_offset = columns[0][0]
#             right_offset = columns[-1][0]
#             upper_offset = rows[0][0]
#             lower_offset = rows[-1][0]

#             width = right_offset - left_offset
#             height = lower_offset - upper_offset

#             center_x = left_offset + (width // 2)
#             center_y = upper_offset + (height // 2)

#             x_center_sum = sum(img[center_y, :])
#             y_center_sum = sum(img[:, center_x])

#             x_threshold = width * 0.5
#             y_threshold = height * 0.5
#             if x_center_sum < x_threshold and y_center_sum < y_threshold:
#                 type = 'open'
#             else:
#                 type = 'closed'

#             return Note.Head(left_offset, upper_offset, width, height, type)

#         height, width, *_ = img.shape
#         img = np.invert(img) / 255.

#         self.stem = detect_stem(img, width, height)

#         if self.stem is not None:
#             width = self.stem.x
#         self.head = detect_head(img, width, height)