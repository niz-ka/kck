# Inheritance probably would be needed here ;)

class Staff:
    def __init__(self, x, y, width, height, staff_lines, order, nparray = None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.staff_lines = staff_lines
        self.order = order
        self.nparray = nparray
        self.y_center = None
        self.x_center = None
        self.avg_staffline_space = None
        self.__initialize()

    def __initialize(self):
        self.y_center = int((2*self.y + self.height) / 2)
        self.x_center = int((2*self.x + self.width) / 2)
        self.avg_staffline_space = self.height / 4


class StaffLine:
    def __init__(self, x, y, width, height, order, nparray = None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.order = order
        self.nparray = nparray
        self.y_center = None
        self.x_center = None
        self.__initialize()

    def __initialize(self):
        self.y_center = int((2*self.y + self.height) / 2)
        self.x_center = int((2*self.x + self.width) / 2)


class Note:
    def __init__(self, x, y, width, height, nparray = None, type = None, staff = None, position = None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.nparray = nparray

        self.type = type
        self.staff = staff
        self.position = position
        
        self.y_center = None
        self.x_center = None
        self.__initialize()
    
    def __initialize(self):
        self.y_center = int((2*self.y + self.height) / 2)
        self.x_center = int((2*self.x + self.width) / 2)
    
    def __str__(self):
        return "Nuta: {} | Wysokość: {} | Szerokość: {} | Nr pięciolinii: {} | Położenie {}".format(self.type, self.height, self.width, self.staff+1, self.position)
