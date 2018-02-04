import numpy

class Box:
    def __init__(self, y, x, height, width, label, score):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.label = label
        self.score = score

    def __str__(self):
        return "y: {}, x:{}, width: {}, height: {}, label: {}, score: {}".format(
            self.y, self.x, self.width, self.height, self.label, self.score)

    def get_corners(self):
        """topleft, topright, botleft, botright"""
        return [(self.y, self.x), (self.y, self.x+self.width), 
                (self.y+self.height, self.x), (self.y+self.height, self.x+self.width)]

    def get_topleft(self):
        return (self.y, self.x)

    def get_topright(self):
        return (self.y, self.x+self.width)

    def get_botleft(self):
        return (self.y+self.height, self.x)

    def get_botright(self):
        return (self.y+self.height, self.x+self.width)
    

def parse_tf_output(frame_shape, boxes, scores, classes, threshold = 0.5):

    parsed_boxes = [] 

    (r,c,_) = frame_shape
    for i in range(len(boxes[0])):
        if scores[0][i] > threshold:
            topleft_row = int(r*boxes[0][i][0])
            topleft_col = int(c*boxes[0][i][1])
            height = int(r*boxes[0][i][2] - r*boxes[0][i][0])
            width = int(c*boxes[0][i][3] - c*boxes[0][i][1])
            
            parsed_boxes.append(Box(topleft_row, topleft_col, height, width, classes[0][i], scores[0][i] ))

    return parsed_boxes

