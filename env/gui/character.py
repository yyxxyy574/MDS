from .cell import Cell

POSE_DICT = {
    'right': 0,
    'back': 1,
    'left': 2,
    'front': 3,
}

ORI_DICT = {
    # [offset_x, offset_y, offset_pose]
    'up': [0, -1, 0],
    'left': [-1, 0, 1],
    'down': [0, 1, 2],
    'right': [1, 0, 3],
}

GENDER_DICT = {
    "female": 0,
    "male": 1,
}

COLOR_DICT = {
    "black": 0,
    "white": 1,
    "yellow": 2,
}

PROFESSION_DICT = {
    "artist": 0,
    "blue-collar": 1,
    "boss": 2,
    "chef": 3,
    "child": 4,
    "doctor": 5,
    "old": 6,
    "police": 7,
    "soldier": 8,
    "student": 9,
    "teacher": 10,
    "thief": 11,
    "unemployed": 12,
    "white-collar": 13,
    "none": 14
}

ANIMAL_DICT = {
    "chick": [0, 0],
    "chicken": [1, 0],
    "goose": [2, 0],
    "pig": [3, 0],
    "sheep": [4, 0],
    "skunk": [0, 1],
    "procupine": [1, 1],
    "boar": [2, 1],
    "fox": [3, 1],
    "wolf": [4, 1],
    "turtle": [0, 2],
    "frog": [1, 2],
    "toad": [2, 2],
    "crab": [3, 2],
    "cat": [4, 2],
}

ARROW_DICT = {
    'yellow': 0,
    'green': 1,
    'orange': 2,
    'blue': 3,
}

UI_DICT = {
    'virus': [[0, 0]],
    'ill': [[1, 0], [2, 0], [3, 0], [4, 0]],
    'food': [[5, 0], [6, 0], [0, 1], [1, 1]],
    'pregnancy': [[2, 1]],
    'baby': [[3, 1]],
    'boom': [[4, 1]],
    'gun': [[5, 1]],
    'noise': [[6, 1]],
    'alarm': [[0, 2]],
    'fairness': [[1, 2]],
    'death': [[2, 2]],
    'poison': [[3, 2], [4, 2]],
    'money': [[5, 2], [1, 5]],
    'blood': [[6, 2]],
    'loyalty': [[0, 3]],
    'inpurity': [[1, 3]],
    'disappointment': [[2, 3], [3, 3], [4, 3], [5, 3]],
    'anger': [[6, 3]],
    'report': [[0, 4]],
    'cheat': [[1, 4], [2, 4], [3, 4]],
    'resume': [[4, 4]],
    'trash': [[5, 4]],
    'complacent': [[6, 4], [0, 5]],
    'belief': [[2, 5], [3, 5], [4, 5], [5, 5], [6, 5]],
    'country': [[0, 6], [1, 6], [2, 6], [3, 6], [4, 6]],
    'pill': [[5, 6]],
    'shield': [[6, 6]]
}

class Character:
    def __init__(self, x, y,  width, height, species='human', ori='up', pose='front', profession='none', gender='none', color='yellow', arrow=None, ui=None):
        self.width = width
        self.height = height
        self.species = species
        self.pose = pose
        self.profession = profession
        self.gender = gender
        self.color = color
        self.arrow = arrow
        
        if self.species == 'human':
            if gender == 'none':
                self.pose_x = POSE_DICT[self.pose] * 4 + ORI_DICT[ori][2]
                self.pose_y =  180 + COLOR_DICT[self.color] * 2
            else:
                self.pose_x = POSE_DICT[self.pose] * 4 + ORI_DICT[ori][2]
                self.pose_y = PROFESSION_DICT[self.profession] * 12 + GENDER_DICT[self.gender] * 6 + COLOR_DICT[self.color] * 2
            self.lower_body = Cell(x, y, self.pose_x, self.pose_y + 1, self.width, self.height)
            self.upper_body = Cell(x + ORI_DICT[ori][0], y + ORI_DICT[ori][1], self.pose_x, self.pose_y, self.width, self.height)
            # for ui
            self.pose_y += 1
        else:
            self.pose_x = ANIMAL_DICT[self.species][0]
            self.pose_y = ANIMAL_DICT[self.species][1]
            self.body = Cell(x, y, self.pose_x, self.pose_y, self.width, self.height)

        if arrow is not None:
            pose_arrow_x = ARROW_DICT[self.arrow]
            pose_arrow_y = int(ori != 'up')
            self.arrow = Cell(x, y - 1, pose_arrow_x, pose_arrow_y, self.width, self.height)