import cv2

# open txt file lines to a list
def file_lines_to_list(path):
    with open(path) as f:
        content = f.readlines()
    content = [x.strip() for x in content] # remove whitespace characters like `\n` at the end of each line
    return content

#prediciton class
class Pred:
    def __init__(self, id,conf,left,top,right,bottom):
        self.id = id
        self.conf = conf
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    # calculate intersections with another prediction
    def calc_pred_intersection(self, pred):
        if not self.id == pred.id:
            return 0
        left = max(self.left, pred.left)
        right = min(self.right, pred.right)
        if left > right:
            return 0
        top = max(self.top, pred.top)
        bottom = min(self.bottom, pred.bottom)
        if top > bottom:
            return 0
        return (right - left) * (bottom - top)

def plot_preds(img, preds, clr=(255, 0, 255)):
    if isinstance(img, str):
        img = cv2.imread(img)
    for pred in preds:
        cv2.rectangle(img, (pred.left, pred.top), (pred.right, pred.bottom), clr)
        cv2.putText(img, pred.id, (pred.left, pred.top), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return img

#read predictions
def read_predictions(txt_file):
    lines_list = file_lines_to_list(txt_file)
    prediction = []
    for line in lines_list:
        id, conf, left, top, right, bottom = line.split(';')
        pred = Pred(id,float(conf),int(left),int(top),int(right),int(bottom))
        prediction.append(pred)
    return prediction

