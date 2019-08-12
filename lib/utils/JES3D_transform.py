import numpy as np
from numpy.linalg import inv
from utils.JES3D_transform_utils import file_lines_to_list
import copy
def read_cam(str):
    cam = {}
    wrds=str.split(' ')
    if not len(wrds)==14:
        return cam
    vals = [float(w) for w in wrds]
    cam['id'] = vals[0]+1
    cam['f'] = vals[1]
    cam['R'] = np.reshape(vals[2:11],(3,3))
    cam['ori'] = np.transpose(cam['R'])
    cam['T'] = np.transpose(np.reshape(vals[11:14],(1,3)))
    cam['pos'] = np.matmul(-cam['ori'],cam['T'])
    cam['z']=np.reshape(cam['ori'][:,2],(3,1))
    return cam

def load_calib_file(calib_file):
    views = ['Left', 'Right', 'Top']
    file_lines = file_lines_to_list(calib_file)
    ncam = int(file_lines[0])
    if not ncam==3:
        print('Error: Invalid calibration file - number of cameras must be 3')

    cams = []
    pts = []
    for i in range(ncam):
        cam = read_cam(file_lines[i+1])
        if len(cam) == 0:
            print('Can not read camera from line {}'.format(i+1))
            return cams, pts
        cam['view'] = views[i]
        cams.append(cam)
    npts = int(file_lines[4])
    pts = np.zeros((npts,3))
    for i in range(npts):
        wrds = file_lines[i+5].split(' ')
        p = [float(x) for x in wrds]
        if not len(p) == 3:
            print('Can not parse point at line {}'.format(i+5))
            return cams, pts
        pts[i,:] = p
    return cams, pts

class JES3D_transform:
    def __init__(self, calib_file):
        self.cams, self.pts = load_calib_file(calib_file)
        self. imgsK = []
        self.W = 1280
        self.H=960
        self.estimated_hight = 0.05
        for i in range(3):
            self.imgsK.append(np.reshape([self.cams[i]['f'], 0 , (self.W-1)/2, 0, self.cams[i]['f'], (self.H-1)/2, 0, 0, 1],(3,3)))

    def rotate90(self,pt, direction=1):
        # rotating a point's cordinates by 90 degree (direction = +-1 )
        rot_pt = pt.copy()
        if direction>0:
            rot_pt[0] = self.W-pt[1]
            rot_pt[1] = pt[0]
        else:
            rot_pt[0] = pt[1]
            rot_pt[1] = self.H - pt[0]
        return rot_pt

    def point_dist_from_cam_for_z(self,R, T, p_n, z):
        # given a camera with R, T(pc=R * pw + T) and normalized point in the camera p_n, find
        # distance from the camera such that pw_z = z.
        a = np.matmul(np.transpose(R),T)
        b = np.matmul(np.transpose(R),p_n)
        p_dist = (z + a[2]) / b[2]
        return p_dist

    #transfering point from src_view to top view (src_view: 0-left, 1-right)
    def trans_pt(self, p_src_i, src_view):
        #rotation
        if src_view==0: #left
            p_src_i = self.rotate90(p_src_i,1)
        if src_view == 1:  # right
            p_src_i = self.rotate90(p_src_i,-1)

        p_src_i.append(1)
        p_src_n = np.matmul(inv(self.imgsK[src_view]),np.reshape(p_src_i,(3,1)))
        p_dist_c = self.point_dist_from_cam_for_z(self.cams[src_view]['R'], self.cams[src_view]['T'], p_src_n, self.estimated_hight)
        p_src_c = p_src_n * p_dist_c
        p_w = np.matmul(np.transpose(self.cams[src_view]['R']),(p_src_c-self.cams[src_view]['T']))
        p_dst_c = np.matmul(self.cams[2]['R'], p_w) + self.cams[2]['T']
        p_dst_n = p_dst_c / p_dst_c[2]
        p_dst = np.matmul(self.imgsK[2],p_dst_n)
        p_dst = np.transpose(p_dst[0: 2])+1

        return p_dst
    
    #transfering point from src_view to top view (src_view/trg_view: 0-left, 1-right,2-top)
    def trans_rot(self, p_src, src_view,trg_view):
        #rotation
        # if src_view==0: #left
        #     p_src_i = self.rotate90(p_src_i,1)
        # if src_view == 1:  # right
        #     p_src_i = self.rotate90(p_src_i,-1)
        p_src_i = copy.deepcopy(p_src)
        p_src_i.append(1)
        p_src_n = np.matmul(inv(self.imgsK[src_view]),np.reshape(p_src_i,(3,1)))
        p_dist_c = self.point_dist_from_cam_for_z(self.cams[src_view]['R'], self.cams[src_view]['T'], p_src_n, self.estimated_hight)
        p_src_c = p_src_n * p_dist_c
        p_w = np.matmul(np.transpose(self.cams[src_view]['R']),(p_src_c-self.cams[src_view]['T']))
        p_dst_c = np.matmul(self.cams[trg_view]['R'], p_w) + self.cams[trg_view]['T']
        p_dst_n = p_dst_c / p_dst_c[2]
        p_dst = np.matmul(self.imgsK[trg_view],p_dst_n)
        p_dst = np.transpose(p_dst[0: 2])+1
        # rotate 180 since the original rotation was clockwise, JS rotation counterclockwise
        p_dst = p_dst[0]
        if trg_view==0:
            p_dst = [self.W -p_dst[0]-1, self.H - p_dst[1]-1]
        return p_dst




    # transforming prediction from src_view to top view (src_view: 0-left, 1-right)
    def trans_pred(self, pred, src_view):
        lt = [pred.left, pred.top]
        new_lt = self.trans_pt(lt,src_view)
        rb = [pred.right, pred.bottom]
        new_rb = self.trans_pt(rb, src_view)
        left = int(min(new_lt[0,0],new_rb[0,0]))
        right = int(max(new_lt[0,0], new_rb[0,0]))
        top = int(min(new_lt[0,1], new_rb[0,1]))
        bottom = int(max(new_lt[0,1], new_rb[0,1]))
        pred.left = max(left,0)
        pred.top = max(top,0)
        pred.right = min(right,self.W)
        pred.bottom = min(bottom,self.H)
        return pred

    # transforming prediction from rotated src_view to top view (src_view: 0-left, 1-right)
    def trans_pred_rot(self, pred, src_view):
        lt = [pred.left, pred.top]
        new_lt = self.trans_pt(lt,src_view)
        rb = [pred.right, pred.bottom]
        new_rb = self.trans_pt(rb, src_view)
        left = int(min(new_lt[0,0],new_rb[0,0]))
        right = int(max(new_lt[0,0], new_rb[0,0]))
        top = int(min(new_lt[0,1], new_rb[0,1]))
        bottom = int(max(new_lt[0,1], new_rb[0,1]))
        
        pred.left = max(left,0)
        pred.top = max(top,0)
        pred.right = min(right,self.W)
        pred.bottom = min(bottom,self.H)
        return pred

