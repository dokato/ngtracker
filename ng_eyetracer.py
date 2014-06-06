import os, time 
import eyetracker.camera.camera as camera
import eyetracker.camera.display as display
from eyetracker.analysis.detect import glint, pupil
from eyetracker.analysis.processing import gray2bgr, bgr2gray, mark, threshold, thresholds
import cv2
import numpy as np
import pygame
from PIL import Image

cRED = pygame.Color(255,0,0)
cBLUE = pygame.Color(0,0,255)
cGREEN = pygame.Color(0, 255, 0)
cYELLOW = pygame.Color(255, 255, 0)
cBLACK = pygame.Color(0,0,0)
cWHITE = pygame.Color(255,255,255)

def nothing(x):
    pass

def find_pup(where_glint,where_pupil):
    """It finds right pupil values from glint *where_glint* 
    and pupils positions *where_pupil*"""
    xsr,ysr=np.mean(where_glint,axis=0)
    ls=[]
    try:
        for wsp in where_pupil:
            r=np.sqrt((xsr-wsp[0])**2+(ysr-wsp[1])**2)
            ls.append([r,wsp[0],wsp[1]])
    except TypeError:
        return np.array([np.NaN,np.NaN])
    return np.array(min(ls)[1:])

def eye_calibr_params(points):
    """Returns parameters needed to scale eyetracker cursor.
    *points* is a vector of data points from callibration session"""
    points = np.asarray(points)
    xmax,ymax = np.max(points,axis=0)
    xmin,ymin = np.min(points,axis=0)
    return xmin, xmax-xmin, ymin, ymax-ymin

def eye_calibr_params2(points):
    """
    Sensitive to a fixation point.
    Returns parameters needed to scale eyetracker cursor
    *points* is a vector of data points from callibration
    """
    points = np.asarray(points)
    xmax,ymax = np.max(points,axis=0)
    xmean,ymean = np.mean(points,axis=0)
    xmin,ymin = np.min(points,axis=0)
    if abs(xmean-xmin)>=abs(xmean-xmax):
        amin = xmean-xmax
        awidth = 2*xmax
    else:
        amin = xmean-xmin
        awidth = 2*xmin
    if abs(ymean-ymin)>=abs(ymean-ymax):
        bmin = ymean-ymax
        bhight = 2*ymax
    else:
        bmin = ymean-ymin
        bhight = 2*ymin       
    return amin, awidth, bmin, bhight

def resize(img_name, new_size, out_name=None):
    """resize image from path *img_name* to size *new_size*
    and saves as 'img_name+res' if *out_name* is None"""
    img = Image.open(img_name)
    img = img.resize(new_size)
    if not out_name:
        nam, ext = img_name.split(".")
        new_name = nam + "_res." + ext
        img.save(new_name)
    return new_name

def check_size(img_name):
    "Returns size of file named *img_name*"
    img = Image.open(img_name)
    return img.size 


class EyeTracer():
    """Class to trace your eye using eyetracker camera"""
    def __init__(self):
        cam_id = camera.lookForCameras()
        cams = cam_id.keys()
        try:
            self.cam = camera.Camera(cam_id['Camera_2'])
        except KeyError:
            self.cam = camera.Camera(cam_id['Camera_1'])
        self.window = 'Camera view'

        cv2.namedWindow(self.window)
        self.threshold_types = thresholds.keys()
        self.thresh_t, self.thresh_v = 1, 40

        self.frame = self.cam.frame()

        self.ry,self.rx = self.frame.shape[:2]
        
        self.numsave = 0

        self.N_b = 10
        self.buf_pup = np.zeros((self.N_b,2))
        self.buf_posit = np.zeros((40,2))
        
    def mean_pupfinder(self):
        """It averages *N_b* positions of the best finded pupil"""
        fp=find_pup(self.where_glint,self.where_pupil)
        if not np.any(np.isnan(fp)):
            self.buf_pup[:self.N_b-1,:] = self.buf_pup[1:,:]
            self.buf_pup[self.N_b-1,:] = fp
        else:
            fp = None 
        return np.mean(self.buf_pup,axis=0).astype(np.uint)

    def mean_eyeposition(self,x,y):
        """todo"""
        Nmb = 40
        self.buf_posit[:Nmb-1,:] = self.buf_posit[1:,:]
        self.buf_posit[Nmb-1,:] = [x,y]

        return np.mean(self.buf_posit,axis=0).astype(np.uint)
    
    def eye_viewer(self):
        """Functions give you possibility to preview finded: glint - as red circle
        pupils - as blue, and final pupil as green"""
        cv2.createTrackbar('threshold value', self.window, self.thresh_v, 255, nothing)
        cv2.createTrackbar('threshold type', self.window, self.thresh_t, 5, nothing)
        while True:
            self.thresh_v = cv2.getTrackbarPos('threshold value', self.window)
            self.thresh_t = int(cv2.getTrackbarPos('threshold type', self.window))
            frame = self.cam.frame()
            frame_gray = bgr2gray(frame)
            frame_thresh = threshold(frame_gray, self.thresh_v, 
                                     thresh_type=self.threshold_types[self.thresh_t])
            self.where_glint = glint(frame_gray,maxCorners=2)
            self.where_pupil = pupil(frame_thresh)
            mx,my =  self.mean_pupfinder()
            mark(frame, self.where_glint,color='red')
            mark(frame, self.where_pupil, color='blue')
            mark(frame, np.array([mx,my]),radius=20, color='green')
            key = display.displayImage(frame, where=self.window)
            if key == 27 or key == ord('q'):
                break

    def calibrate(self, calibr_length = 18):
        """Calibration where maximal range of eye movement is measured 
        and scaled. You can change a time of calibration *calibr_length*
        in seconds"""

        pygame.init()
        self.screen = pygame.display.set_mode((self.rx,self.ry),pygame.FULLSCREEN)
        pygame.display.set_caption('calibration')
        cal_positions = []
        t0 = time.time()
        t_arr = time.time()
        
        rectangle = [(self.screen, cRED, (0,0,40,self.rx)),
                    (self.screen, cRED, (0,0,self.rx,40)),
                    (self.screen, cRED, (self.rx-40,0,40,self.ry)),
                    (self.screen, cRED, (0,self.ry-40,self.rx,40))]
        rl = 0
        blacktangle = [(self.screen, cBLACK, (0,0,40,self.rx)),
                    (self.screen, cBLACK, (0,0,self.rx,40)),
                    (self.screen, cBLACK, (self.rx-40,0,40,self.ry)),
                    (self.screen, cBLACK, (0,self.ry-40,self.rx,40))]
        pygame.draw.circle(self.screen, cWHITE, (self.rx/2,self.ry/2), 10)
        while True:
            keys=pygame.key.get_pressed()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    break
            if keys[ord('q')]:
                pygame.quit()
                break
            if keys[ord('z')]:
                self.screen.fill(cBLACK)

            frame = self.cam.frame()
            frame_gray = bgr2gray(frame)
            frame_thresh = threshold(frame_gray, self.thresh_v, 
                                     thresh_type=self.threshold_types[self.thresh_t])

            self.where_glint = glint(frame_gray,maxCorners=2)
            self.where_pupil = pupil(frame_thresh)
                    
            mx,my =  self.mean_pupfinder()
            pygame.draw.circle(self.screen, cRED, (int(self.rx-mx),my), 1)
            pygame.display.flip()

            if time.time()-t0>1:
                cal_positions.append((int(self.rx-mx),my))
            
            if time.time() - t_arr>2:
                if rl>0:
                    pygame.draw.rect(*blacktangle[rl-1])
                elif rl==0:
                    pygame.draw.rect(*blacktangle[3])
                pygame.draw.rect(*rectangle[rl])
                rl+=1
                t_arr = time.time()
                if rl>=4:
                    rl=0

            if time.time() -t0>calibr_length:
                pygame.quit()
                break

            key = 0
            if key == 27 or key == ord('q'):
                pygame.quit()
                break
        
        print 'Calibration time!'
        if len(cal_positions)>0:
            self.x0, self.Ax, self.y0, self.By = eye_calibr_params2(cal_positions)
    
    @property 
    def calibr_check(self):
        "It checks if calibration was made"
        try:
            self.x0
            self.y0
        except AttributeError:
            raise Exception("Calibration is needed!")

    def new_pos(self,mx,my):
        "Returns new position from calibration results"
        smax = self.x0+self.Ax
        smay = self.y0+self.By
        meanx = 0.5*(2*self.x0+self.Ax)
        meany = 0.5*(2*self.y0+self.By)
        C = 2.
        #zx = int((self.rx*C/self.Ax)*(mx-self.x0-self.rx))
        #zy = int((self.ry*C/self.By)*(my-self.y0-self.ry))
        #meanx,meany = self.mean_eyeposition(zx,zy)
        xp, yp = mx - meanx, my - meany
        zx = (self.rx*C/self.Ax)*xp + self.rx*0.5
        zy = (self.ry*C/self.By)*yp + self.ry*0.5
        #return int(zx-(meanx-zx)), int(zy-(meanx-zy))
        return int(self.rx-zx), int(zy)
        
    def painter(self):
        "Paint by looking on the screen. 'q' -exits the window"
        self.calibr_check
        pygame.init()
        self.screen = pygame.display.set_mode((self.rx,self.ry),pygame.FULLSCREEN)
        pygame.display.set_caption('Eye tracer')
        while True:
            keys=pygame.key.get_pressed()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    break
            if keys[ord('q')]:
                pygame.quit()
                break
            if keys[ord('z')]:
                self.screen.fill(cBLACK)

            frame = self.cam.frame()
            frame_gray = bgr2gray(frame)
            frame_thresh = threshold(frame_gray, self.thresh_v, 
                                     thresh_type=self.threshold_types[self.thresh_t])

            self.where_glint = glint(frame_gray,maxCorners=2)
            self.where_pupil = pupil(frame_thresh)
                    
            mx,my =  self.mean_pupfinder()
            
            zx,zy = self.new_pos(mx,my)

            pygame.draw.circle(self.screen, cGREEN, (zx,zy), 5)
            pygame.display.flip()
            print zx,zy

    def image_trace(self,imag_path, delay = False):
        """
        Give path to your image *imag_path* and look look at it.
        You can save a trace of your eye by pressing 's' if *delay*
        is not zeros if exits after given time in seconds
        """
        self.calibr_check
        pygame.init()
        self.screen = pygame.display.set_mode((self.rx,self.ry),pygame.FULLSCREEN)
        pygame.display.set_caption('NG tracer')
        if not check_size(imag_path) == (self.rx,self.ry):
            imag_path = resize(imag_path, (self.rx,self.ry))
        pic = pygame.image.load(imag_path)
        self.screen.blit(pic,(0,0))
        t0 = time.time()
        while True:
            keys=pygame.key.get_pressed()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    break
            if keys[ord('q')]:
                pygame.quit()
                break
            if keys[ord('s')]:
                self.save_screen()
                pygame.quit()
                break
            if delay:
                if time.time()-t0>delay:
                    self.save_screen()
                    pygame.quit()
                    break
            frame = self.cam.frame()
            frame_gray = bgr2gray(frame)
            frame_thresh = threshold(frame_gray, self.thresh_v, 
                                     thresh_type=self.threshold_types[self.thresh_t])

            self.where_glint = glint(frame_gray,maxCorners=2)
            self.where_pupil = pupil(frame_thresh)
                    
            mx,my =  self.mean_pupfinder()
            zx,zy = self.new_pos(mx,my)

            pygame.draw.circle(self.screen, cGREEN, (zx,zy), 5)
            pygame.display.flip()

    def save_screen(self):
        """
        save trace and picture to file in current directory
        ng_<NR>.jpeg where <NR> is number of showed picture
        """
        pygame.image.save(self.screen, "ng_{0}.jpeg".format(self.numsave))
        self.numsave+=1

    def series(self, lst_pics, delay):
        """
        Series of pictures names *lst_pics* to trace. *delay* is 
        time to show each picture.
        """
        for pic in lst_pics:
            self.image_trace(pic, delay)


if __name__ == '__main__':
    folder = 'pictures/'
    fotos = ['car.png', 'lady.png', 'ketchup.png','cristiano.jpg','starwars.jpg']
    app = EyeTracer()
    app.eye_viewer()
    app.calibrate()
    app.painter()
    #app.image_trace(folder+fotos[-2])
    #app.series([folder + f for f in fotos], 4)
