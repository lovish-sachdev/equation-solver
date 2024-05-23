import cv2
import time
import os
import random
import matplotlib.pyplot as plt
from PIL import Image

char_list = ["0","1","2","3","4","5","6","7","8","9","+","-","x","(",")"]
 
def encode_to_labels(txt):
    # encoding each output word into digits
    dig_lst = []
    for index, char in enumerate(txt):
        try:
            dig_lst.append(char_list.index(char))
        except:
            print(char)
        
    return dig_lst

def find_dominant_color(image):
        #Resizing parameters
        width, height = 150,150
        image = image.resize((width, height),resample = 0)
        #Get colors from image object
        pixels = image.getcolors(width * height)
        #Sort them by count number(first element of tuple)
        sorted_pixels = sorted(pixels, key=lambda t: t[0])
        #Get the most frequent color
        dominant_color = sorted_pixels[-1][1]
        return dominant_color

def preprocess_img(img, imgSize):
    "put img into target img of size imgSize, transpose for TF and normalize gray-values"

    # there are damaged files in IAM dataset - just use black image instead
    if img is None:
        img = np.zeros([imgSize[1], imgSize[0]]) 
        print("Image None!")

    # create target image and copy sample image into it
    (wt, ht) = imgSize
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    newSize = (max(min(wt, int(w / f)), 1),
               max(min(ht, int(h / f)), 1))  # scale according to f (result at least 1 and at most wt or ht)
    img = cv2.resize(img, newSize, interpolation=cv2.INTER_CUBIC) # INTER_CUBIC interpolation best approximate the pixels image
                                                               # see this https://stackoverflow.com/a/57503843/7338066
    most_freq_pixel=find_dominant_color(Image.fromarray(img))
    target = np.ones([ht, wt]) * most_freq_pixel  
    target[0:newSize[1], 0:newSize[0]] = img

    img = target

    return img


import numpy as np
import string
import imgaug.augmenters as iaa


global gray_back
kernel=np.ones((2,2),np.uint8)
kernel2=np.ones((1,1),np.uint8)





# #Base backgound.
# backfilelist=os.listdir('./background/')
# backgroud_list=[]

# for bn in backfilelist:
#     fileloc='./background/'+bn
#     backgroud_list.append(Image.fromarray(cv2.imread(fileloc,0)))


def random_brightness(img):
    img=np.array(img)
    brightness=iaa.Multiply((0.2,1.2))
    img=brightness.augment_image(img)
    return img

def dilation(img):
    img=np.array(img)
    img=cv2.dilate(img,kernel2,iterations=1)
    return img

def erosion(img):
    img=np.array(img)
    img=cv2.erode(img,kernel,iterations=1)
    return img

def blur(img):
    img=np.array(img)
    img=cv2.blur(img,ksize=(3,3))



def fuse_gray(img):
    img=np.array(img)
    ht,wt=img.shape[0],img.shape[1]
    gray_back=cv2.imread('gray_back.jpg',0)
    gray_back=cv2.resize(gray_back,(wt,ht))

    blended=cv2.addWeighted(src1=img,alpha=0.8,src2=gray_back,beta=0.4,gamma=10)
    return blended




def get_random_background():
    gradient_img=get_random_gradient()
    gaussian_noise=get_gaussian_noise()
    random_noise=get_random_noise()
    random_lines=get_random_lines()

    rand1=random.random()
    blended_image=cv2.addWeighted(gradient_img,rand1,gaussian_noise, 1-rand1,0)

    rand1=random.random()
    blended_image=cv2.addWeighted(blended_image,rand1,random_noise, 1-rand1,0)

    rand1=random.random()
    blended_image=cv2.addWeighted(blended_image,rand1,random_lines, 1-rand1,0)
    
    return blended_image

def get_random_noise():
    ran=random.randint(300,500)
    
    width,height=640,640
    white_image=np.ones((height,width,3),dtype="uint8")*255

    for i in range(ran):
        x=random.randint(0,640)
        y=random.randint(0,640)
        
        r=random.randint(0,255)
        b=random.randint(0,255)
        g=random.randint(0,255)

        t=random.choice([1,2])

        cv2.circle(white_image,(x,y),t,(r,g,b),-1)
    return white_image

def get_random_lines():
    coord_list=[]
    for i in range(0,580,30):
        coord_list.append([i,int(random.random()*50)+i])


    
    width,height=640,640
    white_image=np.ones((height,width,3),dtype="uint8")*255

    for a,b in coord_list:
        random_numbers = np.random.randint(a, b + 1, size=random.choice([0,2,4]))
        random_numbers.sort()
        r=random.randint(0,255)
        b=random.randint(0,255)
        g=random.randint(0,255)

    
        for i in range(len(random_numbers)//2):
            x,y=random_numbers[i],random_numbers[i+1]
            cv2.line(white_image,(0,x),(640,y),(r,g,b),1)
    return white_image

def get_random_gradient(width=640,height=640):
    white_image=np.ones((height,width,3),dtype="uint8")*255
    
    start_color=np.random.randint(0,256,3)
    end_color=np.random.randint(0,256,3)
     
    if random.random()<0.5:
        for x in range(width):
                gradiet_ratio=x/width
                color=(1-gradiet_ratio)*start_color+gradiet_ratio*end_color
                white_image[:,x]=color.astype(np.uint8)
            
    else:
        for x in range(height):
                gradiet_ratio=x/height
                color=(1-gradiet_ratio)*start_color+gradiet_ratio*end_color
                white_image[x,:]=color.astype(np.uint8)


    return white_image

def get_gaussian_noise():
        width,height=640,640
        white_image=np.ones((height,width,3),dtype="uint8")*255
    
        mean=0
        varience=100
        sigma=varience**0.5
        gaussian_noise=np.random.normal(mean,sigma,white_image.shape).astype(np.uint8)
        noisy_image=cv2.addWeighted(white_image,0.5,gaussian_noise,0.5,0)

        return noisy_image



def remove_padding(img):

    h1=0
    for x in range(img.shape[0]):
        if img[x,:].max()==0:
            h1=x
        else:
            break
    h2=img.shape[0]-1
    for x in range(img.shape[0]-1,-1,-1):
        if img[x,:].max()==0:
            h2=x
        else:
            break
    w1=0
    for x in range(img.shape[1]):
        if img[:,x].max()==0:
            w1=x
        else:
            break
    w2=img.shape[1]-1
    for x in range(img.shape[1]-1,-1,-1):

        if img[:,x].max()==0:
            w2=x
        else:
            break
    return h1,w1,h2,w2




def get_random_image(images_folder,images_list):
    img_name=random.choice(images_list)
    img_path=os.path.join(images_folder,img_name)
  
    if img_name[0] in ["(",")"]:
        label=img_name.split("_")[0]
        
        KERNEL=np.array([[2.5/4,2.5/4],
                    [2.5/4,2.5/4],
                ])
        img=cv2.imread(img_path)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img=255-img
        h1,w1,h2,w2=remove_padding(img)
        img=img[h1:h2+1,w1:w2+1]
        img=cv2.resize(img,(28,28),interpolation=cv2.INTER_CUBIC)
        img = cv2.filter2D(img, -1, KERNEL)
    else:
        label=img_name.split("-")[0] 
        img=cv2.imread(img_path)
        img=255-img
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return label,img


def random_single(images_folder,final_images_list,width_constraint=51):
      try:
        label,img=get_random_image(images_folder,final_images_list)

        li=remove_padding(img)
        h=li[2]-li[0]
        w=li[3]-li[1]
    
        img=img[li[0]:li[2]+1,li[1]:li[3]+1]
        height,width=img.shape
        randomm=random.randint(28,min(width_constraint-1,50))
        if height>width:
          size=(int(randomm*width/height),randomm)
        else:
          size=(randomm,int(randomm*height/width))
        img=cv2.resize(img,size,interpolation=cv2.INTER_AREA)
        return label,img
      except:
       
       return  random_single(images_folder,final_images_list,width_constraint=width_constraint)






def f1(ini,fini):
    x=np.array([i for i in range(ini,fini)])
    a=(fini+ini+1)//2
    a+=10
    y=(1-((x-a)/a)**2)**0.5
    if random.random()<0.5:
        y=1-y
    rand_y=random.randint(0,50)
    return x,y*rand_y

def f2(ini,fini):
    mid=random.randint(ini,fini)
    x1,y1=f1(ini,mid)
    x2,y2=f1(mid,fini)
    x=np.concatenate([x1,x2])
    y=np.concatenate([y1,y2])
    return x,y





def function(max_len=15):
    strr=""
    for i in range(random.randint(0,15)):
        strr+=random.choice(char_list)
    return strr

def add_padd():
    img=np.zeros((150,1024))
    ques=function()
    
    x,y=f2(0,1024)
    ini_x=0
    for i in ques:
        if i=="-":
            i="minus"
        if i=="+":
            i="plus"
        
        label,img2=random_single(images_folder,my_dict[i])
        ini_x+=random.randint(0,30)
        curr_y=int(y[ini_x])
        h,w=img2.shape
        img[50+curr_y:50+curr_y+h,ini_x:ini_x+w]=img2
        ini_x+=w
    mini=1024
    for i in range(1023,-1,-1):
        if img[:,i].max()==0:
            mini=i
        else:
            break
    img=img[:,:mini]
    maxi=0
    for i in range(0,150):
        if img[i,:].max()==0:
            maxi=i
        else:
            break
    img=img[maxi:,:]
    
    maxi=0
    for i in range(img.shape[0]-1,-1,-1):
        if img[i,:].max()==0:
            maxi=i
        else:
            break
    img=img[:maxi,:] 
    
    return img,ques
listt=[]
images_folder=r"C:\Users\07032\Downloads\bhmsds\symbols"
images_list=os.listdir(images_folder)

final_images_list=[]
for name in images_list:
    if name.split("-")[0]  not in ["slash","z","dot","w","y"]:
        final_images_list.append(name)

total_label=["0","1","2","3","4","5","6","7","8","9","plus","minus","x","(",")"]
total_label2=["0","1","2","3","4","5","6","7","8","9","+","-","x","(",")"]

my_dict={}
for i in total_label:
    my_dict[i]=[]

for name in final_images_list:
    if "_" in name:
        curr=name.split("_")[0]
    else:
        curr=name.split("-")[0]


    my_dict[curr].append(name)



back_img_folder=r"C:\Users\07032\python\projects\captch generator\seg_pred"
back_img_list=os.listdir(back_img_folder)




def random_transformation(img):
    if np.random.rand()<0.5:
        img=fuse_gray(img)
    elif np.random.rand()<0.5:
        img=random_brightness(img)
    elif np.random.rand()<0.5:
        img=dilation(img)
    elif np.random.rand()<0.5:
        img=erosion(img)

    else:
        img=np.array(img)
    return Image.fromarray(img)



 
max_label_len = 15



count=0
for i in range(600):
    try: 
        if i%100==0:
            print(i)
        img,label=add_padd()
        
        img_background=get_random_background()
        img_background=cv2.cvtColor(img_background,cv2.COLOR_BGR2GRAY)
        img_background=cv2.resize(img_background,(img.shape[1],img.shape[0]))
        back_img=cv2.imread(os.path.join(back_img_folder,random.choice(back_img_list)))
        back_img=cv2.cvtColor(back_img,cv2.COLOR_BGR2GRAY)
        back_img=cv2.resize(back_img,(img.shape[1],img.shape[0]))
        a=random.random()
        img=img.astype("uint8")
        if a<0.8:
            a=0.8
        img=cv2.addWeighted(img,a,img_background,1-a,0)
        if a<0.8:
            a=0.8
        img=cv2.addWeighted(img,a,back_img,1-a,0)
    
        if random.random()<0.5:
                img=255-img    
 
        img=preprocess_img(img,(384,96))
        cv2.imshow("img",img/255.)
        img=np.expand_dims(img,axis=-1)
        
        txt = label
        name_img=str(i)+"_img.jpg"
        name_label=str(i)+"_label.txt"
       
        ## for saving images
        
        # cv2.imwrite(r"C:\Users\07032\python\projects\captch generator\Data-generator-for-CRNN\img\\"+name_img, img)
        # with open(r"C:\Users\07032\python\projects\captch generator\Data-generator-for-CRNN\labi\\"+name_label,"w") as f:
        #     f.write(txt)
        cv2.waitKey(200)
    except:
        count+=1
        print(i,count)
    
cv2.destroyAllWindows()
           
        
