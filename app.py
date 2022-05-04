
import streamlit as st
import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image as Img
import cv2
import glob

def process_image_named(name, threshold_cutoff = 0.90, use_transparency = False):

  extension = "."+name.split(".")[-1]
  name = name.replace(extension,"")
  
  result_img = load_img('./test_data/u2net_results/'+name+'.png')
  # convert result-image to numpy array and rescale(255 for RBG images)
  RESCALE = 255
  out_img = img_to_array(result_img)
  out_img /= RESCALE
  # define the cutoff threshold below which, background will be removed.
  THRESHOLD = threshold_cutoff

  # refine the output
  out_img[out_img > THRESHOLD] = 1
  out_img[out_img <= THRESHOLD] = 0

  if use_transparency:
    # convert the rbg image to an rgba image and set the zero values to transparent
    shape = out_img.shape
    a_layer_init = np.ones(shape = (shape[0],shape[1],1))
    mul_layer = np.expand_dims(out_img[:,:,0],axis=2)
    a_layer = mul_layer*a_layer_init
    rgba_out = np.append(out_img,a_layer,axis=2)
    mask_img = Img.fromarray((rgba_out*RESCALE).astype('uint8'), 'RGBA')
  else:
    mask_img = Img.fromarray((out_img*RESCALE).astype('uint8'), 'RGB')

  # load and convert input to numpy array and rescale(255 for RBG images)
  input = load_img('./test_data/test_images/'+name + extension)
  inp_img = img_to_array(input)

  #st.image(inp_img)

  inp_img /= RESCALE

 
  if use_transparency:
    # since the output image is rgba, convert this also to rgba, but with no transparency
    a_layer = np.ones(shape = (shape[0],shape[1],1))
    rgba_inp = np.append(inp_img,a_layer,axis=2)

    #simply multiply the 2 rgba images to remove the backgound
    rem_back = (rgba_inp*rgba_out)
    rem_back_scaled = Img.fromarray((rem_back*RESCALE).astype('uint8'), 'RGBA')
  else:
    rem_back = (inp_img*out_img)
    rem_back_scaled = Img.fromarray((rem_back*RESCALE).astype('uint8'), 'RGB')

  # select a layer(can be 0,1 or 2) for bounding box creation and salient map
  LAYER = 2
  out_layer = out_img[:,:,LAYER]

  # find the list of points where saliency starts and ends for both axes
  x_starts = [np.where(out_layer[i]==1)[0][0] if len(np.where(out_layer[i]==1)[0])!=0 else out_layer.shape[0]+1 for i in range(out_layer.shape[0])]
  x_ends = [np.where(out_layer[i]==1)[0][-1] if len(np.where(out_layer[i]==1)[0])!=0 else 0 for i in range(out_layer.shape[0])]
  y_starts = [np.where(out_layer.T[i]==1)[0][0] if len(np.where(out_layer.T[i]==1)[0])!=0 else out_layer.T.shape[0]+1 for i in range(out_layer.T.shape[0])]
  y_ends = [np.where(out_layer.T[i]==1)[0][-1] if len(np.where(out_layer.T[i]==1)[0])!=0 else 0 for i in range(out_layer.T.shape[0])]
  
  # get the starting and ending coordinated for the box
  startx = min(x_starts)
  endx = max(x_ends)
  starty = min(y_starts)
  endy = max(y_ends)
  
  # show the resulting coordinates
  start = (startx,starty)
  end = (endx,endy)
  start,end

  cropped_rem_back_scaled = rem_back_scaled.crop((startx,starty,endx,endy))
  
  if use_transparency:
    rem_back_scaled.save('./test_data/final_result/'+name+'_cropped_no-bg.png')
  else:
    rem_back_scaled.save('./test_data/final_result/'+name+'_cropped_no-bg.jpg')

  #st.image(rem_back_scaled)
  
  #cropped_mask_img = mask_img.crop((startx,starty,endx,endy))

  # if use_transparency:
  #   cropped_mask_img.save('./test_data/final_result/'+name+'_cropped_no-bg_mask.png')
  # else:
  #   cropped_mask_img.save('./test_data/final_result/'+name+'_cropped_no-bg_mask.jpg')



def main():


    st.title("BACKGROUND_REMOVAL")


    img_file = st.file_uploader("Upload an image",type = ['png','jpg','jpeg'])


    if img_file is not None:


      #Saving upload
      with open(os.path.join("./test_data/test_images",img_file.name),"wb") as f:
        f.write((img_file).getbuffer())
      
      
      os.system('python u2net_test.py')

      names = os.listdir("./test_data/test_images")

      process_image_named(names[0], use_transparency=True)

      with open("./test_data/final_result/"+names[0].split(".")[0]+"_cropped_no-bg"+".png", "rb") as file:

          btn = st.download_button(
                   label="Download image",
                   data=file,
                   file_name=names[0].split(".")[0]+".png",
                   mime="image/png"
                 )

      files = glob.glob('./test_data/test_images/*')
      for f in files:
          os.remove(f)

      files = glob.glob('./test_data/u2net_results/*')
      for f in files:
          os.remove(f)

      files = glob.glob('./test_data/final_result/*')
      for f in files:
          os.remove(f)


main()