# Basketball Players Images Depth Estmation Model

1. Data:

  * 100 Background images 
  * 200 Foreground images (100 + 100 flips)
  * 20 Positions of foreground overlayed on background
  
  Total : 
  * 400000 images of overaleyed FG_BG
  * 400000 images of overaleyed FG_BG Mask
  * 400000 images of overaleyed Depth images
  
  

2. Dataset location:

* Background, Foreground and Depth_Images are placed at https://drive.google.com/drive/u/2/folders/1fhv_7jYn8mBFv9PTnjyrY4nNVWS3COoM
 
* Folder: /Background
* Description: 
  * Number of images : 100
  * Image size :  224x224x3

![alt text](https://github.com/bharathts1507/TSAI-Assignments-EVA4/blob/master/Session%2014-15A/images/Background_sample_images.png) "Background images" 


* Folder: /Foreground
* Description: 
  * Number of images : 100
  * Image size :  100x100x4

![alt text](https://github.com/bharathts1507/TSAI-Assignments-EVA4/blob/master/Session%2014-15A/images/Foreground_sample_images.png) "Foreground images"


* Overlayed FG_BG and Mask are placed at https://drive.google.com/drive/folders/1wQYOtVuu99L74sWx0FjWyP9At5zpRYHm

* Folder: /FG_BG
* Description: 
  * Number of images: 400000
  * Image Size: 224x224x3
![alt text](https://github.com/bharathts1507/TSAI-Assignments-EVA4/blob/master/Session%2014-15A/images/Overlayed_fg_bg.png)  
  
  
* Folder: /FG_BG_MASK
* Description: 
  * Number of images: 400000
  * Image Size: 224x224x1
![](https://github.com/bharathts1507/TSAI-Assignments-EVA4/blob/master/Session%2014-15A/images/fg_bg_mask.png)  
  
  
* Folder: /DEPTH_IMAGES
* Description: 
  * Number of images: 400000
  * Image Size: 224x224x1
![](https://github.com/bharathts1507/TSAI-Assignments-EVA4/blob/master/Session%2014-15A/images/depth_image.png)  
  
  
2. Data Creation process:
* Foreground:
  * Used GIMP tool, to select the foreground of object in an image
  * Added an Alpha channel and removed the background
  * Save the file in PNG format

* Foreground Mask:
  * Read the FG image in python
  * Get alpha channel and make other channel values as 0
  * Assign value 255 to alpha channel to make white background
  
 * FG_BG 20 overlays:
   * Read BG image and FG image
   * Flip FG image using 'flip' option
   * Create 20 set of random positional co-ordinates 
   * Paste FG on BG at position from above list

 * FG_BG_Mask:
  * Follow the same process as mentioned in Foreground mask creation
  
  * Depth Images:
    * The trickiest part of the depth images creation is the data memory handling.
    * Stored FG_BG in a zip file 
    * Read each image from zip file
    * Used DenseDepth model to create depth images from overlayed images
    
4. Refer the code used to create this dataset:
* https://github.com/bharathts1507/TSAI-Assignments-EVA4/blob/master/Session%2014-15A/BasketBall_Players_Image_DepthEstimation_Dataset_Creation.ipynb


