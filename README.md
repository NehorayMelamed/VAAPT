# Project Components

This project focuses on two major components:
1. **Video enhancement system and image processing.**
2. **Data extraction and analytics specific about vehicles and license plates.**



![PICTURE OF THE FLOW DIAGRAM](data_for_readme/window/WhatsApp%20Image%202023-08-17%20at%2001.19.27.jpg)


### Repositories:

1. **Real-ESRGAN** - Enhance low-resolution images with super-resolution techniques.
   - [GitHub Repository](https://github.com/xinntao/Real-ESRGAN)

2. **RVRT** - Techniques for de-blurring images and videos.
   - [GitHub Repository](https://github.com/JingyunLiang/RVRT)

3. **RealBasicVSR** - A repository dedicated to video super-resolution.
   - [GitHub Repository](https://github.com/ckkelvinchan/RealBasicVSR)

4. **NonUniformBlurKernelEstimation** - Estimate the blur kernel in images for de-blurring.
   - [GitHub Repository](https://github.com/GuillermoCarbajal/NonUniformBlurKernelEstimation)

5. **Grounded-Segment-Anything (SAM)** - Segmentation techniques based on text prompts.
   - [GitHub Repository](https://github.com/IDEA-Research/Grounded-Segment-Anything)

6. **Yolo Tracking** - Object detection, tracking, and REID using Yolo.
   - [GitHub Repository](https://github.com/mikel-brostrom/yolo_tracking)

7. **Co-Tracker Optical Flow** - Optical flow-based tracking system.
   - [GitHub Repository](https://github.com/facebookresearch/co-tracker)

8. **Frame Interpolation** - Interpolate frames in videos for smooth transitions.
   - [GitHub Repository](https://github.com/google-research/frame-interpolation)

9. **Inpaint-Anything** - Techniques for inpainting objects in images and videos.
   - [GitHub Repository](https://github.com/geekyutao/Inpaint-Anything)
   
10. **Yolo detection** - Techniques for detections segmentation.
    - [GitHub Repository](https://github.com/ultralytics)

11 **Segment anything ** - Techniques for segmentation.
    - [GitHub Repository](https://github.com/facebookresearch/segment-anything)



### Models:

1. **Lavis Model** - A model for question-answering based on images.
   - [GitHub Model](https://github.com/salesforce/lavis)

2. **ViLT Model** - Another model for image-based question-answering, hosted on Hugging Face.
   - [Hugging Face Model](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa)

## Additional Resources:

1. **PyTorch** - An open-source machine learning library to accelerate the path from research to production.
   - [Official Website](https://pytorch.org/)


2. **Plate recognizer** - A company provide ability to recognize license plate.
    - [Official Website](https://www.googleadservices.com/pagead/aclk?sa=L&ai=DChcSEwjGkJDBxeKAAxVO4XcKHXV2BEkYABAAGgJlZg&gclid=CjwKCAjw5_GmBhBIEiwA5QSMxCZda3a5mE0yoYktbZ_zXEI4fIhpZmXtUGiVFc_J5ROFlDeEhx6yNxoCYwEQAvD_BwE&ohost=www.google.com&cid=CAESauD2_B2NN45-FkV2up5AzKUQrc0nBbUvSoN346Mrn3GFysi08xw2JWmitdYxxFhSzty8r-wQhqbBQe-bMLjWoDI20sbVTkaqCRiwjMHPsUzYxG5WR1PURrx2HPvpk4OzmAOwMrSFI95Ruvc&sig=AOD64_0FowSWa9qeogp1bZ4z0n94Qye_YA&q&adurl&ved=2ahUKEwiGnYrBxeKAAxU3hf0HHZp_AY4Q0Qx6BAgHEAE)

------
_____

## 1. Video Enhancement System

### Image/Video Stabilization:

- **Image stabilization by classical calculations**:
    * **Options**: Directory of images input -> processing -> image result

- **Reticle image stabilization using optical flow and averaging**:
    * **Options**: 
        - Video input -> processing -> image result
          ![PICTURE](path/to/picture.png)
        - Video input -> processing -> video result with averaging
          ![VIDEO](path/to/video1.mp4)
        - Video input -> processing -> video result without averaging
          ![VIDEO](data_for_readme/stabilization_yoav/video_stabilize.mp4)

----
### Video Enhancement:

1. **Averaging and stabilization - ECC** by choosing a segmentation map for an object that we want to stabilize.
    * **Options**:
      - Directory of images + directory of segmentation mask -> processing -> image result
        ![IMAGE RESULT](data_for_readme/ecc_classic_dudy/Screenshot%20from%202023-08-17%2000-04-22.jpg)

2. **De-noise** - [Repository](https://github.com/xinntao/Real-ESRGAN)
    * **Options**: 
        - Video + ROI User selection -> processing -> denoised video result
      ![DEMO VIDEO](data_for_readme/denoise/demo_video.mp4)
      ![DEMO GIF](data_for_readme/denoise/denoise.gif)

3. **De-blur** - [Repository](https://github.com/JingyunLiang/RVRT)
    * **Options**: Video + ROI User selection -> processing -> De-blure video result
      ![GIF](data_for_readme/deblur/teaser_vdb.gif)
      ![IMAGE](data_for_readme/deblur/Screenshot%20from%202023-08-17%2000-21-28.jpg)
      ![ANIMATION](data_for_readme/deblur/animation4.gif)

4. **De-jpeg** - [Repository](https://github.com/ckkelvinchan/RealBasicVSR)
    * **Options**: Video + ROI User selection -> processing -> De-jpeg video result
      ![VIDEO](data_for_readme/dejpeg/143370859-e0293b97-f962-476f-acf8-14fad27cea77.mp4)

-----
### Video Processing 
1. **Blur kernel estimation** - [Repository](https://github.com/GuillermoCarbajal/NonUniformBlurKernelEstimation)
    * **Options**: Image + ROI User selection -> processing -> Kernel blur estimation
      ![IMAGE](data_for_readme/kernel_blur/Corrected_Cropped_Comparison_kernels.jpg)

2. **Object segmentation based on text prompt** - [Repository](https://github.com/IDEA-Research/Grounded-Segment-Anything)
    * **Options**: Directory of images + text prompt -> processing -> segmentation map
      ![IMAGE](data_for_readme/segmantion_sam/Screenshot%20from%202023-08-17%2000-06-49.jpg)

3. **Object Detection & Tracking & REID - YOLO** - [Repository](https://github.com/mikel-brostrom/yolo_tracking)
    * **Options**: Video + classes and other terms -> processing -> crop and other stuff
      ![IMAGE 1](data_for_readme/object_detection/Screenshot%20from%202023-08-17%2000-08-18.jpg)
      ![IMAGE 2](data_for_readme/object_detection/Screenshot%20from%202023-08-17%2000-13-41.jpg)

4. **Optical Flow** - [Repository](https://github.com/facebookresearch/co-tracker)
    * **Options**: Video + ROI User selection -> processing -> Video & PT

5 **Frame Interpolation** - [Repository](https://github.com/google-research/frame-interpolation)
    * **Options**: Image a + image b -> processing -> crop and other stuff
      ![CROP IMAGE](data_for_readme/frame_interpulation/moment.gif)

6. **Questions and Answering** - Based on the [Lavis Model](https://github.com/salesforce/lavis) and [ViLT Model](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa)
    * **Options**: 
      - Person file questions + image -> processing -> Answer
      - Car file questions + image -> processing -> Answer
      - Free text question + image -> processing -> Answer
        ![IMAGE](data_for_readme/questions_and_answering/Screenshot%20from%202023-08-17%2000-13-08.jpg)

7. **Create Motion Blur for Objects** - Based on optical flow and some logic. (Coming Soon)

8. **Remove/Inpaint Any Object**
    * **Options**: 
      - Video + ROI User selection -> processing (Model) -> Inpaint object inside video, [Repository](https://github.com/geekyutao/Inpaint-Anything)
        ![VIDEO 1](data_for_readme/remove_object/car_removed_road_from_stabilize_drone_cut.mp4)
      - Video + ROI User selection -> processing (Classic) -> Inpaint object inside video
        ![VIDEO 2](data_for_readme/remove_object/remove_with_roi_dji_shapira.mp4)
        ![IMAGE](data_for_readme/remove_object/Screenshot%20from%202023-08-17%2000-20-13.jpg)

9. **Segmentation based on Optical Flow** - Get segmentation (BB) based on the optical flow points track
    * **Options**: Video + ROI User selection -> processing -> cropped object, bb
      ![IMAGE Placeholder](path/to/image_placeholder.png)
   
-----

### Video Compression:
1. **Video Compression**: Compress video via H264 using YOLO Detector
    ![VIDEO Placeholder](data_for_readme/smart_compresssion/h264_25_merged_video_0_51_25_25.mp4)



_____
_____

## 2. Data Extraction and Analytics

Specifically regarding vehicles and license plates. Follow the provided web page for results for cars and license plates according to your terms. The results are provided as a JSON file per object (Car and its license) with the best score.
![WebPage](data_for_readme/web_page/Screenshot%20from%202023-08-17%2000-28-25.jpg)
#### Note -> to use this web page u should to buy and install the license for - [PlateRecognizerCompany](https://www.google.com/aclk?sa=l&ai=DChcSEwjx2_2bmuKAAxUY0XcKHZTHANAYABAAGgJlZg&gclid=CjwKCAjw5_GmBhBIEiwA5QSMxHvZG2VmGfzr7A3M9KEBDa1AsNjAU1Io8bo2zuvxvqACoi-ejk36oRoCKTwQAvD_BwE&sig=AOD64_3SFgcETFB0M1Dj-5PB3Za-oEFRKg&q&adurl&ved=2ahUKEwiY-vObmuKAAxX7_rsIHTGBBG8Q0Qx6BAgGEAE)
![LICENSE PLATE IMAGE](data_for_readme/licanse_plate_recognizer/Screenshot%20from%202023-08-17%2000-19-18.jpg)




# Get Started

#### 1. Clone the Repository
```bash
git clone https://github.com/NehorayMelamed/VPAAE.git
```

#### 2. Install PyTorch
```bash
pip3 install torch torchvision torchaudio
```

#### 3. Setup the Project
Navigate to the project's root directory and run:
```bash
pip install -r requirements.txt
sudo apt install python3-tk
sudo apt install tk-dev
```

## 4. Download and Place the .pt and .pth Files

1. **Download the Files**:  
   Download the `pt_and_pth_files.zip` from this [Drive link](https://drive.google.com/drive/folders/1lrCJdvpd-3Zmmeoxqu_uu5BkJkjgl2t5) and place it in the base directory of the project.

2. **Place the .pt Files in Their Locations**:  
   - Ensure you are in the root directory of the project (where the zip file should be located).
   - Navigate to the [SystemBuilding](SystemBuilding) directory:  
     ```
     cd SystemBuilding
     ```
   - Run the code inside the [SystemBuildingForUser.py](SystemBuilding%2FSystemBuildingForUser.py)  script:  
     ```bash
     python SystemBuildingForUser.py
     ```



#### 5. Run the Main Script
```bash
python gui_video_processing_via_tkniret_3.py
```

#### 6. You should see the main window
![window](data_for_readme/window/Screenshot%20from%202023-08-17%2001-04-54.jpg)


## **Remember** #
##### The way to get into the system for the vehicle and license plate is by clicking the below link of the window


## TODO:

- [ ] Add an option to upload a video or a folder of photos, uniformly for everyone.

- [ ] To create a connection between the different services.

- [ ] Make VIDEO COMPRESSION so that it is also possible to choose with a simple YOLO, and not just upload a folder of FRAMES, and MASKS.

- [ ] Support the choice of CPU or CUDA for:
     - Chain_demo_main_interface -(By co-track)
     - FRAME INTERPOLATION
     - DENOISE
     - DEBLUR
