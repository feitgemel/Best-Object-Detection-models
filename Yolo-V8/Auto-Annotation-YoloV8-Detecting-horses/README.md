# No Dataset? No Problem! Create a Horse Race Detection Model using YoloV8 ? | Yolo V8
<p align="center">
  <img width="800" src="Yolo-V8-Horse race.png" "image">
</p>

##
<br/><br/> 

<font size= "4" >
in this video, We will learn how to train a horse race classes , even though we do not have a dataset. .  The pipeline includes video frame extraction, object detection model training, and displaying the results with bounding boxes and labels
<br/><br/> 
 Here’s an overview of what each part of the code does:

1. **Frame Extraction from Video**: The first part of the code extracts frames from a horse race video at regular intervals. These frames are saved as images to be used later for training and testing the YOLOv8 model. OpenCV and Supervision libraries are used to handle video frame extraction.

2. **Image Grid Visualization**: This section of the code helps visualize a sample grid of images that were extracted. The images are arranged in a grid layout, and titles are displayed for easy identification. This helps the user visually inspect the quality of the extracted frames.

3. **Automatic Annotations with GroundingDINO and Autodistill**: The third part automates the annotation process using GroundingDINO from the Autodistill library. It labels the images by detecting objects such as "horse" and "horse race" based on the defined ontology. The annotations are stored in YOLO format to be used for training the model.

4. **Visualizing Annotations**: The next section of the code displays random images from the dataset along with their annotations (bounding boxes). The YOLO format annotations are converted into pixel coordinates, and bounding boxes are drawn on the images, providing a visual representation of the labeled data.

5. **Training the YOLOv8 Model**: This part of the code loads a YOLOv8 model and trains it on the annotated horse race images. The model is configured using a data.yaml file that defines the dataset paths and classes. Training is performed over multiple epochs with the results and checkpoints saved to a specified directory.

6. **Running Object Detection on a Video**: Finally, the code loads the trained YOLOv8 model and applies it to a test video. The model performs object detection on each frame of the video, drawing bounding boxes and labels around detected objects. The processed video is displayed in real-time, allowing the user to see the model’s predictions.

Overall, this code demonstrates the full process of building an object detection model using YOLOv8, from data preparation to model training and testing on new video data. It leverages powerful tools like Supervision, GroundingDINO, and Autodistill to automate and streamline the workflow.

<br/>

You can find the link for the [tutorial](https://youtu.be/ujEDpRmaOaU) here. 

You can find more cool Object Detection projects and tutorials in this  [playlist](https://www.youtube.com/playlist?list=PLdkryDe59y4bXa-1wOEAF4KljIMamhWd0)


Enjoy

Eran
<br/><br/> 

</font>

# Recommended courses and relevant products 
<font size= "4" >

If you are interested in learning modern Computer Vision course with deep dive with TensorFlow , Keras and Pytorch , you can find it [here](http://bit.ly/3HeDy1V).

Perfect course for every computer vision enthusiastic

Before we continue , I actually recommend this [book](https://amzn.to/3STWZ2N) for deep learning based on Tensorflow and Keras : 



</font>

# Connect

<font size= "4" >
If you have any suggestions about papers, feel free to mail me :)

- [☕ Buy me a coffee](https://ko-fi.com/eranfeit)
- [🌐 My Website](https://eranfeit.net)
- [▶️ Youtube.com/@eranfeit](https://www.youtube.com/channel/UCTiWJJhaH6BviSWKLJUM9sg)
- [🐙 Facebookl](https://www.facebook.com/groups/3080601358933585)
- [🖥️ Email](mailto:feitgemel@gmail.com)
- [🐦 Twitter](https://twitter.com/eran_feit )
- [😸 GitHub](https://github.com/feitgemel)
- [📸 Instagram](https://www.instagram.com/eran_feit/)
- [🤝 Fiverr ](https://www.fiverr.com/s/mB3Pbb)
- [📝 Medium ](https://medium.com/@feitgemel)


</font>


