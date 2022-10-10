# Style Transfer
Trained a neural network to learn the styles from one image, and apply them against a given image. Based off research paper [Gatys et al (2016)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) where using a VGG network, style can be extracted. This is done by looking at specific layers inside the feature extractors of the network and using them to grab the important features of the network. This is then used to create a loss against a target image. This loss is then minimized by slowly modifiying the target image using a gram matrix. The more iterations, the more stylized the image becomes.

In Gatys et al (2016), there were a multitude of experiements that were conducted, but the most successfull was used in this repository. To see other experiemnts and how they faired against each other please see the research paper.

# Requirements
Please note that these are general requirements of what enviroment I used to run the code, more up-to-date versions will also suffice. The requirements are also listed in the ```requirements.txt``` file. The file was generated with ```pip freeze > requirements.txt```.

# How to use
Make sure you have the requirements setup in your enviroment. I use a general machine learning enviroment, so some of these you may not need.
* Modify any hyper parameters as needed
* Add own local files for style and content if needed
* Modify path location constants
* Run ```main.py```
