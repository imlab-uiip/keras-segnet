# keras-segnet
Implementation of [SegNet](http://mi.eng.cam.ac.uk/projects/segnet/)-like architecture using [keras](https://keras.io/).<br>
Current version **doesn't support index transferring** proposed in SegNet article, so it's basically a general Encoder-Decoder network. In `index-based-upsampling` folder you can find some code used for index transferring implemented for older version of theano.
<br>
<br>
![SegNet architecture](http://mi.eng.cam.ac.uk/projects/segnet/images/segnet.png)
