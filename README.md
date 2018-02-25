# opticalflow
Comparing FlowNet(DNN) and Dense optical flow(OpenCV).
Running run.py, drawing flow map & error map.

Trained parameter is needed to download from author of FlowNet.
If you need it, please search author's page.

Inputs image (predict flow vector of left -> right)
<table border="0">
<tr>
<td><img src="https://github.com/kou7215/opticalflow/blob/master/samples/0000000-imgL.jpg?raw=true"></td>
<td><img src="https://github.com/kou7215/opticalflow/blob/master/samples/0000000-imgR.jpg?raw=true"></td>
</tr>
</table>

Outputs(Dense optical flow of OpenCV(left) & FlowNet(right))
<table border="0">
<tr>
<td><img src="https://github.com/kou7215/opticalflow/blob/master/results/test_img_vector_cv2.jpg?raw=true"></td>
<td><img src="https://github.com/kou7215/opticalflow/blob/master/results/test_img_vector_dnn.jpg?raw=true"></td>
</tr>
</table>
