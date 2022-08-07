# Binarized-Neural-Network-on-Edges
Classification of images based on edges only. The images are converted to their gray-scale format followed by application of sobel filter to detect the edges. Models are applied on this new set of images with only edges to detect their class.

**Sample Images**

**Original**
![image](https://user-images.githubusercontent.com/65457437/144373938-3299b257-193d-46be-9057-af106b8d21a5.png)

**Edge Detection**
![image](https://user-images.githubusercontent.com/65457437/144374967-ff800beb-baf6-49ae-8dd4-bbf216fbc366.png)

**Original**
![image](https://user-images.githubusercontent.com/65457437/144375683-959066d4-c3f7-4f6d-b3e3-4d6f7d710e07.png)

**Edge Detection**
![image](https://user-images.githubusercontent.com/65457437/144375700-b0187031-71c3-4c59-b482-ffba0d6d7016.png)



Usually, we can easily identify objects in images where only a line drawing is given. Lines correspond to edges or sudden changes in images. Intuitively, most semantic and shape information from image can be encoded in the edges. It is more complicated presentation than original images. So, it can be very useful to extract edges from image, to use it for the recognition. The ideal result, is the drawing of an artist. But artist uses object-level knowledge. But I will mention how the machine learning algorithm can be constructed for edge detection. We all can see the edges as points of rapid change in image intensity. Such points can be identified by considering first derivative of image intensity. Edges will correspond to local extrema of derivative.

![image](https://user-images.githubusercontent.com/65457437/144350461-fde83f59-4450-47d1-b8e1-d9b875f517b8.png)

On the slide, I'll give a one-dimensional example, from one row in image. Because image is a two-dimensional function, we need to calculate image gradients as a vector of partial derivative of image intensity function. So, image gradient is a vector, where first element is partial derivative of f by x, and second element is partial derivative of f by y. The gradient direction is given by a tan of partial derivative of f by y over partial derivative of f by x. The edge strength is given by the gradient magnitude. Because images are discrete, we could approximate partial derivative of optical image. This finite difference in discrete images. Finite difference can be easily written as a simple convolution.

![image](https://user-images.githubusercontent.com/65457437/144350602-8c3512eb-9bcb-4d78-baa9-d6f91dccd40a.png)

![image](https://user-images.githubusercontent.com/65457437/144350662-11d196ac-94f3-48e5-8df9-0841c0379459.png)

For example, this 2 by 1 filter canel and Vel is minus 1 and plus 1. Other approximation of derivative filters exists. I put here a Roberts separator, Prewett, Sobel and Sharr operators. These operators both smothens and estimate finite differences in images. They produce visually similar results. Here are the example of edge maps, computed by convolution with these filters. 

![image](https://user-images.githubusercontent.com/65457437/144351201-a1269d40-d591-4074-8026-60b8ea6e09e7.png)

![image](https://user-images.githubusercontent.com/65457437/144351221-43698458-7e7a-44fe-890e-d4ccce87b8b1.png)


You can see the difference between filter's is mostly small. Consider a single row or column in the real image. Usually, the image contained noise. The plotting intensity as function of position. Finite different filters respond strongly to noise. Noise results in pixels that look very different from the neighborhoods. So, finite difference use a lot of false responses. Generally, the larger the noise, the stronger the response and the larger the number of strong responses. So, if image contain noise, then the real edge can disappear. Like it's demonstrated in this slide. The solution,is to apply Gaussian smooths first to reduce the noise. After smoothen, edge detection kernel can identify the edge in our image. Because convolution is associative operation, we can combine both smooth and edge differentiation kernels into one filter kernel. This save us one operation and produce the same result, as applying smoothen and differentiation operator successively. Here are visualization for derivative of Gaussian filter which can be used for gradient estimation in real images. I give visualization of two filters. One estimate gradients in x direction, and one estimate gradient in y direction. Smooth derivative removes noise but blurs edges in images. The larger the Gaussian filter, the stronger the smoothen. As a result, we find images in different scales and appliance smoothen with different filter kernels. 

![image](https://user-images.githubusercontent.com/65457437/144353758-163e78e7-8393-4f2a-a6d1-4b522503f979.png)

In this example, if use small filter, we get a lot of small edges, small details. When we use a larger filter with the gradients only corresponding to large objects in the image. Gradient magnitude estimation is not a complete edge detector. Edges are usually six stripes or edges of large image gradients, as you can see in this example. Also they receive only if the joint set of points is large gradients, without connectivity information that link neighboring points into real edges. This problems, were addressed by well known canny edge detector.

![image](https://user-images.githubusercontent.com/65457437/144353943-87ea7044-b6f5-40e8-985b-8fe0b073d426.png)

This is probably the most openly used edge detection algorithm despite the fact that this was proposed in 1986. It has two steps for gradient estimation. First, there is non-maximum suppression. During the step, the thin multi-pixel of wide reaches down to a single pixel width. Second, is linking edge pixels together to form continuous boundaries. Here is the example. We take an original image of Lena, and apply gradient estimation first. The obtain map of gradient norm. You can notice a lot of thicker ridges at the edges. Now, we apply non-maximum suppression. This can be done in several ways. The easiest, is just to lose the pixels that a local maxima in the square neighborhood. For example, 3 by 3 pixels. The more complicated way, is to identify the points in either pixel rows along the image gradient vector. At q, We have a maximum if, its value is larger than those of both p and r. To get values in p and r, we should interpolate from neighbour pixels to get these values. So, we found a local maximum along the image gradient vector.

![image](https://user-images.githubusercontent.com/65457437/144354258-9a827fff-3cd1-4742-9717-9daee86ee9d3.png)

Edge linking is performed as following, assume the marked point is an edge point, then reconstruct the tangent to the edge curve, which is normal to the gradient in this point. And use it to predict the next point. This light is either r or s points. We select the point with largest gradient and link it to the current point q. When linking edge points, we use hysteresis. Hysteresis is just two thresholds. We use high threshold to start edge curves and low threshold to continue them. This allows us to trace weak edges, if they're connected to the strong edges. 

![image](https://user-images.githubusercontent.com/65457437/144354356-66b951f9-dabb-4001-bcd6-c25aca3b33eb.png)

Weak edges themselves won't be detected. This hysteresis threshold. Here is example of thinning on non-maximum suppression applies to the edge map obtained for this particular image. Hysteresis allow us to achieve a balance between the amount of detective edges and noise. If only one threshold is used, we either miss thick edges which are extensions of strong edges or get a lot of noise from weak edges alone. 

![image](https://user-images.githubusercontent.com/65457437/144354514-5fb8cc20-2b9b-4eba-b6c6-25fab40a398e.png)

By varying the sigma in gradient computation all the size of Gaussian smoothen or the size of derivative for Gaussian filter. It can detect edges on different scales, large sigma detects large scale edges. Small sigma detects fine features in images. On this example, you can see that by increasing the size of sigma from 1 to 2, removes vertical lines originated from stripes on the row. The change in image intensity is not the only source of edges in real images. As can be seen from the slight change in color or texture, also gives us visible edges in images. But such changes cannot be detected by image gradient or canny operator. Of course there are more advanced methods which can achieve this result. Mostly, they can see that edge detection as pixel classification problem and use machine learning techniques. We create a data set of ground truth labeling of edges and trade in which classifier with classified pixels to either edges or non-edges by considering different features, like texture, color, image, intensity. But for many computer vision tasks canny edge detector proves to be sufficient. For example it's often used as feature extractor and produce features which are later used for image recognition.
