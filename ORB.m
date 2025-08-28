I = imread('imagename');
if size(I, 3) == 3
    I = rgb2gray(I);
end
points = detectORBFeatures(I);
imshow(I);
hold on;
plot(points.selectStrongest(20));
%https://www.mathworks.com/help/vision/ref/orbpoints.html