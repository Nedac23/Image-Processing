imageFile = 'image name';
inputImage = imread(imageFile);
if size(inputImage, 3) == 3
    inputImageGray = rgb2gray(inputImage);
else
    inputImageGray = inputImage;
end
imshow(inputImageGray);
disp('Running SURF Detection in MATLAB....');
tic;
points = detectSURFFeatures(inputImageGray);
execTime = toc;
fprintf('Found %d SURF intrest points in %f seconds.\n', points.Count, execTime);
hold on;
plot(points.selectStrongest(50));
hold off;
%https://www.mathworks.com/help/gpucoder/ug/feature-extraction-using-surf.html