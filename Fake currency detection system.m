input = GUI;

if isvalid(input)
    waitfor(input)
end

RMtype = str2num(RMtype);
imageFilePath = imread(imageFilePath);
templateImage = imread('templateFeature3.jpg');
newSize = [450, 870];
resultImage = imresize(imageFilePath, newSize);

if RMtype == 5
    % Call functions for feature 1 and 3
    [mainFeature1, Check1] = main(RMtype, imageFilePath);
    [feature3, Check3] = componentDetection(imageFilePath, templateImage);

    % Show final result
    figure, imshow(resultImage);
    if Check1 == 1 && Check3 == 1
        annotation('textbox', [0, 0, 1, 0.05], 'String', 'REAL CURRENCY', ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom',...
            'FontSize', 28, 'FontWeight', 'bold', 'EdgeColor', 'none', 'Color', 'green');
    else
        annotation('textbox', [0, 0, 1, 0.05], 'String', 'FAKE CURRENCY', ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
            'FontSize', 28, 'FontWeight', 'bold', 'EdgeColor', 'none', 'Color', 'red');
    end
elseif RMtype == 10 || RMtype == 20 || RMtype == 50 || RMtype == 100 
    % Call functions for feature 1, 2 and 3
    [mainFeature1, Check1] = main(RMtype, imageFilePath);
    [image1, image2, featureCheck] = feature2Check(RMtype, imageFilePath);
    [feature3, Check3] = componentDetection(imageFilePath, templateImage);

    % Show final result
    figure, imshow(resultImage);
    if Check1 == 1 && featureCheck == 1 && Check3 == 1
        annotation('textbox', [0, 0, 1, 0.05], 'String', 'REAL CURRENCY', ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom',...
            'FontSize', 28, 'FontWeight', 'bold', 'EdgeColor', 'none', 'Color', 'green');
    else
        annotation('textbox', [0, 0, 1, 0.05], 'String', 'FAKE CURRENCY', ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom',...
            'FontSize', 28, 'FontWeight', 'bold', 'EdgeColor', 'none', 'Color', 'red');
    end  
else
    % Show the applicable type of currencies
    figure, 
    subplot(3,2,1), imshow('5_v1.jpeg'), title('RM5');
    subplot(3,2,2), imshow('10_v1.jpeg'), title('RM10');
    subplot(3,2,3), imshow('20_v1.jpeg'), title('RM20');
    subplot(3,2,4), imshow('50_v1.jpeg'), title('RM50');
    subplot(3,2,5), imshow('100_v1.jpeg'), title('RM100');
    sgtitle('The system can only check for the following currencies.')
end


% -----Feature 1-----

function [mainFeature1, Check1] = main(RMtype, imageFilePath)
    % process image for 'BANKNEGARAMALAYSIA'
    croppedImg = cropToCenter(imageFilePath);
    gaussian = 3; % fixed gaussian value
    sharpenedImg = sharpenImg(croppedImg, gaussian);
    segmentedImg = segmentImage(sharpenedImg);
    processedBNMImg = segmentedImg;
    
    % process image for 'RINGGITMALAYSIA'
    croppedImg2 = cropToBottomRight(imageFilePath);
    clipLimit = 0; 
    if RMtype == 5
        clipLimit = 0.01;
        gaussian = 3;
    elseif RMtype == 10
        clipLimit = 0;
        gaussian = 4;
    elseif RMtype == 20
        clipLimit = 1;
        gaussian = 3;
    elseif RMtype == 50
        clipLimit = 0.01;
        gaussian = 6;
    elseif RMtype == 100
        clipLimit = 0.001;
        gaussian = 5;
    end

    sharpenedImg = sharpenImg(croppedImg2, gaussian);
    enhancedImg = adapthisteq(sharpenedImg, "ClipLimit", clipLimit);
    processedRMImg = enhancedImg;

    % check result
    result = checkFeature1(processedBNMImg, processedRMImg);
    
    % Display the result
    if result
        disp('Feature 1 PASSED');
        mainFeature1 = 'Feature 1 PASSED';
        Check1 = 1;
    else
        disp('Feature 1 FAILED');
        mainFeature1 = 'Feature 1 FAILED';
        Check1 = 0;
    end

    figure,
    subplot(2, 2, 1), imshow(croppedImg), title('Original BNM Image');
    subplot(2, 2, 2), imshow(processedBNMImg), title('Processed BNM Image');
    subplot(2, 2, 3), imshow(croppedImg2), title('Original RM Image');
    subplot(2, 2, 4), imshow(processedRMImg), title('Processed RM Image');
    sgtitle(mainFeature1); 
end


% function for cropping 'BANKNEGARAMALAYSIA'
function croppedImg = cropToCenter(inputImg)
    % Read image
    % inputImg = imread(inputImg);

    % Calculate crop dimensions 
    width = size(inputImg, 2)*0.45;
    height = size(inputImg, 1)/4;

    % Calculate coordinates
    topLeftX = floor((size(inputImg,2)-width)/2)+1;
    topLeftY = 1;

    bottomRightX = topLeftX + width -1;
    bottomRightY = topLeftY + height -1;

    % Perform cropping
    croppedImg = inputImg(topLeftY: bottomRightY, topLeftX: bottomRightX, :);
end

% function for cropping 'RINGGITMALAYSIA'
function croppedImg2 = cropToBottomRight(inputImg)
    % Read image
    % inputImg = imread(inputImg);

    % Calculate crop dimensions 
    width = size(inputImg, 2) * 0.38;
    height = size(inputImg, 1) / 7;

    % Calculate bottom-right coordinates
    bottomRightX = size(inputImg, 2);
    bottomRightY = size(inputImg, 1);

    % Calculate top-left coordinates based on the specified width and height
    topLeftX = bottomRightX - width + 1;
    topLeftY = bottomRightY - height + 1;

    % Perform cropping
    croppedImg2 = inputImg(topLeftY:bottomRightY, topLeftX:bottomRightX, :);
end

% function for Sharpening image
function sharpenedImg = sharpenImg(inputImg, gaussian)
    inputImg = rgb2gray(inputImg);

    % Denoise the image using a Gaussian filter
    denoised_img = imgaussfilt(inputImg, gaussian);

    % Define a 3x3 smoothing filter for further noise reduction
    smooth1 = (1/9)*[1,1,1;1,1,1;1,1,1];
    
    % Apply the smoothing filter to the denoised image
    smoothed_img = imfilter(denoised_img, smooth1);
    
    % Define a 3x3 sharpening filter to enhance image features
    sharp1 = [-1,-1,-1;-1,9,-1;-1,-1,-1];
    
    % Apply the sharpening filter to the smoothed image
    sharpened_img = imfilter(smoothed_img, sharp1);
    
    % Combine the original grayscale image with the sharpened image
    sharpenedImg = inputImg + sharpened_img;
    
    % figure,
    % subplot(2, 2, 1), imshow(inputImg), title('Original Image');
    % subplot(2, 2, 2), imshow(smoothed_img), title('Smoothed Image');
    % subplot(2, 2, 3), imshow(sharpened_img), title('Sharpened Image');
    % subplot(2, 2, 4), imshow(sharpenedImg), title('Final Sharpened Image');
end

% function for Segmenting image
function segmentedImg = segmentImage(inputImg)
    inputImg = double(inputImg);

    % Edge detection 
    BWs = edge(inputImg, 'sobel', (graythresh(inputImg)*.1));
    % figure, imshow(BWs), title('binary gradient mask');
    
    % Fill gaps
    se90 = strel('line',3,90);
    se0 = strel('line',3,0);
    
    % Dilate the image
    BWsdil = imdilate(BWs, [se90 se0]);
    %figure, imshow(BWsdil), title('dilated gradient mask');
    
    % Fill interior gaps
    BWdfill = imfill(BWsdil, "holes");
    %figure, imshow(BWdfill), title('binary image with filled holes');
    
    % Remove connected objects on border
    BWnoboard = imclearborder(BWdfill, 4);
    %figure, imshow(BWnoboard), title('cleared border image');
    
    % Smooth the object
    seD = strel('diamond', 1);
    segmentedImg = imerode(BWnoboard, seD);
    segmentedImg = imerode(segmentedImg, seD); % erode twice
    %figure, imshow(segmentedImg), title('segmented image');
end


function result = checkFeature1(inputImg1, inputImg2)
    % Define targets and corresponding display messages
    targets = {'BANKNEGARAMALAYSIA', 'RINGGITMALAYSIA'};

    % Initialize result array
    result = false(1, length(targets));

    % Loop through each target
    for i = 1:length(targets)
        % Perform OCR on the respective image
        ocrResult = ocr(eval(['inputImg', num2str(i)]), 'TextLayout', 'Block', 'CharacterSet', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ');

        % Extract recognized text
        recognizedText = ocrResult.Text;

        % Remove space and newline
        processedText = strrep(recognizedText, ' ', '');
        processedText = strrep(recognizedText, newline, '');

        % Convert to uppercase
        textUpper = upper(processedText);

        % Check the text with the target text
        if i == 1
            targetCheck = contains(textUpper, targets{i});
        else
            % Check for both variations of the second target
            targetCheck = any(contains(textUpper, 'RINGGITMALAYSIA')) || any(contains(textUpper, 'RINGGIT MALAYSIA'));
        end

        % Set the result of text detection
        result(i) = targetCheck;

        % Display the processed text and result
        disp(['Image ', num2str(i), ': ', processedText]);
        disp(['Target "', targets{i}, '" Found: ', num2str(targetCheck)]);
        disp(' '); % Add a blank line for better readability
    end
end


% -----Feature 2-----

function [imageOri, imageProcessed, Check] = feature2Check(RMtype, imageFilePath)
    if RMtype == 10
        [image1, image2, result] = IsTen(imageFilePath);
        result = str2double(result);
        if result == 10
            Check = 1;
        else
            Check = 0;
        end
    elseif RMtype == 20
        [image1, image2, result] = IsTwenty(imageFilePath);
        result = str2double(result);
        if result == 20
            Check = 1;
        else
            Check = 0;
        end
    elseif RMtype == 50
        [image1, image2, result] = IsFifty(imageFilePath);
        result = str2double(result);
        if result == 50 || result == 51 || result == 511
            Check = 1;
        else
            Check = 0;
        end
    elseif RMtype == 100
        [image1, image2, result] = IsHundred(imageFilePath);
        result = str2double(result);
        if result == 100 || result == 1111
            Check = 1;
        else
            Check = 0;
        end
    end
     
    imageOri = image1;
    imageProcessed = image2;

    if Check == 1
        figure,
        subplot(121), imshow(imageOri), title('Original Image');
        subplot(122), imshow(imageProcessed), 
        rectangle('Position', [140,1230,400,320], 'EdgeColor', 'r', 'LineWidth', 2), title('Processed Image');
        sgtitle('Feature 2 PASSED');
    else
        figure,
        subplot(121), imshow(imageOri), title('Original Image');
        subplot(122), imshow(imageProcessed),
        rectangle('Position', [140,1230,400,320], 'EdgeColor', 'r', 'LineWidth', 2), title('Processed Image');
        sgtitle('Feature 2 FAILED');
    end
end


function [oriroiImage, highContrastImage, ocrResult] = IsTen(image)
    img = image;
    newSize = [1696, 3560];
    img = imresize(img, newSize);
    
    % CROPPING REGION OF INTEREST
   
    % Calculate the center of the image
    centerX = size(img, 2) / 2;

    % Define the width and height of the region of interest (ROI)
    roiWidth = 670;  % Adjusting width
    roiHeight = size(img, 1);  % Use full height

    % Set the offset from the left edge
    leftOffset = 1260; 

    % Calculate the starting point for the crop based on the offset
    roiX = round(centerX - roiWidth / 2) - leftOffset;

    % Ensure the ROI is within the image boundaries
    roiX = max(1, roiX);

    % Extract the region of interest from the original image
    oriroiImage = imcrop(img, [roiX, 1, roiWidth, roiHeight]);
    roiImage = im2gray(oriroiImage);

    % Brightness Enhancement
    brightenedImage = imadjust(roiImage, [0.6, 0.9], [0.1, 1.0]);
    
    % Noise Removal
    sigma = 6.0;  % Adjust standard deviation 
    smoothedImage = imgaussfilt(brightenedImage, sigma);

    %smoothedImage = medfilt2(brightenedImage, [5, 5])S;

    % Contrast Adjustment 
    adjustedImage = imadjust(smoothedImage, [28, 180]/255, [32, 255]/255);
    
    % Setting New ROI
    roiCoordinates = [270, 1255, 260, 230];
  
    % Create a binary mask for new ROI
    mask = zeros(size(adjustedImage, 1), size(adjustedImage, 2));
    mask(roiCoordinates(2):roiCoordinates(2)+roiCoordinates(4)-1, roiCoordinates(1):roiCoordinates(1)+roiCoordinates(3)-1) = 1;

    % Contrast enhancement at masked region
    highContrastImage = adjustedImage;
    highContrastImage(mask == 1) = imadjust(adjustedImage(mask == 1), [0.4, 1.0], [0.2, 1.0]);
    
%     % Bounding Box
%     targetRegion = imcrop(highContrastImage, roiCoordinates);
%     targetStats = regionprops(targetRegion, 'BoundingBox');
% 
%     % Check the aspect ratio for the target region
%     aspectRatio = targetStats.BoundingBox(3) / targetStats.BoundingBox(4);
% 
%     % Check if the aspect ratio resembles the number 10
%     if aspectRatio > 0.8 && aspectRatio < 1.2
%         disp('Region resembles the number 10.');
%     else
%         disp('Wrong');
%     end

    % Define OCR to Extract Number
    ocrResult = ocr(highContrastImage, [270, 1255, 260, 230], 'TextLayout', 'Block', 'CharacterSet', '01');
    ocrResult = (strrep(ocrResult.Text, ' ', ''));
end

function [oriroiImage, highContrastImage, ocrResult] = IsTwenty(image)
    img = image;
    newSize = [1696, 3560];
    img = imresize(img, newSize);
    
    centerX = size(img, 2) / 2;
    % Define the width and height of the region of interest (ROI)
    roiWidth = 670;  % Adjusting Width
    roiHeight = size(img, 1);  % Use full height
    
    % Set the offset from the left edge
    leftOffset = 1260;  

    % Calculate the starting point for the crop based on the offset
    roiX = round(centerX - roiWidth / 2) - leftOffset;

    % Ensure the ROI is within the image boundaries
    roiX = max(1, roiX);

    % Extract the region of interest from the original image
    oriroiImage = imcrop(img, [roiX, 1, roiWidth, roiHeight]);
    roiImage = im2gray(oriroiImage);
    
    %Brightness Enhancement

    brightenedImage = imadjust(roiImage, [0.6, 0.9], [0.1, 1]);
    %brightenedImage1 = histeq(roiImage);
    
    % Noise Removal
    
    sigma = 6.0;  % Adjust standard deviation 
    smoothedImage = imgaussfilt(brightenedImage, sigma);
    
    
    %smoothedImage = medfilt2(brightenedImage, [5, 5]);
    
    % Contrast Adjustment
    adjustedImage = imadjust(smoothedImage, [146, 228]/255, [26, 255]/255);
    
    
    % Setting new ROI
    roiCoordinates = [267, 1250, 220, 180];
   
    
    % Create a binary mask for new ROI
    mask = zeros(size(adjustedImage, 1), size(adjustedImage, 2));
    mask(roiCoordinates(2):roiCoordinates(2)+roiCoordinates(4)-1, roiCoordinates(1):roiCoordinates(1)+roiCoordinates(3)-1) = 1;
    
    
    % Applying contrast on the masked region
    highContrastImage = adjustedImage;
    highContrastImage(mask == 1) = imadjust(adjustedImage(mask == 1), [0.6, 1.0], [0.1, 1.0]);
    
    
    % Defining OCR
    ocrResult = ocr(highContrastImage, [267, 1250, 220, 180], 'TextLayout', 'Block', 'CharacterSet', '012');
    ocrResult = (strrep(ocrResult.Text, ' ', ''));
    
end


function [oriroiImage, highContrastImage, ocrResult] = IsFifty(image)
    %Resizing Image to common size
    img = image;
    newSize = [1696, 3560];
    img = imresize(img, newSize);
    
    % CROPPING REGION OF INTEREST
    
    centerX = size(img, 2) / 2;
    roiWidth = 670;  
    roiHeight = size(img, 1);  
    leftOffset = 1260;  
    roiX = round(centerX - roiWidth / 2) - leftOffset;
    roiX = max(1, roiX);
    oriroiImage = imcrop(img, [roiX, 1, roiWidth, roiHeight]);
    roiImage = im2gray(oriroiImage);
    
    %Brightness Enhancement
    brightenedImage = imadjust(roiImage, [0.6, 0.9], [0.1, 1.0]);
   
    % Noise Removal
    sigma = 6.0;
    smoothedImage = imgaussfilt(brightenedImage, sigma);
    %smoothedImage = medfilt2(brightenedImage, [5, 5]);
    
    % Contrast Adjustment
    adjustedImage = imadjust(smoothedImage, [146, 228]/255, [26, 255]/255);
  
    % Setting new ROI
    roiCoordinates = [210, 1280, 240, 180];
    
    % Create a binary mask for new ROI
    mask = zeros(size(adjustedImage, 1), size(adjustedImage, 2));
    mask(roiCoordinates(2):roiCoordinates(2)+roiCoordinates(4)-1, roiCoordinates(1):roiCoordinates(1)+roiCoordinates(3)-1) = 1;
    
    % Applying contrast on the masked region
    highContrastImage = adjustedImage;
    highContrastImage(mask == 1) = imadjust(adjustedImage(mask == 1), [0.76, 1.0], [0.3, 0.9]);
    binaryImage = im2bw(highContrastImage, 0.7);
    binaryImage(mask == 0) = 0;  % Set pixels outside the mask to 0
    
    % Defining OCR
    ocrResult = ocr(binaryImage, [210, 1280, 240, 180], 'TextLayout', 'Block', 'CharacterSet', '015');
    ocrResult = (strrep(ocrResult.Text, ' ', ''));
end



function [oriroiImage, highContrastImage, ocrResult] = IsHundred(image)
    %Resizing Image to common size
    img = image;
    newSize = [1696, 3560];
    img = imresize(img, newSize);


    % CROPPING REGION OF INTEREST

    % Calculate the center of the image
    centerX = size(img, 2) / 2;

    roiWidth = 670; 
    roiHeight = size(img, 1); 
    leftOffset = 1260;
    roiX = round(centerX - roiWidth / 2) - leftOffset;
    roiX = max(1, roiX);
    oriroiImage = imcrop(img, [roiX, 1, roiWidth, roiHeight]);
 
    roiImage = im2gray(oriroiImage);

    %Brightness Enhancement
    brightenedImage = imadjust(roiImage, [0.7, 1.0], [0.1, 1.0]);

    % Noise Removal
    sigma = 6.0; 
    smoothedImage = imgaussfilt(brightenedImage, sigma);

    %smoothedImage = medfilt2(brightenedImage, [5, 5]);

    % Contrast Adjustment
    adjustedImage = imadjust(smoothedImage, [60, 200]/255, [32, 255]/255);
    
    % Setting new ROI
    roiCoordinates = [150, 1230, 270, 160];
    % roiCoordinates = [190, 1205, 270, 160]; for v1_test

    % Create a binary mask for new ROI
    mask = zeros(size(adjustedImage, 1), size(adjustedImage, 2));
    mask(roiCoordinates(2):roiCoordinates(2)+roiCoordinates(4)-1, roiCoordinates(1):roiCoordinates(1)+roiCoordinates(3)-1) = 1;


    % Applying contrast on the masked region
    highContrastImage = adjustedImage;
    highContrastImage(mask == 1) = imadjust(adjustedImage(mask == 1), [0.2, 0.8], [0.1, 1.0]);
    % highContrastImage(mask == 1) = imadjust(adjustedImage(mask == 1),
    % [0.6,1.0], [0.1, 1.0]); for v1_test

    % Defining OCR
    ocrResult = ocr(highContrastImage, [150, 1230, 270, 160], 'TextLayout', 'Block', 'CharacterSet', '01');
    ocrResult = (strrep(ocrResult.Text, ' ', ''));
end


% -----Feature 3-----
function [feature3, Check3] = componentDetection(templateImage, targetImage)

    % Convert images to grayscale as SURF works on single channel images
    templateImageGray = rgb2gray(templateImage);
    targetImageGray = rgb2gray(targetImage);

    % Denoise and sharpen the images
    templateImageGray = preprocessImage(templateImageGray);
    targetImageGray = preprocessImage(targetImageGray);

    % Detect SURF features for both images
    pointsTemplate = detectSURFFeatures(templateImageGray, 'MetricThreshold', 500);
    pointsTarget = detectSURFFeatures(targetImageGray, 'MetricThreshold', 500);

    % Extract feature descriptors
    [featuresTemplate, validPointsTemplate] = extractFeatures(templateImageGray, pointsTemplate);
    [featuresTarget, validPointsTarget] = extractFeatures(targetImageGray, pointsTarget);

    % Match features between template and target image
    indexPairs = matchFeatures(featuresTemplate, featuresTarget, 'MaxRatio', 0.7);

    if isempty(indexPairs)
        % No matches found, component is not detected
        feature3 = 'Fake: Component Not Detected';
        imshow(targetImage);
        title(feature3);
        return;
    end

    % Retrieve locations of corresponding points for each image
    matchedPoints1 = validPointsTemplate(indexPairs(:, 1));
    matchedPoints2 = validPointsTarget(indexPairs(:, 2));

    % Estimate the geometric transform between the template and the target image
    [tform, ~] = estimateGeometricTransform(matchedPoints1, matchedPoints2, 'affine', 'MaxDistance', 1.5);

    % Find the bounding box of the recovered template in the target image
    transformedCorners = transformPointsForward(tform, [0, 0; size(templateImageGray, 2), 0; ...
        size(templateImageGray, 2), size(templateImageGray, 1); ...
        0, size(templateImageGray, 1)]);
    minX = min(transformedCorners(:, 1));
    minY = min(transformedCorners(:, 2));
    maxX = max(transformedCorners(:, 1));
    maxY = max(transformedCorners(:, 2));
    width = maxX - minX;
    height = maxY - minY;
    boundingBoxTemplate = [minX, minY, width, height];

    % Crop the region from the target image
    croppedImage = imcrop(targetImage, boundingBoxTemplate);

    % Find the bounding box of the detected component in the target image
    boundingBoxDetected = [min(matchedPoints2.Location(:, 1)), min(matchedPoints2.Location(:, 2)), ...
        max(matchedPoints2.Location(:, 1)) - min(matchedPoints2.Location(:, 1)), ...
        max(matchedPoints2.Location(:, 2)) - min(matchedPoints2.Location(:, 2))];

    % Display matched features along with the transformed bounding box
    figure;

    subplot(1, 2, 1);
    imshow(templateImage);
    title('Original Image');

    subplot(1, 2, 2);
    imshow(targetImage);
    hold on;
    plot(matchedPoints2.Location(:, 1), matchedPoints2.Location(:, 2), 'go');
    rectangle('Position', boundingBoxDetected, 'EdgeColor', 'y');
    hold off;
    
    % Check if Feature 3 passed or failed
    if compareBoundingBoxes(boundingBoxTemplate, boundingBoxDetected, 0.8)
        feature3 = 'Feature 3 PASSED'; 
    else
        feature3 = 'Feature 3 FAILED';
    end
    
    title('Component to be Detected');
    sgtitle(feature3);

    % Compare the bounding boxes
    similarityThreshold = 0.8; % Adjust the threshold as needed
    isReal = compareBoundingBoxes(boundingBoxTemplate, boundingBoxDetected, similarityThreshold);

    % Display result message
    if isReal
        feature3 = 'Real: Component Detected in Target Image';
        Check3 = 1;
    else
        feature3 = 'Fake: Component Shape Not Similar';
        Check3 = 0;
    end
end

function preprocessedImg = preprocessImage(inputImg)
    % Denoise using a Gaussian filter
    denoisedImg = imgaussfilt(inputImg, 1);

    % Sharpen using a basic filter
    sharpeningFilter = [0, -1, 0; -1, 5, -1; 0, -1, 0];
    sharpenedImg = imfilter(denoisedImg, sharpeningFilter);

    preprocessedImg = inputImg + sharpenedImg;
end

function isReal = compareBoundingBoxes(box1, box2, threshold)
    % Compare bounding boxes based on area similarity
    area1 = box1(3) * box1(4);
    area2 = box2(3) * box2(4);
    
    overlapArea = rectint(box1, box2);
    overlapRatio = overlapArea / min(area1, area2);

    % Compare the overlap ratio with the threshold
    isReal = overlapRatio >=threshold;
end
