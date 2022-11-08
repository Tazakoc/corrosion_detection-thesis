clc, close all,clear all, clear variables

%% Read image
% Read-Show image
[img, folder] = uigetfile('.\dataset\hull.jpg');

% User clicked the Cancel button.
if img == 0
      
      return;
end

I1 = fullfile(folder, img); 
I1 = imread(I1);

% Image Tool
% imtool(I1)

%% Draw - Edit image
%Show image and draw the interest point
question = menu('Do you want to draw region of interest?', 'Yes', 'No');

if question == 1
    
    imshow(I1) 
    roi = drawfreehand('Color','k'); % pick ROI
    roi_mask = uint8(roi.createMask()); % get BW mask for that ROI
    I1 = I1 .* roi_mask; % apply mask

    % get BB
    pos = roi.Position;
    x1 =  round(min(pos(:,2)));
    y1 =  round(min(pos(:,1)));
    x2 =  round(max(pos(:,2)));
    y2 =  round(max(pos(:,1)));
    
    I1 = I1(x1:x2, y1:y2, :);
end

% Extract the individual red, green, and blue color channels.
redChannel = I1(:, :, 1); % Red channel
greenChannel = I1(:, :, 2); % Green channel
blueChannel = I1(:, :, 3); % Blue channel

% Increase Channels
Red = 0;
Green = 0; 
Blue = 0;
    
redChannel = uint8(double(redChannel) + Red);
greenChannel = uint8(double(greenChannel) + Green);
blueChannel = uint8(double(blueChannel) + Blue);

% Create an all black channel.
allBlack = zeros(size(I1, 1), size(I1, 2), 'uint8');

% Create color versions of the individual color channels.
just_red = cat(3, redChannel, allBlack, allBlack);
just_green = cat(3, allBlack, greenChannel, allBlack);
just_blue = cat(3, allBlack, allBlack, blueChannel);

% Recombine separate color channels into a single, true color RGB image.
I1_edited = cat(3, redChannel, greenChannel, blueChannel);

% Increasing brightness/contrast
% low = 0.3;
% high = 0.4;
% I1_edited = imadjust(I1,[low high],[]); % I is double % Contrast
% I1 = I1_edited + 0; % Brightness


%% Functions
% imbin
K = 7; % produce seven clusters
[I2,C,tot_idx,Iall] = imbin(I1,K); % Call imbin

I4 = cell2mat(Iall(1,4)); % Node 4
I5 = cell2mat(Iall(1,5)); % Node 5
I6 = cell2mat(Iall(1,6)); % Node 6
I7 = cell2mat(Iall(1,7)); % Node 7

figure(7)
title('All nodes C{n4-n7}[Quantised]');
subplot(4,1,1); imshow(I4); title('4th node [n4]')
subplot(4,1,2); imshow(I5); title('5th node [n5]')
subplot(4,1,3); imshow(I6); title('6th node [n6]')
subplot(4,1,4); imshow(I7); title('7th node [n7]')

% imthresh 
N = 4; % Produce four classes

% Call imthresh
idx = imthresh(I2,N); % Best Image

% Thresholding for nodes 4-7
idx4 = imthresh(I4,N); % Node 4
idx5 = imthresh(I5,N); % Node 5
idx6 = imthresh(I6,N); % Node 6
idx7 = imthresh(I7,N); % Node 7

% Extract the individual red, green, and blue color channels for best image.
mask2 = (idx(:,:) == 2); % Returns logical array of Class 2

% Extract the individual red, green, and blue color channels for nodes 4-7.
mask4 = (idx4(:,:) == 2); 
mask5 = (idx5(:,:) == 2); 
mask6 = (idx6(:,:) == 2); 
mask7 = (idx7(:,:) == 2); 

% Extract the individual red, green, and blue color channels.
% Best Image
I3(:,:,1) = uint8(I2(:,:,1)) .* uint8(mask2);
I3(:,:,2) = uint8(I2(:,:,2)) .* uint8(mask2);
I3(:,:,3) = uint8(I2(:,:,3)) .* uint8(mask2);

% Nodes 4-8
I4(:,:,1) = uint8(I4(:,:,1)) .* uint8(mask4);
I4(:,:,2) = uint8(I4(:,:,2)) .* uint8(mask4);
I4(:,:,3) = uint8(I4(:,:,3)) .* uint8(mask4);

I5(:,:,1) = uint8(I5(:,:,1)) .* uint8(mask5);
I5(:,:,2) = uint8(I5(:,:,2)) .* uint8(mask5);
I5(:,:,3) = uint8(I5(:,:,3)) .* uint8(mask5);

I6(:,:,1) = uint8(I6(:,:,1)) .* uint8(mask6);
I6(:,:,2) = uint8(I6(:,:,2)) .* uint8(mask6);
I6(:,:,3) = uint8(I6(:,:,3)) .* uint8(mask6);

I7(:,:,1) = uint8(I7(:,:,1)) .* uint8(mask7);
I7(:,:,2) = uint8(I7(:,:,2)) .* uint8(mask7);
I7(:,:,3) = uint8(I7(:,:,3)) .* uint8(mask7);

% Extract RGB values  of each image pixel row-wise.
I1_values = permute(I1, [3,2,1]);
serialValuesI1 = I1(:);

% Extract RGB values  of each image pixel row-wise.
I2_values = permute(I2, [3,2,1]);
serialValuesI3 = I2(:);

% Extract RGB values  of each image pixel row-wise.
I3_values = permute(I3, [3,2,1]);
serialValuesI3 = I3(:);

% Create color versions of the individual color channels.
% Quantised
redChannel_quantised = I2(:, :, 1); % Red channel
greenChannel_quantised = I2(:, :, 2); % Green channel
blueChannel_quantised = I2(:, :, 3); % Blue channel

% Thresholded
redChannel_thresh = I3(:, :, 1); % Red channel
greenChannel_thresh = I3(:, :, 2); % Green channel
blueChannel_thresh = I3(:, :, 3); % Blue channel

% Create an all black channel.
allBlack_quantised = zeros(size(I2, 1), size(I2, 2), 'uint8'); % Quantised
allBlack = zeros(size(I1, 1), size(I1, 2), 'uint8'); % Thresholded

% Create color versions of the individual color channels.
% Quantised
just_red_quantised = cat(3, redChannel_quantised, allBlack_quantised, allBlack_quantised);
just_green_quantised = cat(3, allBlack_quantised, greenChannel_quantised, allBlack_quantised);
just_blue_quantised = cat(3, allBlack_quantised, allBlack_quantised, blueChannel_quantised);

% Thresholded
just_red_thresh = cat(3, redChannel_thresh, allBlack, allBlack);
just_green_thresh = cat(3, allBlack, greenChannel_thresh, allBlack);
just_blue_thresh = cat(3, allBlack, allBlack, blueChannel_thresh);

%% Show Results
figure(1)
subplot(1,3,1); imshow(I1); title('Original Image')
subplot(1,3,2); imshow(I2); title('Quantised Image')
subplot(1,3,3); imshow(I3); title('Thresholded Image [Class 2]')

figure(2)
[B,L,N] = bwboundaries(mask2,'noholes');
imshow(I1)
hold on
for k = 1:length(B)
    
    boundary = B{k};
    plot(boundary(:,2), boundary(:,1), 'k', 'LineWidth', 4)
    
end

figure(3)
title('All nodes C{n4-n7} [Thresholded]');
subplot(4,1,1); imshow(I4); title('4th node [n4]')
subplot(4,1,2); imshow(I5); title('5th node [n5]')
subplot(4,1,3); imshow(I6); title('6th node [n6]')
subplot(4,1,4); imshow(I7); title('7th node [n7]')
%% Histogram of an Original image
figure(4)
subplot(4,2,1);imshow(just_red);title('Red Channel [Original]'); % Show specific channel
subplot(4,2,2);[red_data, pixel_level]=imhist(redChannel); % Plot Histogram
bar(pixel_level, red_data,'r');
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0, 1, 1]);
title('Histogram Red Channel[Original]');
xlabel('Pixel intensity(0-255)')
xlim([1 255])
ylabel('No. of Pixels')
grid on;

subplot(4,2,3);imshow(just_green);title('Green Channel [Original]');
subplot(4,2,4);[green_data, pixel_level]=imhist(greenChannel);
bar(pixel_level, green_data,'g');
title('Histogram Green Channel[Original]');
xlabel('Pixel intensity(0-255)')
xlim([1 255])
ylabel('No. of Pixels')
grid on;

subplot(4,2,5);imshow(just_blue);title('Blue Channel [Original]');
subplot(4,2,6);[blue_data, pixel_level]=imhist(blueChannel);
bar(pixel_level, blue_data,'b');
title('Histogram Blue Channel[Original]');
xlabel('Pixel intensity(0-255)')
xlim([1 255])
ylabel('No. of Pixels')
grid on;

subplot(4,2,[7,8]);
[red_data, pixel_level]=imhist(redChannel);
bar(pixel_level, red_data,'r');
hold on
[green_data, pixel_level]=imhist(greenChannel);
bar(pixel_level, green_data,'g');
hold on
[blue_data, pixel_level]=imhist(blueChannel);
bar(pixel_level, blue_data,'b');
title('Histogram RGB [Original]');
xlabel('Pixel intensity(0-255)');
xlim([1 255])
ylabel('No. of Pixels')
grid on;

%% Histogram of an Quantised image
figure(5)
subplot(4,2,1);imshow(just_red_quantised);title('Red Channel [Quantised]'); 
subplot(4,2,2);[red_data_quantised, pixel_level_quantised]=imhist(redChannel_quantised); 
bar(pixel_level_quantised, red_data_quantised,'r');
title('Histogram Red Channel[Quantised]');
xlabel('Pixel intensity(0-255)')
xlim([1 255])
ylabel('No. of Pixels')
ylim([0 1499])
grid on;

subplot(4,2,3);imshow(just_green_quantised);title('Green Channel [Quantised]');
subplot(4,2,4);[green_data_quantised, pixel_level_quantised]=imhist(greenChannel_quantised);
bar(pixel_level_quantised, green_data_quantised,'g');
title('Histogram Green Channel[Quantised]');
xlabel('Pixel intensity(0-255)')
xlim([1 255])
ylabel('No. of Pixels')
ylim([0 1500])
grid on;

subplot(4,2,5);imshow(just_blue_quantised);title('Blue Channel [Quantised]');
subplot(4,2,6);[blue_data_quantised, pixel_level_quantised]=imhist(blueChannel_quantised);
bar(pixel_level_quantised, blue_data_quantised,'b');
title('Histogram Blue Channel[Quantised]');
xlabel('Pixel intensity(0-255)')
xlim([1 255])
ylabel('No. of Pixels')
ylim([0 1500])
grid on;

subplot(4,2,[7,8]);
[red_data_quantised, pixel_level_quantised]=imhist(redChannel_quantised);
bar(pixel_level_quantised, red_data_quantised,'r');
hold on
[green_data_quantised, pixel_level_quantised]=imhist(greenChannel_quantised);
bar(pixel_level_quantised, green_data_quantised,'g');
hold on
[blue_data_quantised, pixel_level_quantised]=imhist(blueChannel_quantised);
bar(pixel_level_quantised, blue_data_quantised,'b');
title('Histogram RGB [Quantised]');
xlabel('Pixel intensity(0-255)')
xlim([1 255])
ylabel('No. of Pixels')
ylim([0 1499])
grid on;

%% Histogram of an Thresholded image
figure(6)
subplot(4,2,1);imshow(just_red_thresh);title('Red Channel [Thresholded]'); 
subplot(4,2,2);[red_data_thresh, pixel_level_thresh]=imhist(redChannel_thresh); 
bar(pixel_level_thresh, red_data_thresh,'r');
title('Histogram Red Channel[Thresholded]');
xlabel('Pixel intensity(0-255)')
xlim([1 255])
ylabel('No. of Pixels')
ylim([0 1499])
grid on;

subplot(4,2,3);imshow(just_green_thresh);title('Green Channel [Thresholded]');
subplot(4,2,4);[green_data_thresh, pixel_level_thresh]=imhist(greenChannel_thresh);
bar(pixel_level_thresh, green_data_thresh,'g');
title('Histogram Green Channel[Thresholded]');
xlabel('Pixel intensity(0-255)')
xlim([1 255])
ylabel('No. of Pixels')
ylim([0 1500])
grid on;

subplot(4,2,5);imshow(just_blue_thresh);title('Blue Channel [Thresholded]');
subplot(4,2,6);[blue_data_thresh, pixel_level_thresh]=imhist(blueChannel_thresh);
bar(pixel_level_thresh, blue_data_thresh,'b');
title('Histogram Blue Channel[Thresholded]');
xlabel('Pixel intensity(0-255)')
xlim([1 255])
ylabel('No. of Pixels')
ylim([0 1500])
grid on;

subplot(4,2,[7,8]);
[red_data_thresh, pixel_level_thresh]=imhist(redChannel_thresh);
bar(pixel_level_thresh, red_data_thresh,'r');
hold on
[green_data_thresh, pixel_level_thresh]=imhist(greenChannel_thresh);
bar(pixel_level_thresh, green_data_thresh,'g');
hold on
[blue_data_thresh, pixel_level_thresh]=imhist(blueChannel_thresh);
bar(pixel_level_thresh, blue_data_thresh,'b');
title('Histogram RGB [Thresholded]');
xlabel('Pixel intensity(0-255)')
xlim([1 255])
ylabel('No. of Pixels')
ylim([0 1499])
grid on;

%% Save Results
% question = menu('Do you want to save the Trhesholded Image?', 'Yes', 'No');
% if question == 1
%     
%     startingFolder = 'C:\Users\Tazakoc\OneDrive\AMC\Master\[Project]\Code\CorrosionDetection-master-edits\dataset\results_data';
%     defaultFileName = fullfile(startingFolder, '*.jpg');
%     [baseFileName, folder] = uiputfile(defaultFileName, '.jpg');
%    
%     % User clicked the Cancel button.
%     if baseFileName == 0
%       return;
%     end
%     
%     fullFileName = fullfile(folder, baseFileName);
%     imwrite(I3, fullFileName)
% end
% 
% if question == 2
%     fprintf('Aight \n')
% end