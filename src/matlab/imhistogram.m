clc, close all,clear all, clear variables

% Specify the folder where the files live, change read_folder to a desired folder
read_folder = '.\dataset\mob_data';

% Check to make sure that folder actually exists.  Warn user if it doesn't.
if ~isfolder(read_folder) 
        errorMessage = sprintf('Error: The following folder does not exist:\n%s\nPlease specify a new folder.', read_folder);
        uiwait(warndlg(errorMessage));
        read_folder = uigetdir(); % Ask for a new one.
        
        % User clicked Cancel
        if read_folder== 0
                return;
        end
end


% Get a list of all files in the folder_two with the desired file name pattern.
filePattern = fullfile(read_folder, '*.jpg');
theFiles = dir(filePattern);

for k = 1 : length(theFiles)
    baseFileName = theFiles(k).name;
    fullFileName = fullfile(theFiles(k).folder, baseFileName);
    I1 = imread(fullFileName);
 
    % imbin
    K = 7;
    [I2,C,tot_idx,Iall] = imbin(I1,K); % Call imbin
   
    % imthresh 
    N = 3; % Produce four classes
    idx2 = imthresh(I2,N); % Call imthresh

    mask2 = (idx2(:,:) == 2); % Returns logical array of Class 2

    % Extract the individual red, green, and blue color channels.
    I2(:,:,1) = uint8(I2(:,:,1)) .* uint8(mask2);
    I2(:,:,2) = uint8(I2(:,:,2)) .* uint8(mask2);
    I2(:,:,3) = uint8(I2(:,:,3)) .* uint8(mask2);

    % Create color versions of the individual color channels.
    redChannel_quantised = I2(:, :, 1); % Red channel
    greenChannel_quantised = I2(:, :, 2); % Green channel
    blueChannel_quantised = I2(:, :, 3); % Blue channel
    
    % Create an all black channel.
    allBlack_quantised = zeros(size(I2, 1), size(I2, 2), 'uint8');

    % Create color versions of the individual color channels.
    just_red_quantised = cat(3, redChannel_quantised, allBlack_quantised, allBlack_quantised);
    just_green_quantised = cat(3, allBlack_quantised, greenChannel_quantised, allBlack_quantised);
    just_blue_quantised = cat(3, allBlack_quantised, allBlack_quantised, blueChannel_quantised);

    %% Histogram of an Quantised image
    figure(1)
    subplot(4,1,1);[red_data_quantised, pixel_level_quantised]=imhist(redChannel_quantised); 
    bar(pixel_level_quantised, red_data_quantised,'r');
    title('Histogram Red Channel[Quantised]');
    xlabel('Pixel intensity(0-255)')
    xlim([1 254])
    ylabel('No. of Pixels')
    grid on;
    hold on;
    
    subplot(4,1,2);[green_data_quantised, pixel_level_quantised]=imhist(greenChannel_quantised);
    bar(pixel_level_quantised, green_data_quantised,'g');
    title('Histogram Green Channel[Quantised]');
    xlabel('Pixel intensity(0-255)')
    xlim([1 254])
    ylabel('No. of Pixels')
    grid on;
    hold on;
    
    subplot(4,1,3);[blue_data_quantised, pixel_level_quantised]=imhist(blueChannel_quantised);
    bar(pixel_level_quantised, blue_data_quantised,'b');
    title('Histogram Blue Channel[Quantised]');
    xlabel('Pixel intensity(0-255)')
    xlim([1 254])
    ylabel('No. of Pixels')
    grid on;
    hold on;
    
end
