clc, close all,clear all, clear variables

% Specify the folder where the files live, change read_folder to a desired folder
read_folder = '.\dataset\mob_data';

% Check to make sure that folder actually exists.  Warn user if it doesn't
if ~isfolder(read_folder) 
        errorMessage = sprintf('Error: The following folder does not exist:\n%s\nPlease specify a new folder.', read_folder);
        uiwait(warndlg(errorMessage));
        read_folder = uigetdir(); % Ask for a new one
        
        % User clicked Cancel
        if read_folder== 0
                return;
        end
end

% Get a list of all files in the folder_two with the desired file name pattern
filePattern = fullfile(read_folder, '*.jpg');
theFiles = dir(filePattern);

for k = 1 : length(theFiles)
    baseFileName = theFiles(k).name;
    fullFileName = fullfile(theFiles(k).folder, baseFileName);
    I1 = imread(fullFileName);
 
    % imbin
    K = 7;
    [I2,C,tot_idx,Iall] = imbin(I1,K); % Call imbin
   
    % I2  Best node
    I4 = cell2mat(Iall(1,4)); % node 4
    I5 = cell2mat(Iall(1,5)); % node 5
    I6 = cell2mat(Iall(1,6)); % node 6
    I7 = cell2mat(Iall(1,7)); % node 7

    % imthresh 
    N = 4; % Produce four classes
    % Call imthresh
    idx2 = imthresh(I2,N); 
    idx4 = imthresh(I4,N); 
    idx5 = imthresh(I5,N); 
    idx6 = imthresh(I6,N); 
    idx7 = imthresh(I7,N); 
    
    % Returns logical array of Class 2
    mask2 = (idx2(:,:) == 2);
    mask4 = (idx4(:,:) == 2);
    mask5 = (idx5(:,:) == 2);
    mask6 = (idx6(:,:) == 2);
    mask7 = (idx7(:,:) == 2);
    
    % Extract the individual red, green, and blue color channels
    I2(:,:,1) = uint8(I2(:,:,1)) .* uint8(mask2);
    I2(:,:,2) = uint8(I2(:,:,2)) .* uint8(mask2);
    I2(:,:,3) = uint8(I2(:,:,3)) .* uint8(mask2);
    
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
    
    % Extract the individual red, green, and blue color channels and threshold them
    binaryR2 = I2(:, :, 1) > 0;
    binaryG2 = I2(:, :, 2) > 0;
    binaryB2 = I2(:, :, 3) > 0;
    
    binaryR4 = I4(:, :, 1) > 0;
    binaryG4 = I4(:, :, 2) > 0;
    binaryB4 = I4(:, :, 3) > 0;
    
    binaryR5 = I5(:, :, 1) > 0;
    binaryG5 = I5(:, :, 2) > 0;
    binaryB5 = I5(:, :, 3) > 0;
    
    binaryR6 = I6(:, :, 1) > 0;
    binaryG6 = I6(:, :, 2) > 0;
    binaryB6 = I6(:, :, 3) > 0;
    
    binaryR7 = I7(:, :, 1) > 0;
    binaryG7 = I7(:, :, 2) > 0;
    binaryB7 = I7(:, :, 3) > 0;
    
    % AND the binary images together to find out where all are > 0
    binaryImage2 = binaryR2 & binaryG2 & binaryB2;
    
    binaryImage4 = binaryR4 & binaryG4 & binaryB4;
    binaryImage5 = binaryR5 & binaryG5 & binaryB5;
    binaryImage6 = binaryR6 & binaryG6 & binaryB6;
    binaryImage7 = binaryR7 & binaryG7 & binaryB7;
    
    % Count the number of pixels where it's true that all 3 are > 0
    pixelCount2 = sum(binaryImage2(:));
    
    pixelCount4 = sum(binaryImage4(:));
    pixelCount5 = sum(binaryImage5(:));
    pixelCount6 = sum(binaryImage6(:));
    pixelCount7 = sum(binaryImage7(:));
    
    % Calculate values
    node_sizes = [pixelCount4, pixelCount5, pixelCount6, pixelCount7];
    index = find( min(node_sizes) == node_sizes );
    
    % Store values    
    predicted(k) = tot_idx;
    actual(k) = index;
end

figure(1);cm = confusionchart(actual,predicted);
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
cm.Title = 'Confusion Matrix';