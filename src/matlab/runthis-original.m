close all, clear variables

I1=imread('hull6.jpg');

K = 7; %produce seven clusters
[I2,C] = imbin(I1,K);

N=3; %produce four classes
idx = imthresh(I2,N);

mask2 = (idx(:,:) == 2); %returns logical array of Class 2

I3(:,:,1) = uint8(I2(:,:,1)) .* uint8(mask2);
I3(:,:,2) = uint8(I2(:,:,2)) .* uint8(mask2);
I3(:,:,3) = uint8(I2(:,:,3)) .* uint8(mask2);

figure
subplot(1,3,1); imshow(I1); title('Original Image')
subplot(1,3,2); imshow(I2); title('Quantised Image')
subplot(1,3,3); imshow(I3); title('Thresholded Image')

% [B,L,N] = bwboundaries(mask2,'noholes');
% figure
% imshow(I1)
% hold on
% for k = 1:length(B)
%     boundary = B{k};
%     plot(boundary(:,2), boundary(:,1), 'k', 'LineWidth', 4)
% end