function [Io, C,tot_idx,Iall] = imbin(I,K)
%       Colour quantisation based on image covariance eigen values.
%
%       IMBIN utilises a binary tree approach to store all statistical properties and
%       image resulting clusters.
%       The resulting image stored (Io) is optimised to be from the
%       cluster that fulfils the condition of both max eigenvector and
%       max sum of eigenvectors values
%
% Syntax:  [Iout,Clusters] = imbin(Iinput,K)
%
% Inputs:
%    Iinput - an RGB image matrix
%    K - number of clusters to be inferred
%
% Outputs:
%    Iout - the output image result RGB array
%    C - an array results for all clusters
%
% Example:
%    I1=imread('hull10.jpg');
%    figure, imshow(I1)
%    K = 7;
%    I2 = bin(I1,K);
%    figure, imshow(I2)
%
% Reference: Michael T. Orchard, Charles A. Bouman (1991). Color quantization
% of images, IEEE Trans. Signal Processing 39(12): 2677-2690
%
%       Other m-files required: none
%       Subfunctions: none
%       MAT / data files required: none
%
%       See also: GRAYTHRESH, IM2BW, IMTHRESH
% 
%% Initializations
narginchk(1,2)
nargoutchk(1,4)

% check number of clusters (pointless to have less than 2)
if nargin==1
    K = 2;
end

I=im2double(I); % to avoid numerical issues in matlab

% Get image dimensions
[rows, cols, ~] = size(I);

% Treat complete image as single cluster and compute different fields of cluster.
R1=zeros(3,3);

% Total number of pixels
N1 = rows * cols;

si=0; % index used later for loops

% Initialize matrix to store abs median eigenvalue
val = zeros(1, 7);

% Initialize cell to store nodes & its corresponding properties
C = {1, K};

%% Evaluate nodes & its statistical properties
for i=1:K
    
    %% Calculate first node
    if i == 1
        % Evaluate total of each colour channel
        for j=1:3
            for k=1:3
                R1(j, k) = sum(sum(I(:,:,j) .* I(:,:,k)));
            end
        end
        
        % Flatten each colour channel
        I_Comp = reshape(I, [], 3);
        
        % Sum of each colour channel
        M1 = reshape(sum(sum(I)), [1, 3]);
        
        % Compute covariance of the Image.
        R1_bar = R1-((M1*M1')/N1);
        
        % Compute eigen values and eigen vectors of the image.
        [V,D] = eig(R1_bar);
        
        % Points of first node
        point = 1:N1;
        
        % Form the initial cluster.
        cluster.R = R1;
        cluster.M = M1;
        cluster.N = N1;
        cluster.point = point;
        % Store only center column right eigenvectors
        cluster.vector = V(:,2);
        
        % Place the cluster in Cell array.
        C{1}=cluster;
        % Store the absolute median eigenvalue
        val(1)=abs(D(2,2));
    end
    
    %% Evaluate remaining nodes
    if(max(val(:))==-1) && i > 1
        break;
    end
        
  % Pick the cluster having highest eigen value and split it into two parts.
  % place max value in index
  index = find(max(val(:))== val);

  % Current node's center column right eigenvectors
  vector = C{index}.vector;

  % Current node's points (regions of interest)
  point = C{index}.point;
  
  % Flatten points
  point = point(:);

  % 
  M = C{index}.M;
  N = C{index}.N;
  % Quantization (mean) value of current cluster
  Q = M / N;

  R = C{index}.R;

  a = vector(1,1)* I_Comp(point(:), 1) + vector(2,1)* I_Comp(point(:), 2) + vector(3,1)* I_Comp(point(:), 3);

  % Pixels of the cluster are divided into two parts based on their closeness
  % to a plane perpendicular to principal eigen vector and passing through mean.
  point1 = point(a >= vector'*Q');
  point2 = point(a < vector'*Q');
  
  % Compute various statistical value for the newly formed clusters.
  for j=1:3
      for k=1:3
          R1(j, k) = sum(I_Comp(point1(:), j) .* I_Comp(point1(:), k));
      end
  end

  M1 = sum(I_Comp(point1(:), :), 1);
 
  R2=R-R1;
  M2=M-M1;

  N1 = size(point1, 1);
  N2 = size(point2, 1);
  % Place the clusters into cell array and delete the parent cluster.
  if(N1>0 && N2>0)
      % Evaluate statistical properties for node1
      cluster1.point=point1;
      cluster1.R=R1;
      cluster1.M=M1;
      cluster1.N=N1;
      R1_bar=R1-((M1*M1')/N1);
      [V1 , D1]=eig(R1_bar);
      cluster1.vector=V1(:,2);
      val(index)=abs(D1(2,2));
      C{index}=cluster1;
      
      % Evaluate statistical properties for node2
      cluster2.R=R2;
      cluster2.N=N2;
      cluster2.M=M2;
      cluster2.point=point2;
      R2_bar=R2-((M2*M2')/N2);
      [V2 , D2]=eig(R2_bar);
      cluster2.vector=V2(:,2);
      val(si+1)=abs(D2(2,2));
      C{si+1}=cluster2;
      
      si=si+1;
  else
     val(index)=-1;
     i=i-1;
  end
end

%% Select desired node based on eigenvalues
% Condition to check the eigen values and the sums of the produced clusters
total = zeros(1, 4);
median = zeros(1, 4);
for i = 4:si
    % finds position of cluster with max mid eigen value
    % and max sum of eigen values
    vector = C{i}.vector;
    vector_tot{i} = C{i}.vector;
    total(i-3) = sum(vector);
    median(i-3) = vector(2,1);
    % Find index of max value of total eigenvalues
    ind2 = find(max(total(1:i-3))== total(1:i-3));
    % Find index of max value of median eigenvalues
    ind3 = find(max(median(1:i-3))== median(1:i-3));
    % Index of max value of total set to -1 if not equal indices
    if ind2 ~= ind3
        total(i-3) = -1;
    end
    
    % All nodes
    temp=C{i}.point;
    I1=zeros(size(I,1),size(I,2),3);
    [y_p, x_p]=ind2sub([size(I,1) size(I,2)],temp);
    for t=1:size(y_p,1)
        y=y_p(t);
        x=x_p(t);
        I1(y,x,1)=I(y,x,1);
        I1(y,x,2)=I(y,x,2);
        I1(y,x,3)=I(y,x,3);
    end
    I1 = im2uint8(I1);
    Iall{i} = I1;
end

% Find index of max values
tot_idx = find(max(total(:))== total); % Correct node
med_idx = find(max(median(:))== median);

% Safeguard index shapes
tot_idx = [tot_idx zeros(1, numel(med_idx) - numel(tot_idx))];
med_idx = [med_idx zeros(1, numel(tot_idx) - numel(med_idx))];
% condition to select the appropriate cluster with both max eigenvector
% and max sum of eigenvectors values
Io=zeros(rows, cols, 3);
if (tot_idx == med_idx) 
    fpoint = C{tot_idx+3}.point;
    [y_p, x_p]=ind2sub([rows cols],fpoint);
    for s=1:size(y_p,1)
        y=y_p(s);
        x=x_p(s);
        Io(y,x,1)=I(y,x,1);
        Io(y,x,2)=I(y,x,2);
        Io(y,x,3)=I(y,x,3);
    end
end

Io = im2uint8(Io); % back to integer values 0-255 (image array)