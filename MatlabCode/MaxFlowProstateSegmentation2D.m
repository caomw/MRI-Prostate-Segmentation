function [] = MaxFlowSegmentation2D()

%   A log likelihood cost of intensities sampled via user scribbles is used
%   to segment the background, whole gland, central zone and periferal zone 
%   from user initialized points of a prostate MRI image.

%%------------------------------------------------------------------------
% The input:  .mat file containting user`s predefined scribbles
%
% The output:
% (1) the original image 
% (2) segmentation of the background 
% (3) segmentation of the whole gland  
% (4) segmentation of the central zone 
% (4) segmentation of the peripheral zone 

%----------------------------------------------------------------------
% Start cleaning everything
%----------------------------------------------------------------------
close all;
clear all;


%----------------------------------------------------------------------
% load image and user`s previously initialized scribbles
%----------------------------------------------------------------------
load('ProstateLabels.mat', 'Im', 'Original_Im', 'scribbles');

% incrementation of the scribbles's values
scribbles = scribbles + 1;


%----------------------------------------------------------------------
% Initializing parameters
%----------------------------------------------------------------------
alpha1 = 0.025;     % alpha = penalty parameter to the total variation term.
[r, c] = size(Im);
labelIds = unique(scribbles(scribbles ~= 0));
numberOfLabels = length(labelIds); % 4 labels: backgnd, whole gland, central zone, peripheral zone

%----------------------------------------------------------------------
% Defining the Cost Functions 'Ct' for each label i
%----------------------------------------------------------------------
% allocate the sink links Ct(x)
Ct = zeros(r,c,  numberOfLabels);
alpha = alpha1.*ones(r, c,  numberOfLabels);

%----------------------------------------------------------------------
% Compute the likelihood from the probabilities to be in [0,1] 
%----------------------------------------------------------------------
% Set up an error bound at which we consider the solver converged 
epsilon = 1e-10;

% Since this is a multi-label graph, where the source flows ps(x) are unconstrained,
% we define our sink capacities Ct(x,l) for each label l:
for i=1:numberOfLabels
    Ct(:,:,i) = computeLogLikelihoodCost(Im, scribbles == i, epsilon);   
end


%----------------------------------------------------------------------
% Performing max flow optimization (Potts Model) 
%----------------------------------------------------------------------
% Setting parameters for calling the Potts Model Function in 2D
params = [r; c;  numberOfLabels; 200; 1e-11; 0.25; 0.11];

% Call the Potts Model max flow optimizer Ct(x,l), alpha(x) and pars to obtain
% the continuous labelling function u(x,l).
[u, erriter, i, timet] = Potts2D(Ct, alpha, params);


% Discretizing the continuous labels
[uu, I] = max(u, [], 3);

%----------------------------------------------------------------------
% Visualizing the results
%----------------------------------------------------------------------
figure();
subplot(1,5,1); imshow (Original_Im, []);  title('Original image');
subplot(1,5,2); imshow(squeeze(I(:,:)),[1 numberOfLabels]); title('Whole gland');
subplot(1,5,3); imshow(squeeze(u(:,:,1)),[0 1]); title('Background');
subplot(1,5,4); imshow(squeeze(u(:,:,2)),[0 1]); title('Central zone');
subplot(1,5,5); imshow(squeeze(u(:,:,3)),[0 1]); title('Peripheral zone');


end
