%%%%%%  INTERACTIVE PROSTATE SEGMENTATION IN 2D  %%%%%%%%%
%%% Authors: Carmen Moreno Genis, Katherine Sheran, Benjamin Chatain
%%% Master in Medical Imaging and Applications (MAIA)

% Based on the article: 
%%% "Dual optimization based prostate zonal segmentation in 3D MR images".
%%% by Wu Qiu, Jing Yuan, Eranga Ukwatta, Yue Sun, Martin Rajchl,Aaron Fenser

% This program receives a single 2D T2w MR image and allows the user 
% to segment the image in 2 sections by initializng 10-12 points.
% The selected regions are saved as the variable 'scribbles' 
% and are then exported as a .mat file called 'prostate_labels.mat' for
% further segmentation in the "MaxFlowProstateSegmentation2D".m  file.

%----------------------------------------------------------------------
% Start cleaning everything
%----------------------------------------------------------------------
clear       %no variables
close all   %no figures
clc         %empty command window
%----------------------------------------------------------------------
%% Loading DICOM file containing the image of a prostate
%----------------------------------------------------------------------
img = dicomread ('IM-0001-0038'); % reading dicom image 

img = mat2gray(img); % converting to greyscale image

%----------------------------------------------------------------------
%% Initializing 10-12 points
%----------------------------------------------------------------------
%%%%% Extracting the central gland with roipoly
% The roipoly function returns a binary image that can be used as a mask
% The user should colocate 10-12 points areound the whole prostate gland.
% The background should be ignored.
[B1, x1, y1] = roipoly(img); 
whole_prostate_mask = roipoly (B1, x1, y1); 
whole_prostate_image = img.* whole_prostate_mask;

background = img - whole_prostate_image;
figure 
imshow (background);
title('Background');

% Displaying the results of original image and extraction of whole prostate gland
figure
imshow (whole_prostate_image);
title('Whole prostate gland');

%%%%% Extracting the central gland with roipoly
% The user colocates 10-12 points around the peripheral zone of the
% The central zone of the gland must be ignored.
[B2, x2, y2] = roipoly(whole_prostate_image);
central_gland_mask = roipoly (B2, x2, y2);
central_gland_image = img.* central_gland_mask; 

figure
imshow (central_gland_image);
title('Central gland');

%%%% Extracting the peripheral zone substracting whole gland-central gland
peripheral_zone = whole_prostate_image - central_gland_image;
peripheral_zone_mask = B1 - B2;
figure 
imshow (peripheral_zone);
title('Peripheral zone');

%% Obtaining PDFs
%----------------------------------------------------------------------
% PDF of whole prostate
[f1, x1] = ksdensity(img(B1)) 

%PDF of central gland
[f2, x2] = ksdensity(img(B2)) 

%PDF of background
logi_background = logical(background)
[f3, x3] = ksdensity( img (logi_background) ) 


%PDF of peripheral zone
logi_peripheral = logical(peripheral_zone)
[f4, x4] = ksdensity( img (logi_peripheral) ) 


%% Plot all PDFS in one graph 
figure()
plot (x1, f1,'b', x2, f2, 'g', x3, f3,'m', x4, f4, 'r')
title('PDF models of different regions in a 3D prostate T2w MR image')
legend('WG', 'CG', 'BG', 'PZ')

%% SAVE IMAGES IN .MAT FILE 
scribbles = ( B1 + B2 ); 

Original_Im = img;

Im = Original_Im .* scribbles;

save('ProstateLabels.mat', 'Im', 'Original_Im', 'scribbles');











