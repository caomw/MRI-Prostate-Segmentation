function [cost] = computeLogLikelihoodCost(img, lbl, epsilon)

% The function computes the log likelihood from any pixel 
% to correspond to a certain predefined label or region of the image. 
% It returns 0 when the pixel does not correspond to the PDF of the region 
% and returns 1 when the pixel is inside the PDF of the region.

img = single(img);

nBins = 256;

minI = min(img(:));
maxI = max(img(:));

% normalize image to 8 bit
img_n = ((img - minI)/ (maxI - minI)).*255.0;

% compute histogram
[binCounts] = histc(img_n(lbl == 1),linspace(0,255,nBins));


% normalize to compute the probabilities
binCounts = binCounts./sum(binCounts(:));

% compute LL
P = binCounts( uint16(img_n/ (256/nBins)) + 1);
cost = -log10(P  + epsilon);


end