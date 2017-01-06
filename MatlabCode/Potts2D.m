function [u, erriter, i, timet] = Potts2D (Ct, alpha, pars)

% Performing the continuous max-flow algorithm to solve the 
%   continuous min cut problem in 2D
%%------------------------------------------------------------------------
%   Inputs: 
%   1) Ct: cost function for each label i
%   2) alpha: penalty parameter to the total variation term.
%   3) pars: an array containing a set of values for initializing parameters.

%   Outputs: 
%   1) u: the final results u(x) in [0,1].
%   2) erriter: the error evaluation of each iteration
%   3) i: total number of iterations
%   4) timet: total computational time 
%%------------------------------------------------------------------------


%%% Based on the previous implementations of [1] and [2]: 
%%% [1] "A Continuous Max-Flow Approach to Potts Model", 2010.
%%%      by Yuan, J, Bae, E.,Tai, Boykov, Y.
%%% [2]  "Matlab implementation of continuous max flow variants"
%%%       Martin Rajchl, Imperial College London, 2015

%%%   The original algorithm was proposed in the following papers:
%   [1] Yuan, J.; Bae, E.;  Tai, X.-C. 
%       A Study on Continuous Max-Flow and Min-Cut Approaches 
%       CVPR, 2010
%   [2] Yuan, J.; Bae, E.; Tai, X.-C.; Boycov, Y.
%       A study on continuous max-flow and min-cut approaches. Part I: Binary labeling
%       UCLA CAM, Technical Report 10-61, 2010


if(nargin < 3)
    error('Not enough args. Exiting...');
end


% Initializing parameters
rows = pars(1);
cols = pars(2);
nlab = pars(3);
iterNum= pars(4);
beta = pars(5);
cc = pars(6);
steps = pars(7);
vol = rows*cols*nlab;


% Setting initial values 
%   u(x,i=1...nlab) is set to be an initial cut
    u = zeros(rows,cols,nlab, 'like', Ct);

%   the source flow field 'ps(x)'
    ps = zeros(rows,cols, 'like', Ct);

%    the nlab sink flow fields pt(x,i=1...nlab)
    pt = zeros(rows,cols,nlab, 'like', Ct);
    
    
% initialize the flow buffers for faster convergence
[ps,I] = min(Ct, [], 3);

for i=1:nlab
    pt(:,:,i) = ps;
    tmp = I == i;
    u(:,:,i) = tmp;
end


divp = zeros(rows,cols,nlab, 'like', Ct);
pp1 = zeros(rows, cols+1,nlab, 'like', Ct);
pp2 = zeros(rows+1, cols,nlab, 'like', Ct);

erriter = zeros(iterNum,1, 'like', Ct);

tic
for i = 1:iterNum
    
    pd = zeros(rows,cols, 'like', Ct);
    
    % update the flow fields within each layer i=1...nlab
    for k= 1:nlab
        
        % update the spatial flow field p(x,i) = (pp1(x,i), pp2(x,i)):
        % the following steps are the gradient descent step with steps as the
        % step-size.
        
        ud = divp(:,:,k) - (ps - pt(:,:,k) + u(:,:,k)/cc);
        pp1(:,2:cols,k) = steps*(ud(:,2:cols) - ud(:,1:cols-1)) + pp1(:,2:cols,k);
        pp2(2:rows,:,k) = steps*(ud(2:rows,:) - ud(1:rows-1,:)) + pp2(2:rows,:,k);
        
        % the following steps are the projection to make |p(x,i)| <= alpha(x)
        
        gk = sqrt((pp1(:,1:cols,k).^2 + pp1(:,2:cols+1,k).^2 +...
            pp2(1:rows,:,k).^2 + pp2(2:rows+1,:,k).^2)*0.5);
        
        gk = double(gk <= alpha(:,:,k)) + double(~(gk <= alpha(:,:,k))).*(gk ./ alpha(:,:,k));
        gk = 1 ./ gk;
        
        pp1(:,2:cols,k) = (0.5*(gk(:,2:cols) + gk(:,1:cols-1))).*pp1(:,2:cols,k);
        pp2(2:rows,:,k) = (0.5*(gk(2:rows,:) + gk(1:rows-1,:))).*pp2(2:rows,:,k);
        
        divp(:,:,k) = pp1(:,2:cols+1,k)-pp1(:,1:cols,k)+pp2(2:rows+1,:,k)-pp2(1:rows,:,k);
        
        % update the sink flow field pt(x,i)
        
        ud = - divp(:,:,k) + ps + u(:,:,k)/cc;
        pt(:,:,k) = min(ud, Ct(:,:,k));
        
        % pd: the sum-up field for the computation of the source flow field
        %      ps(x)
        
        pd = pd + (divp(:,:,k) + pt(:,:,k) - u(:,:,k)/cc);
        
    end
    
    % update the source flow ps
    ps = pd / nlab + 1 / (cc*nlab);
    
    % update the multiplier u
    erru_sum = 0;
    for k = 1:nlab
        erru = cc*(divp(:,:,k) + pt(:,:,k) - ps);
        u(:,:,k) = u(:,:,k) - erru;
        erru_sum = erru_sum + sum(sum(abs(erru)));
    end
    
    % evaluate the average error  
    erriter(i) = erru_sum/vol;
    
    if erriter(i) < beta
        break;
    end
    
end

if(strcmp(class(u), 'gpuArray'))
    u = gather(u);
    erriter = gather(erriter);
end

timet = toc;

% display the total number of iterations
msg = sprintf('number of iterations = %u; time = %f \n', i, timet);
disp(msg);


end