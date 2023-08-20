function sig = ctypi_v6_fst_vel(vid)

params.reg_step=.1/4;
params.filt_len=7;
params.dif_ord=4;
params.dif_len=7;

for ii=1:size(vid,3)-1
    sig(ii,:) = ctypi_v6_1fr(vid(:,:,ii+1),vid(:,:,ii),params);
%     ii/size(vid,3)
end

% mvid = mean(vid,3);
% for ii=1:size(vid,3)
%     sig(ii,:) = ctypi_v6_1fr(vid(:,:,ii),mvid,params);
%     ii/size(vid,3)
% end

end

function d = ctypi_v6_1fr(A,B,params)
% version 6 of the ctypi alg.
% This function Finds the shift between two images A and B using three steps alg. 
    % the first step is subpixel shifting of one of the frames using correlation filter 
        % (this shift makes low-pass filter that is applied on the second frame using another filter)
    % the second step is based on first order of taylor series of the diff-image 
        % and pseudo inverse of the two gradient images (x and y).
    % * the diff. operator makes some low-pass filter this is corrected
        % with correlation filter on the diff. between the two frames.
% Input:
    % dx0 = input appr. of the x-shift
    % dy0 = input appr. of the y-shift
    % params.reg_step = steps of pre-reg. ( = 0.5/n, where n is natural number larger than 1. n = 2 or 3 or 4 ...)
    % params.filt_len = length of reg. filts. (odd)
% output: 
    % d - is a two enrties vector of the shifts that the alg. founds
        % the first entry is the x-direction shift of the B frame about the A frame
        % the second entry is the y-direction shift of the B frame about the A frame
        % the positive x direction is right and the positive y direction is down
% without setting params the default is as follows:
    % params.reg_step=.1/4;
    % params.filt_len=15;

persistent c_shift reg_lst
if size(c_shift,1)==0
    fprintf('\n----------typi_v6: create filter ----------\n');
    params.filt_len=13;[c_shift, reg_lst] = creat_filt(params);
end

ddy0r = 0;
ddx0r = 0;

% choosing the filters according to D alg results
[~, ilowx]=min(abs(abs(ddx0r)-reg_lst));
[~, ilowy]=min(abs(abs(ddy0r)-reg_lst));
% choose which frame is moved and which gets the non-moving filter
clpx = c_shift.lp{ilowx}(circshift(1:2,ceil(sign(ddx0r)/2)),:);
clpy = c_shift.lp{ilowy}(circshift(1:2,ceil(sign(ddy0r)/2)),:);
cdx = c_shift.d{ilowx}(circshift(1:2,ceil(sign(ddx0r)/2)),:);
cdy = c_shift.d{ilowy}(circshift(1:2,ceil(sign(ddy0r)/2)),:);
% large shift correction
Alp=conv2(clpy(1,:),clpx(1,:),A,'same');
Blp=conv2(clpy(2,:),clpx(2,:),B,'same');
Adx=conv2(clpy(1,:),cdx(1,:),A,'same');
Bdx=conv2(clpy(2,:),cdx(2,:),B,'same');
Ady=conv2(cdy(1,:),clpx(1,:),A,'same');
Bdy=conv2(cdy(2,:),clpx(2,:),B,'same');


% Alp0y=circshift(conv2(clpy(1,:),1,A,'same'),[dy0r dx0r]);
% Blp0y=conv2(clpy(2,:),1,B,'same');
% Alpx0=circshift(conv2(1,clpx(1,:),A,'same'),[dy0r dx0r]);
% Blpx0=conv2(1,clpx(2,:),B,'same');
% Adx=conv2(1,cdx(1,:),Alp0y,'same');
% Bdx=conv2(1,cdx(2,:),Blp0y,'same');


% the pseudo inverse step
px = Bdx/4+Adx/4;
py = Bdy/4+Ady/4;
% low-pass correction of the image due to the gradient operator
ABtx = Alp-Blp;% or; ABtx = conv2(1,c_dif.lp,At-Bt,'same');

% removing the edges of the images len.
cutl = round(size(c_shift.lp{2},2)/4);% or; cutl = round((length(c_dif.lp)+size(c_shift{2},2))/8);
ABtx=ABtx(1+cutl:end-cutl,1+cutl:end-cutl);
px=px(1+cutl:end-cutl,1+cutl:end-cutl);py=py(1+cutl:end-cutl,1+cutl:end-cutl);
p=[px(:) py(:)];

% shift find using all steps
d = -p\ABtx(:);

end

function [c_shift, reg_lst] = creat_filt(params)

if isfield(params,'filt_len');filt_len=params.filt_len;
else; filt_len=11; end

if ~isfield(params,'reg_step');reg_step=.1/3;
else; reg_step=params.reg_step; end

reg_lst=0:reg_step:.5;

if mod(filt_len,2)==0;filt_len=filt_len+1;end
L = max(2^12,2^(nextpow2(filt_len)+6));
lv = -floor(filt_len/2):floor(filt_len/2);
% c_shift = cell(length(reg_lst),1);

k=fftshift(2i*pi*(-.5:1/L:.5-1/L));
% mov_k = @(d) exp(k*d).*exp(-abs(k).^8/(2*pi*.4)^8);
mov_k = @(d) exp(k*d);
mov = @(d) fftshift(ifft(mov_k(d)));
mov_d = @(d) fftshift(ifft(mov_k(d).*fft([0 1 zeros(1,L-3) -1])));
mov_lp = @(d) fftshift(ifft(mov_k(d).*sinc(fftshift(-1:2/L:1-2/L))));

for ii=1:length(reg_lst)
    ymov = real(mov(reg_lst(ii)/2));
    yd = real(mov_d(reg_lst(ii)/2));
    ylp = real(mov_lp(reg_lst(ii)/2));
    clp_for = ylp(L/2+1+lv)/sum(ylp(L/2+1+lv));
    clp_back = clp_for(end:-1:1);
    c_shift.lp{ii}=[clp_for' clp_back']';
    cd_for = yd(L/2+1+lv)/sum(ymov(L/2+1+lv));
    cd_back = -cd_for(end:-1:1);
    c_shift.d{ii}=[cd_for' cd_back']';
end

end
