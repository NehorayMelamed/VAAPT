function [align_seq1,sig] = Gimbaless_ctypi(sequence,dwn_smpl_ratio)
n = size(sequence,3);

% seq. resize (binning)
b = fir1(32,1/dwn_smpl_ratio/2);
for ii=1:n
    seq(:,:,ii) = imresize(conv2(b,b,sequence(:,:,ii),'valid'),1/dwn_smpl_ratio);
end
% seq = imresize3(sequence,round(size(sequence).*[1/dwn_smpl_ratio 1/dwn_smpl_ratio 1]));

sig = zeros(n,2);

% b = fir1(10,.05);
for ii=1:n-1
%     a1 = conv2(b,b,seq(:,:,ii),'valid');
%     a2 = conv2(b,b,seq(:,:,ii+1),'valid');
%     sig(ii+1,:) = typi_v2_1fr_gimbaless(a1,a2)*dwn_smpl_ratio;
    sig(ii+1,:) = typi_v2_1fr_gimbaless(seq(:,:,ii),seq(:,:,ii+1))*dwn_smpl_ratio;
end
sig = bsxfun(@minus,cumsum(sig),mean(cumsum(sig)));

align_seq1 = align_seq(sequence,sig,'fft');
% align_fr = mean(align_seq1,3);

end


function align_seq = align_seq(seq,sig,spline_or_fft)

if strcmp(spline_or_fft,'spline')
    align_seq = zeros(size(seq));
    [x,y] = meshgrid(1:size(seq,1),1:size(seq,2));
    for ii=1:n
        align_seq(:,:,ii) = real(interp2(seq(:,:,ii),...
            x + sig(ii,1),y + sig(ii,2),'spline',1i));
    end
else
    persistent kx ky;
    if isempty(kx) || (size(kx,1)~=size(seq,1) || size(kx,2)~=size(seq,2))
        [kx, ky] = meshgrid(ifftshift(0:size(seq,2)-1)/size(seq,2)*2i*pi,...
            ifftshift(0:size(seq,1)-1)/size(seq,1)*2i*pi);
    end
    sig_dim3 = permute(sig,[3 2 1]);
    SEQ = fft2(seq);
    SHIFT_OP = exp(kx.*sig_dim3(1,1,:) + ...
        ky.*sig_dim3(1,2,:) ...
        -1i*pi*sum(sig_dim3));
    align_seq = real(ifft2(SEQ.*SHIFT_OP));
end

end


function d = typi_v2_1fr_gimbaless(A,B,params)
% version 2 of the alg.
% Find  shift between two images A and B using alg. that is based on first
    % order of taylor series of the diff-image and pseudo inverse of the
    % two gradient images with spectral correction of the diff image
% Input:
    % params.dif_len = length of low-pass filter that corrects the grad. low-pass (odd)
% output: d - is a two enrties vector of the disp. that the alg. found. the
    % first entry relates to the vertical dimension and the second relates
    % to the horizontal dimension

if ~exist('params','var')
    params.dif_len = 7;
end
% creat filter
persistent c_grad
if size(c_grad,1)==0
    fprintf('\n----------typi_v2: create filter----------\n');
    c_grad = creat_filt(params);
end

% the pseudo inverse step
[px, py]=gradient(A/2+B/2);
px=px(4:end-3,4:end-3);py=py(4:end-3,4:end-3);
p=[px(:) py(:)];
% low-pass correction of the image due to the gradient operator
ABtx = conv2(1,c_grad,A-B,'same');
ABty = conv2(c_grad,1,A-B,'same');
ABtx=ABtx(4:end-3,4:end-3);ABty=ABty(4:end-3,4:end-3);
% shift find using all steps
d=inv(p'*p)*[ABtx(:)'*px(:);ABty(:)'*py(:)];

end

function c_grad = creat_filt(params)
L = 2^10;
c_grad=ifft(fftshift(sin(((-L/2:L/2-1)+eps)'/2/L*4*pi)./(((-L/2:L/2-1)+eps)'/2/L*4*pi)));
c_grad = c_grad([end-params.dif_len/2+1.5:end 1:params.dif_len/2+.5]);
c_grad = c_grad'/sum(c_grad);
end