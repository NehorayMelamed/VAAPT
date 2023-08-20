function sig = ctypi_v1_vel_mean_fr(vid)

if ~isa(vid(1),'double'); vid = double(vid); end

n3 = size(vid,3);

% % [n1, n2, n3] = size(vid,3);
% mm = reshape(vid(2:end-1,2:end-1,2:end)-vid(2:end-1,2:end-1,1:end-1),[],n3-1);
% mp = vid(2:end-1,2:end-1,2:end)+vid(2:end-1,2:end-1,1:end-1);
% [px,py]=gradient(mp);
% px = reshape(px,[],n3-1); py = reshape(py,[],n3-1);
% pxpx = sum(px.*px); pypy = sum(py.*py); pxpy = sum(px.*py);
% pxpydet = pxpx.*pypy-pxpy.^2;
% pxmm = sum(px.*mm); pymm = sum(py.*mm);
% 
% 
% sig(:,1) = (pypy .* pxmm - pxpy .* pymm) ./ pxpydet;
% sig(:,2) = (pxpx .* pymm - pxpy .* pxmm) ./ pxpydet;

% % mm = reshape(vid(2:end-1,2:end-1,2:end)-vid(2:end-1,2:end-1,1:end-1),[],n3-1);
% % [px,py]=gradient(mean(vid,3)); 
% % % [px,py]=gradient(vid(:,:,1));
% % px = px(2:end-1,2:end-1); py = py(2:end-1,2:end-1);
% % p = [px(:) py(:)]; pt = inv(p'*p)*p';
% % 
% % sig = (pt*mm)';


% [px,py]=gradient(mean(vid,3)); 
[px,py]=gradient(vid(:,:,1));
px = px(2:end-1,2:end-1); py = py(2:end-1,2:end-1);
p = [px(:) py(:)]; pt = inv(p'*p)*p';
for ii=1:n3-1
    m1m = vid(2:end-1,2:end-1,ii+1)-vid(2:end-1,2:end-1,ii);
%     m1p = vid(2:end-1,2:end-1,ii+1)+vid(2:end-1,2:end-1,ii);
    sig(ii,:) = pt*m1m(:);
end

end