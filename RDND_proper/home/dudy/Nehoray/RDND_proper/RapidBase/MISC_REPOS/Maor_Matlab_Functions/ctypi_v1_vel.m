function sig = ctypi_v1_vel(vid)

if ~isa(vid(1),'double'); vid = double(vid); end

n3 = size(vid,3);
% 
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

% mm = reshape(vid(2:end-1,2:end-1,2:end)-vid(2:end-1,2:end-1,1:end-1),[],n3-1);
% [px,py]=gradient(mean(vid,3)); 
% px = px(2:end-1,2:end-1); py = py(2:end-1,2:end-1);
% p = [px(:) py(:)]; pt = inv(p'*p)*p';
% 
% sig = (pt*mm)';


sig = zeros(n3-1,2);
for ii=1:n3-1
    m1m = vid(2:end-1,2:end-1,ii+1)-vid(2:end-1,2:end-1,ii);
    m1p = vid(:,:,ii+1)+vid(:,:,ii);
    [px,py]=gradient(m1p); 
    p = [reshape(px(2:end-1,2:end-1),[],1) reshape(py(2:end-1,2:end-1),[],1)]/2;
    
    sig(ii,:) = p\m1m(:);
%     ii/n3
end

end