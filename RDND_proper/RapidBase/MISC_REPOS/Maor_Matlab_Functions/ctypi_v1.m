function sig = ctypi_v1(vid,mvid)

% if ~isa(vid,'double') && ~isa(vid,'single')
%     vid = single(vid);
% end
% 
% [n1, n2, n3] = size(vid);
% if ~exist('mvid','var'); mvid = mean(vid,3); end
% 
% [px, py] = gradient(mvid);
% sig = [px(:) py(:)]\(reshape(vid - mvid,n1*n2,n3));
% sig = (double(sig'));

mvid = mean(vid,3);
% mvid = vid(:,:,1);
[px, py] = gradient(mvid); p = [px(:) py(:)];
for ii=1:size(vid,3)
    v = double(vid(:,:,ii))-mvid;
    sig(ii,:) = p\v(:);
    ii/size(vid,3)
end

% sig = [px(:) py(:)]\(reshape(vid(:,:,1:end-1) - vid(:,:,2:end),n1*n2,n3-1));
% sig = double(sig');

end