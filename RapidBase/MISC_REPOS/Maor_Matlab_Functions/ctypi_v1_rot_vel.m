function sig = ctypi_v1_rot_vel(vid)

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

% mvid = mean(vid,3);

[x,y]=meshgrid(((-size(vid,1)/2+1.5):(size(vid,1)/2-1.5)),...
    ((-size(vid,2)/2+1.5):(size(vid,2)/2-1.5)));

for ii=1:size(vid,3)-1
    m1m = vid(2:end-1,2:end-1,ii+1)-vid(2:end-1,2:end-1,ii);
    m1p = vid(:,:,ii+1)+vid(:,:,ii);
    [px,py]=gradient(m1p); 
    
    p = [reshape(px(2:end-1,2:end-1),[],1) reshape(py(2:end-1,2:end-1),[],1)...
        reshape(px(2:end-1,2:end-1).*x'-py(2:end-1,2:end-1).*y',[],1)];
    
    sig(ii,:) = p\m1m(:);

    ii/size(vid,3)
end

% sig = [px(:) py(:)]\(reshape(vid(:,:,1:end-1) - vid(:,:,2:end),n1*n2,n3-1));
% sig = double(sig');

end