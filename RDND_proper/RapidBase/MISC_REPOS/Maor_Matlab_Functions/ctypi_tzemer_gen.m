function [sig, w, fr_sz_lst] = ctypi_tzemer_gen(v,fr_sz_lst,ver3flg)

sz = size(v);
if ~exist('fr_sz_lst','var') || isempty(fr_sz_lst); fr_sz_lst = 2.^(4:8); end
if ~exist('ver3flg','var') || isempty(ver3flg); ver3flg = 0; end
for ifr_sz = 1:length(fr_sz_lst)
    fr_sz = fr_sz_lst(ifr_sz);
    if mod(size(v,1),fr_sz)~=0; error('fr_sz doesn''t suit image sz'); end
    if mod(size(v,2),fr_sz)~=0; error('fr_sz doesn''t suit image sz'); end
    sig{ifr_sz} = zeros(sz(3)-1,2,sz(1)/fr_sz,sz(2)/fr_sz);
    w{ifr_sz} = zeros(sz(1)/fr_sz,sz(2)/fr_sz);
end

if ver3flg
    lp1 = [.011684807659500685...
        -.027972794566202423...
        .223900559315387264...
        .5847748551826291038...
        .223900559315387264...
        -.027972794566202423...
        .011684807659500685];
    lp2 = int16(round(lp1*72));
    vx = lp2(1)*circshift(v,[-3 0 0])+...
        lp2(2)*circshift(v,[-2 0 0])+...
        lp2(3)*circshift(v,[-1 0 0])+...
        lp2(4)*v+...
        lp2(5)*circshift(v,[1 0 0])+...
        lp2(6)*circshift(v,[2 0 0])+...
        lp2(7)*circshift(v,[3 0 0]);
    vy = lp2(1)*circshift(v,[0 -3 0])+...
        lp2(2)*circshift(v,[0 -2 0])+...
        lp2(3)*circshift(v,[0 -1 0])+...
        lp2(4)*v+...
        lp2(5)*circshift(v,[0 1 0])+...
        lp2(6)*circshift(v,[0 2 0])+...
        lp2(7)*circshift(v,[0 3 0]);
end
vt = diff(v,[],3);
v2 = v(:,:,1:end-1) + v(:,:,2:end);
clear v
px = circshift(v2,[1 0 0]) - circshift(v2,[-1 0 0]);
py = circshift(v2,[0 1 0]) - circshift(v2,[0 -1 0]);
clear v2

nn=0;
for ii=1:sz(3)-1
    for jj=1:nn; fprintf('\b'); end
    nn = fprintf('Ctypi tzemer gen. alg. prog.: %d%%',round(ii/sz(3)*100));
    pxii = double(px(:,:,ii)); 
    pyii = double(py(:,:,ii)); 
    vtii = double(vt(:,:,ii));
    for ifr_sz = 1:length(fr_sz_lst)
        fr_sz = fr_sz_lst(ifr_sz);
        for isz1 = 0:fr_sz:sz(1)-1
            for isz2 = 0:fr_sz:sz(2)-1
                ind1 = max(3,isz1+1):min(sz(1)-2,isz1+fr_sz);
                ind2 = max(3,isz2+1):min(sz(2)-2,isz2+fr_sz);
                if rank(reshape([pxii(ind1,ind2) pyii(ind1,ind2)],[],2))==2
                    sig{ifr_sz}(ii,:,isz1/fr_sz+1,isz2/fr_sz+1) = ...
                        reshape([pxii(ind1,ind2) pyii(ind1,ind2)],[],2)\...
                        double(reshape(vtii(ind1,ind2),[],1));
                end
            end
        end
    end
end
fprintf('\n');

w = calc_w(mean(px,3),mean(py,3),fr_sz_lst);

% w_pxy = (mean(px,3).^2 + mean(py,3).^2).^.5; 
% for ifr_sz = 1:length(fr_sz_lst)
%     fr_sz = fr_sz_lst(ifr_sz);
%     for isz1 = 0:fr_sz:sz(1)-1
%         for isz2 = 0:fr_sz:sz(2)-1
%             ind1 = max(3,isz1+1):min(sz(1)-2,isz1+fr_sz);
%             ind2 = max(3,isz2+1):min(sz(2)-2,isz2+fr_sz);
%             w{ifr_sz}(isz1/fr_sz+1,isz2/fr_sz+1) = sum(w_pxy(ind1,ind2),'all')/fr_sz^2;
%         end
%     end
% end

end

