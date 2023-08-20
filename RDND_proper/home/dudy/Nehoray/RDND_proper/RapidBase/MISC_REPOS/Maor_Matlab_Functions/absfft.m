function [X, f, ph, F] = absfft(x,fs,dim,n)
if size(x,1)==1 && ~exist('dim') && size(size(x),2)<3;x=x';end
if nargin<3 || isempty(dim); dim = 1; end
xsz = size(x); l = xsz(dim); 
if nargin<4; n = prod(l); end
for idim = 1:length(dim)
    x = fftshift(fft(x,l(idim),dim(idim)),dim(idim));
end
F = x/n;
X = abs(F);
if nargout>2; ph = angle(F); end
if nargout>1
    if nargin<2; fs = 1; end
    if mod(n,2)
        f = (-(.5-1/n/2):1/n:(.5-1/n/2))*fs;
    else
        f = (-.5:1/n:.5-1/n)*fs;
    end
end
end

