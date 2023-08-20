function sig = ctypi_v6_vel(vid)

nn=0;
for ii=1:size(vid,3)-1
    for jj=1:nn; fprintf('\b'); end
    nn = fprintf('Ctypi v6 alg. prog.: %d%%',round(ii/(size(vid,3)-1)*100));
    
    [dx0, dy0]=Dg(vid(:,:,ii),vid(:,:,ii+1),0,0,7);
    params.reg_step=.1/4;
    params.filt_len=15;
    params.dif_ord=1;
    params.dif_len=7;
    sig(ii,:) = ctypi_v6_1fr(vid(:,:,ii+1),vid(:,:,ii),dx0,dy0,params)';
%     ii/size(vid,3)
end
fprintf('\n');
end

function d = ctypi_v6_1fr(A,B,dx0,dy0,params)
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

dy0r = round(dy0);ddy0r = dy0 - dy0r;
dx0r = round(dx0);ddx0r = dx0 - dx0r;

% choosing the filters according to D alg results
[~, ilowx]=min(abs(abs(ddx0r)-reg_lst));
[~, ilowy]=min(abs(abs(ddy0r)-reg_lst));
% choose which frame is moved and which gets the non-moving filter
clpx = c_shift.lp{ilowx}(circshift(1:2,ceil(sign(ddx0r)/2)),:);
clpy = c_shift.lp{ilowy}(circshift(1:2,ceil(sign(ddy0r)/2)),:);
cdx = c_shift.d{ilowx}(circshift(1:2,ceil(sign(ddx0r)/2)),:);
cdy = c_shift.d{ilowy}(circshift(1:2,ceil(sign(ddy0r)/2)),:);
dshcr = reg_lst([ilowx ilowy]).*sign([ddx0r ddy0r]);
% large shift correction
At = circshift(A,[dy0r dx0r]);
Alp=conv2(clpy(1,:),clpx(1,:),At,'same');
Blp=conv2(clpy(2,:),clpx(2,:),B,'same');
Adx=conv2(clpy(1,:),cdx(1,:),At,'same');
Bdx=conv2(clpy(2,:),cdx(2,:),B,'same');
Ady=conv2(cdy(1,:),clpx(1,:),At,'same');
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
d = -p\ABtx(:) + dshcr' + [dx0r;dy0r];

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

function [dx,dy,corr_max_est] =  Dg(sRoi_A,sRoi_B,dx_Exp,dy_Exp,CorW)

% sRoi_A = First frame
% sRoi_B = Second frame
% CorrRim = integer from 1 and above (1/2/3...) 
% dx_Exp = Expected dx, integer (...-3/-2/-1/0/1/2/3...) 
% dy_Exp = Expected dy, integer (...-3/-2/-1/0/1/2/3...) 
% sRoiW = sRoi width (left-right dimention)
% sRoiH = sRoi High  (up-down dimention) 

sRoiW=size(sRoi_A,1);sRoiH=size(sRoi_A,2);

CorrRim_x = (CorW-1)/2 ; 
CorrRim_y = (CorW-1)/2 ; 

 [T,B] = TB_in(sRoi_A,sRoi_B,dx_Exp,dy_Exp,CorrRim_x,CorrRim_y,sRoiW,sRoiH);

 CorrMtx = TBCorrMtx_in(T,B) ; 
 Weight_Power = 4;
 
  [xx_2d,yy_2d] = meshgrid(-CorrRim_x:CorrRim_x,-CorrRim_y:CorrRim_y);

 [dx,dy,corr_max_est,speckle_size] = ddShifts_byCorrMtx_AxGauss_5x5_ask_in(xx_2d,yy_2d,CorrMtx,Weight_Power);
 
end
function [T,B] = TB_in(sRoi_A,sRoi_B,dx_Exp,dy_Exp,CorrRim_x,CorrRim_y,sRoiW,sRoiH)
% sRoi_A = First frame
% sRoi_B = Second frame
% CorrRim = integer from 1 and above (1/2/3...) 
% dx_Exp = Expected dx, integer (...-3/-2/-1/0/1/2/3...) 
% dy_Exp = Expected dy, integer (...-3/-2/-1/0/1/2/3...) 
% sRoiW = sRoi width (left-right dimention)
% sRoiH = sRoi High  (up-down dimention) 
   
a =  (CorrRim_x+1)- dx_Exp ; 
b =  1 ; 
X1 = max(a,b) ; 

a =  (CorrRim_y+1)- dy_Exp ; 
Y1 = max(a,b) ; 

a =  sRoiW - CorrRim_x - dx_Exp ; 
b =  sRoiW ; 
X2 = min(a,b) ;  

a =  sRoiH - CorrRim_y -dy_Exp ; 
b =  sRoiH ; 
Y2 = min(a,b) ;  

T = sRoi_A(X1:X2,Y1:Y2) ; 
Xb1 = X1 + dx_Exp - CorrRim_x ; 
Yb1 = Y1 + dy_Exp - CorrRim_y ; 
Xb2 = X2 + dx_Exp + CorrRim_x ; 
Yb2 = Y2 + dy_Exp + CorrRim_y ; 

B = sRoi_B(Xb1:Xb2,Yb1:Yb2) ; 
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function CorrMtx = TBCorrMtx_in(T,B)
%calc correlation by scalar product
%single pixel resloved steps,
% Inputs:
% T (Template)
% B (Base)
%
% Outputs:
% CorrMtx
%with average substraction
% coder.varsize('T',[512 640]);
% coder.varsize('B',[512 640]);

[ntx,nty] = size(T) ;
[nbx,nby] = size(B) ;

delx = nbx-ntx+1 ;
dely = nby-nty+1 ;

mt = mean(T(:)) ;

TvN = T(:)-mt ; 
st = sum(TvN.^2) ;

CorrMtx=zeros(delx,dely);
for iix=1:1:delx
    for iiy=1:1:dely
        b = B((1:1:ntx)+iix-1,(1:1:nty)+iiy-1) ;
        mb = mean(b(:)) ;
        BvN = b(:)-mb ; 
        sb = sum(BvN.^2) ;
        CorrMtx(iix,iiy) = sum(TvN.*BvN) /sqrt(sb*st) ;
    end
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [dx,dy,corr_max_est,speckle_size] = ddShifts_byCorrMtx_AxGauss_5x5_ask_in(xx_2d,yy_2d,CorrMtx,Weight_Power)
%calc sub pixel shifts by axial gauss algorithm
% Inputs:
% CorrMtx (normalized correlation function)
% Outputs:
% dx sub pixel x shift
% dy sub pixel y shift
% use only 5x5 matric around  maximal correlation
% take into account positive correlations only
% the final results depends on the coordinated supplied with the input
% for example:
%if the input is arranged so that the center of the correlation is
%(x,y)=(0,0), and the shift is symmetrical, the calculated shift is zero


%create sub matrix
[iim,jjm]=find(CorrMtx == max(CorrMtx(:)),1);
% iim = 6;
% jjm = 6;
%create submatrix around the correlation maxima, preferably 5x5, if possible
ii_s=max(1,iim-2);
ii_e=min(size(CorrMtx,1),iim+2);
jj_s=max(1,jjm-2);
jj_e=min(size(CorrMtx,2),jjm+2);

xx_2d_sub=xx_2d(ii_s:1:ii_e,jj_s:1:jj_e);
yy_2d_sub=yy_2d(ii_s:1:ii_e,jj_s:1:jj_e);
CorrMtx_sub=CorrMtx(ii_s:1:ii_e,jj_s:1:jj_e);

%check if correlation matrix is reasonable check local maxima, and allow 
%only one
%locate and null local maxima
[nn,mm]=size(CorrMtx_sub);
CorrMtx_fr=zeros(size(CorrMtx_sub)+2);
CorrMtx_fr(2:end-1,2:end-1)=CorrMtx_sub;
local_iden=(CorrMtx_fr(3:end,2:end-1) < CorrMtx_fr(2:end-1,2:end-1))+...
    (CorrMtx_fr(2:end-1,3:end) < CorrMtx_fr(2:end-1,2:end-1))+...
     (CorrMtx_fr(1:end-2,2:end-1) < CorrMtx_fr(2:end-1,2:end-1))+...
      (CorrMtx_fr(2:end-1,1:end-2) < CorrMtx_fr(2:end-1,2:end-1));
%allow one maxima
cmax=max(CorrMtx_sub(:));
[iimx,iimy]=find(CorrMtx_sub == cmax);
iimx=iimx(1);
iimy=iimy(1);
local_iden(iimx,iimy)=0;
ii_iden=find(or(local_iden == 4,CorrMtx_sub <= 0));
[ii_iden_x,ii_iden_y]=find(local_iden == 4);
CorrMtx_mod=CorrMtx_sub;
%find the minimal distance between modified pixels to maxima
if (length(ii_iden) > 0)
    CorrMtx_mod(ii_iden)=0;
    max_dis=ones(length(ii_iden),1);
    dii_iden_x=abs(ii_iden_x-iimx);
    dii_iden_y=abs(ii_iden_y-iimy);
    for kk=1:1:length(ii_iden_x),
        if (dii_iden_x(kk) == 0)
            max_dis(kk)=dii_iden_y(kk);
        elseif (dii_iden_y(kk) == 0)
            max_dis(kk)=dii_iden_x(kk);
        else
            max_dis(kk)=min(dii_iden_x(kk),dii_iden_y(kk));
        end
    end
    max_dis_min=min(max_dis);
    %    calculate new xGrid,yGrid
    xs=max(1,iimx-max_dis_min);
    xe=min(size(CorrMtx_mod,1),iimx+max_dis_min);
    ys=max(1,iimy-max_dis_min);
    ye=min(size(CorrMtx_mod,2),iimy+max_dis_min);
    
    xx_2d_sub=xx_2d_sub(xs:xe,ys:ye);
    yy_2d_sub=yy_2d_sub(xs:xe,ys:ye);
    CorrMtx_sub=CorrMtx_mod(xs:xe,ys:ye);
end

ii_pos=find(CorrMtx_sub > 0);

%check if there are enough points for extracting 4 parameters
ValidVals = CorrMtx_sub > 0;
ColSum = sum(ValidVals,1);
RowSum = sum(ValidVals,2);

if (length(ii_pos) > 4 & max(RowSum)>1 & max(ColSum)>1)
    
    xx_pos=xx_2d_sub(ii_pos);
    yy_pos=yy_2d_sub(ii_pos);
    x2py2_pos=xx_pos.^2+yy_pos.^2;
    cc_pos=log(CorrMtx_sub(ii_pos));
    
    wei=diag(CorrMtx_sub(ii_pos).^Weight_Power,0);
    mm=[x2py2_pos(:) xx_pos(:) yy_pos(:) ones(length(xx_pos(:)),1)];
    
    numerator_p=mm'*wei*mm;
    denominator_p=mm'*wei*cc_pos(:);
    vv=numerator_p\denominator_p;
    
    corr_max_est=exp(vv(4)-((vv(2)^2+vv(3)^2)/(4*vv(1))));
    var_sq=log(2)/(-vv(1));
    if (var_sq > 0)
        speckle_size=2*sqrt(var_sq);
    else
        speckle_size = 0;
    end
    dx=vv(2)/(-2*vv(1));
    dy=vv(3)/(-2*vv(1));
    
    if (dx < min(xx_2d(:)))
        dx=min(xx_2d(:));
    elseif (dx > max(xx_2d(:)))
        dx=max(xx_2d(:));
    end
    
    if (dy < min(yy_2d(:)))
        dy=min(yy_2d(:));
    elseif (dy > max(yy_2d(:)))
        dy=max(yy_2d(:));
    end
    
else
    corr_max_est=max(CorrMtx_sub(:));
    speckle_size = 0;
    dx=0;
    dy=0;
end
end