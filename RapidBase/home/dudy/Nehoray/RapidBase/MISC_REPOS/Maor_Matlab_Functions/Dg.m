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