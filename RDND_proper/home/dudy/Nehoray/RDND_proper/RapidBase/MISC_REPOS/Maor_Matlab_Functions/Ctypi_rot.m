function [sig] = Ctypi_rot(Sequence)

[c_shift, c_dif] = CreatShiftFilter;
for ii=2:size(Sequence,3)
    sig(ii,:) = ctypi_rot_v7(Sequence(:,:,1),Sequence(:,:,ii),c_shift,c_dif)';
end
sig= bsxfun(@minus,sig,mean(sig));

end

function d = ctypi_rot_v7(A,B,c_shift,c_dif,CorW,dYExp,dXExp,reg_step,filt_len,dif_ord,dif_len)
% version 7 of the alg.
% This function Finds the shift between two images A and B using three steps alg. 
    % the first step of the alg. is based simple correlation search and it is done with D alg.
    % the second step is subpixel shifting of one of the frames using correlation filter 
        % (this shift makes low-pass filter that is applied on the second frame using another filter)
    % the third step is based on first order of taylor series of the diff-image 
        % and pseudo inverse of the two gradient images (x and y).
    % * the diff. operator makes some low-pass filter this is corrected
        % with correlation filter on the diff. between the two frames.
% Input:
    % c_shift = shifting filters c_shift{n}(2,:) is the n'th shifting filter
        % (according to reg_step) and c_shift{n}(1,:) is the n'th corector filter
        % where one frame gets the first filter and the other one gets the second (for each axis)
    % c_dif = diff. filter (c_dif.d) and and corrector filter (c_dif.lp)
    % dYExp = the expected shift in full pixels (round number: ... -2,-1,0,1,2 ...)
    % dXExp = the expected shift in full pixels (round number: ... -2,-1,0,1,2 ...)
    % CorW = Correlation window (full width)
    % reg_step = steps of pre-reg. ( = 0.5/n, where n is natural number larger than 1. n = 2 or 3 or 4 ...)
    % filt_len = length of reg. filts. (odd)
    % dif_ord = ord. of the grad. (2,4,6 or 8)
    % dif_len = length of low-pass filter that corrects the grad. low-pass (odd)
% output: 
    % d - is a two enrties vector of the shifts that the alg. founds
        % the first entry is the x-direction shift of the B frame about the A frame
        % the second entry is the y-direction shift of the B frame about the A frame
        % the positive x direction is right and the positive y direction is down
        
% Examples:
% option (1)
% d = ctypi_v7(A,B); % takes alot of time
% automatically creates filters with the following params.: 
% reg_step=.1;filt_len=11;dif_ord=2;dif_len=7; and set: CorW=7;dYExp = 0;dXExp = 0;
% option (2)
% reg_step=.1;filt_len=13;dif_ord=2;dif_len=7;
% d = ctypi_v7(A,B,[],[],[],[],[],reg_step,filt_len,dif_ord,dif_len);
% automatically creates the filters and set: CorW=7;dYExp = 0;dXExp = 0;
% option (3)
% reg_step=.1;filt_len=13;dif_ord=2;dif_len=7;CorW=7;dYExp = 1;dXExp = -1;
% d = ctypi_v7(A,B,[],[],CorW,dYExp,dXExp,reg_step,filt_len,dif_ord,dif_len);
% automatically creates the filters and set: CorW=7;dYExp = 0;dXExp = 0;
% option (4)
% reg_step=.1;filt_len=13;dif_ord=2;dif_len=7;reg_lst=0:reg_step:.5;
% [c_shift, c_dif] = CreatShiftFilter(reg_lst,filt_len,dif_ord,dif_len);
% d = ctypi_v7(A,B,c_shift,c_dif);
% * Note: if one of the params. is missing it is complited with the automatically

if ~exist('CorW') || size(CorW,1)==0;CorW=7;end
if ~exist('dYExp') || size(dYExp,1)==0;dYExp = 0;end
if ~exist('dXExp') || size(dXExp,1)==0;dXExp = 0;end

if ~exist('c_shift') || size(c_shift,1)==0 || ~exist('c_dif') || size(c_dif,1)==0
    % create filters
    if ~exist('filt_len') || size(filt_len,1)==0;filt_len=11;end
    if ~exist('dif_ord') || size(dif_ord,1)==0;dif_ord=2;end
    if ~exist('dif_len') || size(dif_len,1)==0;dif_len=7;end
    if ~exist('reg_step') || size(reg_step,1)==0;reg_step=.1;end
    reg_lst=0:reg_step:.5;
    [c_shift, c_dif] = CreatShiftFilter(reg_lst,filt_len,dif_ord,dif_len);
else
    reg_lst=0:.5/round(-.5/((-floor(size(c_shift{2},2)/2):floor(size(c_shift{2},2)/2))*c_shift{2}(1,:)')):.5;
end

% % the correlation search step
% % while true
% %     d1=D1(A,B,CorW,-dYExp,dXExp);
% %     if abs(abs(d1.YDispAll)-abs(dYExp))==(CorW-1)/2 || abs(abs(d1.XDispAll)-abs(dXExp))==(CorW-1)/2
% %         dYExp = round(d1.YDispAll);
% %         dXExp = round(d1.XDispAll);
% %     elseif d1.YDispAll==0 || d1.XDispAll==0
% %         CorW=2*CorW+1;
% %     else
% %         break
% %     end
% % end
% % % the positive y direction in the D alg. is defined in the opposite direction than here
% 
% % dy0=d1.YDispAll;dx0=d1.XDispAll;

dy0=0;dx0=0;
dy0r = round(dy0);ddy0r = dy0 - dy0r;
dx0r = round(dx0);ddx0r = dx0 - dx0r;

% choosing the filters according to D alg results
[~, ilowx]=min(abs(abs(ddx0r)-reg_lst));
[~, ilowy]=min(abs(abs(ddy0r)-reg_lst));
% choose which frame is moved and which gets the non-moving filter
clowx = c_shift{ilowx}(circshift(1:2,ceil(sign(ddx0r)/2)),:);
clowy = c_shift{ilowy}(circshift(1:2,ceil(sign(ddy0r)/2)),:);
dshcr = reg_lst([ilowx ilowy]).*sign([ddx0r ddy0r]);
% large shift correction
At=circshift(conv2(clowy(1,:),clowx(1,:),A,'same'),[dy0r dx0r]);
Bt=conv2(clowy(2,:),clowx(2,:),B,'same');

% the pseudo inverse step
py = conv2(c_dif.d,1,At/2+Bt/2,'same');
px = conv2(1,c_dif.d,At/2+Bt/2,'same');
% low-pass correction of the image due to the gradient operator
ABtx = conv2(1,c_dif.lp,At-Bt,'same');
ABty = conv2(c_dif.lp,1,At-Bt,'same');

cutl = round((length(c_dif.lp)+size(c_shift{2},2))/4); % removing the edges of the images len.
ABtx=ABtx(1+cutl:end-cutl,1+cutl:end-cutl);ABty=ABty(1+cutl:end-cutl,1+cutl:end-cutl);
px=px(1+cutl:end-cutl,1+cutl:end-cutl);py=py(1+cutl:end-cutl,1+cutl:end-cutl);
[x,y]=meshgrid(((-size(A,1)/2+.5+cutl):(size(A,1)/2-.5-cutl)),...
    ((-size(A,2)/2+.5+cutl):(size(A,2)/2-.5-cutl)));
p=[px(:) py(:) px(:).*y(:)-py(:).*x(:)];

% shift find using all steps
d = inv(p'*p)*[ABtx(:)'*px(:);ABty(:)'*py(:);(ABtx(:)'*(px(:).*y(:))-ABty(:)'*(py(:).*x(:)))]+[dshcr';0]+[dx0r;dy0r;0];

end

function Ord1 = D1(A,B,CorW,dYExp,dXExp)

if ~exist('dYExp','var');dYExp=0;end
if ~exist('dXExp','var');dXExp=0;end

Parabola1_Gaus2 = 1;
CorWv = CorW ; 

[XDisp , YDisp , SelfDispX , SelfDispY ,CorrtMtx, OutRange , TemplateSiz] = CalcDisp_by_2DSmartSearch_in(A,B,CorWv,dYExp,dXExp,Parabola1_Gaus2) ;  
     
Ord1.XDispAll  = XDisp  ; 
Ord1.YDispAll  = -YDisp  ; 
  
Ord1.SelfDispXAll = SelfDispX ;
Ord1.SelfDispYAll = SelfDispY ;
 
Ord1.CorrtMtxAll  = CorrtMtx(1:CorW,1:CorW) ; 
Ord1.OutRangeAll = OutRange ; 
 
Ord1.TemplateSizAll = TemplateSiz ;
end
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5%%%%%%
function [XDisp , YDisp  ,SelfDispX , SelfDispY, CorrtMtx, OutRange , TemplateSiz ] = CalcDisp_by_2DSmartSearch_in(A,B,CorW,dYExp,dXExp,Parabola1_Gaus2)
% Parabola1_Gaus2 = 1 or 2 (fit to parabola if ==1 ; or fit to gaussian if ==2 );
% For on the move scenarion 
% CorW = 5,7,9 ...
%dYExp = 1, 0 ,-1 ,-2 ... 
 ParMaxVal = 0.3 ; % Calc shifts between A and B,  only if max correlation value > "ParMaxVal"
 % if correlation value < ParMaxVal then :  OutRange=2 , XDisp=0 ,YDisp=0
 %%A =A(2:end,1:end) ; 
 %%B =B(2:end,1:end) ;  
 
[CorrtMtx , SelfCorrtMtx , TemplateSiz] = Calc_CorrMat_SS_in(A,B,CorW,dYExp,dXExp) ;
% [CorrtMtx , SelfCorrtMtx , TemplateSiz] = Calc_CorrMat_SS_in(A,B,CorW+4,dYExp,dXExp) ;
%    TemplateSiz=TemplateSiz+4;
% CorrtMtx = conv2(CorrtMtx,fspecial('gaussian',5,1),'valid');
MxV = max(CorrtMtx(:)) ; 
MxVself = max(SelfCorrtMtx(:)) ; 
My_eps = 1e-10 ;  

CorrtMtxB       = CorrtMtx ; 
SelfCorrtMtxB = SelfCorrtMtx ; 

if Parabola1_Gaus2==2
    CorrtMtxB(CorrtMtxB<0) = My_eps ; 
    SelfCorrtMtxB(SelfCorrtMtxB<0) = My_eps ; 
    CorrtMtxB = log(CorrtMtxB) ;
    SelfCorrtMtxB = log(SelfCorrtMtxB) ;
end 
 
SelfParNan = isnan(SelfCorrtMtxB(1,1)) ;

  if SelfParNan==1
    XDisp =  0 ; 
    YDisp =  0 ;  
    OutRange = 2 ; 
    SelfDispX = 0 ;
    SelfDispY = 0 ;
  else 

   if MxVself>ParMaxVal
     [SelfDispX , SelfDispY , OutRange] = Calc_Disp_byCorrMtx_in(SelfCorrtMtxB,0,0) ;
     
     SelfParNan = isnan(SelfDispX*SelfDispY) ;

       if SelfParNan==1
          [SelfDispX , SelfDispY , OutRange] = Calc_Disp_byCorrMtx_in(SelfCorrtMtx,0,0) ;
      end
    
   else
       SelfDispX = 0 ; 
       SelfDispY = 0 ; 
       OutRange = 3 ; 
   end
   
 
        
   if MxV>ParMaxVal 

     [XDisp , YDisp , OutRange]             = Calc_Disp_byCorrMtx_in(CorrtMtxB,dYExp,dXExp);
       
     ParNan = isnan(XDisp*YDisp) ;
           if ParNan==1
              [XDisp , YDisp , OutRange]    = Calc_Disp_byCorrMtx_in(CorrtMtx,dYExp,dXExp);
           end

   else 
       XDisp =  0 ; 
       YDisp =  0 ;  
       OutRange = 3 ; 
   end
  
  end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [XDisp , YDisp , OutRange] = Calc_Disp_byCorrMtx_in(CorrtMtx,dYExp,dXExp) 
% Up = "+"
% Right = "+"
dXExp = round(dXExp);
dYExp = round(dYExp);

CorW = size(CorrtMtx,2); 
OutRange=0 ; 
[iiM , jjM] = find(CorrtMtx==max(CorrtMtx(:)),1) ; 
        
if iiM==size(CorrtMtx,1) || iiM==1 ; 
    OutRange = 1 ;
    YDisp =  ceil(CorW/2)  - iiM + dYExp ; 
else
   CorY = [ CorrtMtx(iiM-1,jjM)  , CorrtMtx(iiM,jjM) , CorrtMtx(iiM+1,jjM) ] ; 
   YsubPix = (CorY(3) - CorY(1)) / (2*(CorY(1) - 2*CorY(2)+CorY(3)))  ; 
   YDisp = YsubPix + ceil(CorW/2)  - iiM + dYExp ; 
end


if jjM==size(CorrtMtx,2) || jjM==1 ; 
    OutRange = 1 ;
    XDisp =  - ceil(CorW/2) + jjM + dXExp ; 
else
  CorX = [ CorrtMtx(iiM,jjM-1)  , CorrtMtx(iiM,jjM) , CorrtMtx(iiM,jjM+1) ] ; 
  XsubPix = (CorX(1) - CorX(3)) / (2*(CorX(1) - 2*CorX(2)+CorX(3)))  ; 
  XDisp = XsubPix - ceil(CorW/2) + jjM + dXExp ; 
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [CorrtMtx , SelfCorrtMtx , TempSiz] = Calc_CorrMat_SS_in(A,B,CorW,dYExp,dXExp) 
hCorW = floor(CorW/2) ; 
[XVecIndOrig , YVecIndOrig]  = Get_Template_Index_in(dYExp,dXExp,CorW,A) ; 
   
TempA = A(YVecIndOrig,XVecIndOrig) ; 
TempSiz = size(TempA) ;
CorrtMtx = Corr_Loop(hCorW,XVecIndOrig,YVecIndOrig,dXExp,dYExp,TempA,B) ;

SelfCorrtMtx = Corr_Loop(1,XVecIndOrig,YVecIndOrig,0,0,TempA,A) ; % Self corr-mtx for speckle noise reduction
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ CorrtMtx ] = Corr_Loop(hCorW,XVecIndOrig,YVecIndOrig,dXExp,dYExp,Temp,B) 
Temp0 = Temp - mean(Temp(:)) ; 
Tnorm = norm(Temp0,'fro') ; % same as :  Anorm = sqrt (sum(sum(SpWAn.^2)) ) ;
 Dw = 2*hCorW+1 ; 
CorrtMtx = zeros(Dw,Dw) ; 
if Tnorm~=0
IndY = 1 ; 
for ii = -hCorW:1:hCorW
    IndX = 1 ;  
    for jj =  -hCorW:1:hCorW
        XVecInd = XVecIndOrig + dXExp + jj   ; 
        YVecInd = YVecIndOrig - dYExp +  ii   ;  
        SpWB = B(YVecInd,XVecInd) ; 
        SpWBn = SpWB - mean(SpWB(:)) ;  
        Bnorm =  norm(SpWBn,'fro') ;  
        if Bnorm==0
            Bnorm=1 ; 
        end
        CorrtMtx(IndY,IndX) = sum(sum(Temp0.*SpWBn))  / (Tnorm*Bnorm)  ; 
        IndX = IndX+1 ; 
    end
        IndY = IndY+1 ; 
end

 else
     
 end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [XVecIndOrig , YVecIndOrig]  = Get_Template_Index_in(dYExp,dXExp,CorW,A)
% Find Max Template size for search in +-(CorW/2) around dYExp and dXExp
dYExp = round(dYExp) ;
dXExp = round(dXExp) ;

OutRange = 0 ; 
hCorW = floor(CorW/2) ; 
Nx = size(A,2) ; 
Ny = size(A,1) ; 

% Choose Smart Template from Matrix A
SpWL = 1     +  max(hCorW-dXExp,0) ;  
SpWR = min( Nx-( hCorW + dXExp ), Nx )  ; 

SpWU = 1     +  max(hCorW+dYExp,0) ;  
SpWD = min( Ny-( hCorW - dYExp ), Ny )  ; 
%%%%%%%%%%%%%%%%%%%%%%%%%%

%Check that at least 1pixel of margin is present, for self-corr calculations (see "SelfCorrtMtx" below)
SpWL = max([2 SpWL]) ; 
SpWR = min([(Nx-1) , SpWR]) ; 
SpWU = max([2 , SpWU]) ; 
SpWD = min([(Ny-1) , SpWD]) ; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
XVecIndOrig = SpWL:SpWR  ; 
YVecIndOrig = SpWU:SpWD  ; 
end

function [c_shift, c_dif, reg_lst] = CreatShiftFilter(reg_step,filt_len,dif_ord,dif_len)
% this is a simple code for ceating the filters of sub pxl shift and spectral correction of the gradient oprator.
% related to versions 4-8
% reg_step is the shift of the "moving" filter we would like to creat
    % reg_step = steps of pre-reg. ( = 0.5/n, where n is natural number larger than 1. n = 2 or 3 or 4 ...)
% filt_len - is the length of the filters (forced to be odd)
% dif_ord - ord. of the grad. (2,4,6 or 8)
% dif_len - length of low-pass filter that corrects the grad. low-pass (odd)
% Examples:
% option (1)
% [c_shift, c_dif, reg_lst] = CreatShiftFilter;
% default params.: reg_step=.1;filt_len=11;dif_ord=2;dif_len=7;
% option (2)
% reg_step=.1;
% [c_shift, c_dif, reg_lst] = CreatShiftFilter(reg_step);
% default params.: filt_len=11;dif_ord=2;dif_len=7;
% option (3)
% reg_step=.1;dif_ord=2;dif_len=7;
% [c_shift, c_dif, reg_lst] = CreatShiftFilter(reg_step,[],dif_ord,dif_len);
% default params.: filt_len=11;
% option (4)
% reg_step=.1;filt_len=11;dif_ord=2;dif_len=7;
% [c_shift, c_dif, reg_lst] = CreatShiftFilter(reg_step,filt_len,dif_ord,dif_len);

if ~exist('filt_len') || size(filt_len,1)==0;filt_len=11;end
if ~exist('reg_step') || size(reg_step,1)==0;reg_step=.1;end
if ~exist('dif_ord') || size(dif_ord,1)==0;dif_ord=2;end
if ~exist('dif_len') || size(dif_len,1)==0;dif_len=7;end
reg_lst=0:reg_step:.5;

if mod(filt_len,2)==0;filt_len=filt_len+1;end
L = max(2^12,2^(nextpow2(filt_len)+6));
lv = -floor(filt_len/2):floor(filt_len/2);
iix=1:L;
x=zeros(L,1);x(1)=1;x=fftshift(x);
for ii=1:length(reg_lst)
    y1=interp1(x,iix+reg_lst(ii),'spline');
    c_low_mov = y1(L/2+1+lv)'-(lv*y1(L/2+1+lv)'+reg_lst(ii))*lv'/filt_len^2;
    for jj=1:14
        c_low_mov = (c_low_mov*filt_len^2-(lv*c_low_mov+reg_lst(ii))*lv')/filt_len^2;
        c_low_mov = c_low_mov/sum(c_low_mov);
    end
    x1=zeros(L,1);x1(1:length(c_low_mov))=c_low_mov;
    x2=ifft(abs(fft(x1)));c_low_dc=x2([end-floor(filt_len/2)+1:end 1:ceil(filt_len/2)]);
    c_low_dc = c_low_dc/sum(c_low_dc);
    c_shift{ii}=[c_low_mov';c_low_dc'];
end

x=zeros(L,1);if nargin<3;dif_ord=2;end
switch dif_ord
    case 2
        x(1:2)=[.5 .5];
        c_dif.d = -[-1 0 1]/2;
    case 4
        x(1:4)=[-1 8 8 -1]/14;
        c_dif.d = -[1 -8 0 8 -1]/12;
    case 6
        x(1:6)=[1 -9 45 45 -9 1]/74;
        c_dif.d = -[-1 9 -45 0 45 -9 1]/60;
    case 8
        x(1:8)=[-3 32 -168 672 672 -168 32 -3]/1066;
        c_dif.d = -[3 -32 168 -672 0 672 -168 32 -3]/420/2;
end
xlp=ifft(abs(fft(x)));
c_lp = xlp([end-dif_len/2+1.5:end 1:dif_len/2+.5]);
c_dif.lp = c_lp'/sum(c_lp);

end