function sig = Dp(V,CorW,dYExp,dXExp)

if nargin==1
    CorW = 5;
    dYExp = 0;
    dXExp = 0;
end

% mV = mean(V,3);
mV = V(:,:,1);
sig = zeros(size(V,3)-1,2);
for ii=1:size(V,3)
    [sx,sy] = D1(V(:,:,ii),mV,CorW,dYExp,dXExp);
    sig(ii,:) = [sx,sy];
    ii/size(V,3)
end

end

function [sx,sy] = D1(A,B,CorW,dYExp,dXExp)

if ~exist('dYExp','var');dYExp=0;end
if ~exist('dXExp','var');dXExp=0;end

Parabola1_Gaus2 = 1;

[XDisp , YDisp] = CalcDisp_by_2DSmartSearch_in(A,B,CorW,dYExp,dXExp,Parabola1_Gaus2) ;  
     
sx  = XDisp  ; 
sy  = -YDisp  ; 

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