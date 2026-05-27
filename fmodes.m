% -------------------------------------------------------------------------- 
% This m-file is part of the VCSEL simulation package VISTAS advanced v.3.02
% --------------------------------------------------------------------------
% computes the eigenvalue equation for the calculation of the LP-modes of a 
% weakly guided step-indes waveguide
%
% -------------------------------------------------------------------------- 
% Copyright (C) 2002  Marc Jungo
%
% This program is free software; you can redistribute it and/or modify it 
% under the terms of the GNU General Public License as published by the Free 
% Software Foundation.
% This program is distributed in the hope that it will be useful, but 
% WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY 
% or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for 
% more details.
% You should have received a copy of the GNU General Public License along with
% this program (gpl.txt); if not, write to the Free Software Foundation, Inc.,
% 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
%
% --------------------------------------------------------------------------
% Swiss Federal Institute of Technology (ETH) Zurich                        
% Laboratory for Electromagnetic Fields and Microwave Electronics           
% Communication Photonics Group                                           
% Gloriastr. 35  /  CH-8092 Zurich  /  Switzerland                     
% e-mail: jungo@photonics.ee.ethz.ch                                              
% homepage: http://www.ifh.ee.ethz.ch/~jungo                                
% ------------------------------------------------------------------------- 

%function fmodes=f(x,l,v)
%y=sqrt(v^2-x.^2);
%fmodes=x.*besselj(l+1,x)./besselj(l,x)-y.*besselk(l+1,y)./besselk(l,y); %Bessel function

function res = fmodes(x, l, v)
    % Evaluates the characteristic equation for the LP modes
    
    y = sqrt(v^2 - x.^2);
    
    % Bessel function calculations
    res = x .* besselj(l+1, x) ./ besselj(l, x) - y .* besselk(l+1, y) ./ besselk(l, y);
    
end