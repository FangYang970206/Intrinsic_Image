function d = dssim(a,b)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
d = (1-ssim(a,b)) /2;
end

