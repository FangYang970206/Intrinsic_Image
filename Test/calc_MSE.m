function res = calc_MSE( img_GT,img_Res,mask )
%UNTITLED 此处显示有关此函数的摘要
%  计算图像的mes
    img_GT=im2double(img_GT);
    img_Res=im2double(img_Res);
    num=sum(sum(sum((mask.*255)>0)))
    mask=double(mask);
    [w,h,c] = size(img_Res);
    square = (img_Res-img_GT).*(img_Res-img_GT);
    square=square.*mask;

    res= sum(sum(sum(square)))/num;
    return ;
end

