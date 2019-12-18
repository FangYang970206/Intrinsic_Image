%origin_path='F:/DB_BOLD/test/orig'
reflect_path = 'D:/fangyang/intrinsic_by_fangyang/logs_shapenet/RIN_CosBF_VGG0.1_updateLR/refl_target/'
shading_path = 'D:/fangyang/intrinsic_by_fangyang/logs_shapenet/RIN_CosBF_VGG0.1_updateLR/shad_target/'
%result_origin_path='F:/DB_BOLD/test_result_RIN_attention_CosBF_VGG0.1/orig/';
result_reflect_path = 'D:/fangyang/intrinsic_by_fangyang/logs_shapenet/RIN_CosBF_VGG0.1_updateLR/refl_output/';
result_shading_path = 'D:/fangyang/intrinsic_by_fangyang/logs_shapenet/RIN_CosBF_VGG0.1_updateLR/shad_output/';
mask_path = 'D:/fangyang/intrinsic_by_fangyang/logs_shapenet/RIN_CosBF_VGG0.1_updateLR/mask/'

fileFolder = fullfile(reflect_path);

dirOutput = dir(fullfile(fileFolder,'*.png'));

fileNames = {dirOutput.name}';
len = length(fileNames); 
loss_1 = 0;
loss_2 = 0;
loss_shading = [];
loss_reflect = [];
for i=1:len
    im_reflect_path = strcat(reflect_path,fileNames(i));
    im_reflect_path = im_reflect_path{1};
    im_shading_path = strcat(shading_path,fileNames(i));
    im_shading_path = im_shading_path{1};
%     im_mask_path = strcat(mask_path,fileNames(i));
%     im_mask_path =im_mask_path{1};
    
    im_reflect_result_path = strcat(result_reflect_path,fileNames(i));
    im_reflect_result_path = im_reflect_result_path{1};
    im_shading_result_path = strcat(result_shading_path,fileNames(i));
    im_shading_result_path = im_shading_result_path{1};
    
    %读取原始图像
    reflect_img = imread(im_reflect_path);
%     reflect_img=imresize(reflect_img,[112 112],'bilinear');
    shading_img = imread(im_shading_path); 
%     shading_img=imresize(shading_img,[112 112],'bilinear');
%     shading_img=repmat(shading_img,1,1,3);
%     mask_img=im2double(imread(im_mask_path)); 
    %mask_img=imresize(mask_img,[112 112],'bilinear');
    mask_img = ones(size(res_reflect_img));

    %读取生成图像
    res_reflect_img = imread(im_reflect_result_path);
    res_shading_img = imread(im_shading_result_path);
%     reflect_mask = imread(im_mask_path);
%     res_reflect_img = res_reflect_img * mask_img;
%     res_shading_img = res_shading_img * mask_img;
%     reflect_mask = ones(size(res_reflect_img));
%     shading_mask = ones(size(res_shading_img));
    %mask_img=double(mask_img);
    %mask=mask_img./255;
%     temp_loss_1 = local_MSE(reflect_img,res_reflect_img);
%     temp_loss_1=calc_MSE(reflect_img,res_reflect_img,reflect_mask); %dssim
    temp_loss_1 = local_MSE(reflect_img, res_reflect_img, mask_img);
    loss_1 = loss_1 + temp_loss_1;
%     temp_loss_2=dssim(shading_img,res_shading_img,shading_mask);
    temp_loss_2 = local_MSE(shading_img, res_shading_img, mask_img); %dssim
    loss_2 = loss_2 + temp_loss_2;
    loss_shading = [loss_shading temp_loss_2];
    loss_reflect = [loss_reflect temp_loss_1];
    disp(i)
    disp(temp_loss_1)
    disp(temp_loss_2)
end
loss_1=loss_1/len;
loss_2=loss_2/len;